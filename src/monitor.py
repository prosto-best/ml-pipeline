"""Мониторинг качества уже задеплоенных моделей на живых данных.

Запускается периодически (в кластере -- как Kubernetes CronJob, см.
helm/cnyrub-predictor/templates/monitor-cronjob.yaml). Для каждого тикера,
для которого есть обученная модель в ARTIFACTS_DIR:

1. "Живое" качество (walk-forward). Модель предсказывает return на 1 день
   вперёд, значит через день мы уже ЗНАЕМ фактический результат. Скрипт
   берёт последние `monitoring_window_days` полностью реализованных дней,
   прогоняет по ним модель и сравнивает предсказания с реальностью --
   это честная оценка качества модели "в бою", а не на историческом
   hold-out при обучении. Результат сравнивается с тренировочным RMSE
   (metrics.json) -- если живой RMSE намного хуже, это сигнал деградации.

2. Data drift (Evidently). Сравнивает распределение признаков на свежих
   данных с baseline-сэмплом, сохранённым во время обучения
   (reference_features.csv). Существенный дрифт -- признак того, что
   рыночный режим изменился и модель, возможно, скоро устареет, даже если
   пока формально не ошибается сильно.

3. Результаты логируются в отдельный MLflow-эксперимент "мониторинга" (не
   смешиваются с экспериментами обучения) и пушатся в Prometheus Pushgateway
   как gauge-метрики -- чтобы в Grafana можно было построить график деградации
   качества и дрифта во времени и настроить алерты в Alertmanager.

Использование:
    python src/monitor.py
    python src/monitor.py --tickers CNYRUB_TOM
"""
import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from config import MonitorConfig
from data_ingestion import load_ohlcv
from feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _discover_tickers(cfg: MonitorConfig):
    pattern = os.path.join(cfg.artifacts_dir, "*", cfg.model_filename)
    return sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(pattern))


def _live_quality(ticker: str, cfg: MonitorConfig, model, meta: dict) -> dict:
    """Считает качество модели на последних monitoring_window_days реализованных днях."""
    # Запас данных с учётом окон индикаторов (MA50 и т.д.) + окно мониторинга
    lookback_days = int((cfg.monitoring_window_days + 80) * 1.6)
    start = (datetime.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    raw = load_ohlcv(ticker, start_date=start, source=cfg.source, board=cfg.board)
    horizon_days = meta.get("horizon_days", 1)
    featured, feature_cols = build_features(raw, [1, 2, 3, 5, 10, 20], [5, 10, 20, 50], 14, horizon_days)

    # build_features уже дропнул последние horizon_days строк (для них target
    # ещё не реализован) -- всё, что осталось, полностью реализовано.
    window = featured.tail(cfg.monitoring_window_days)
    if window.empty:
        raise ValueError(f"Недостаточно данных для окна мониторинга по {ticker}")

    X = window[feature_cols]
    y_true = window["target"]
    y_pred = model.predict(X)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    directional_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
        "n_samples": len(window),
        "current_features": featured[feature_cols],  # для drift-проверки ниже
    }


def _data_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Считает долю признаков с задетектированным дрифтом через Evidently."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError as e:
        logger.warning("Evidently недоступен (%s) -- пропускаю drift-проверку", e)
        return {"drift_share": None, "dataset_drift": None}

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current.tail(len(reference)) if len(current) > len(reference) else current)
    result = report.as_dict()

    drift_share = None
    dataset_drift = None
    for m in result.get("metrics", []):
        res = m.get("result", {})
        if "share_of_drifted_columns" in res:
            drift_share = res["share_of_drifted_columns"]
            dataset_drift = res.get("dataset_drift")
            break

    return {"drift_share": drift_share, "dataset_drift": dataset_drift}


def _log_to_mlflow(ticker: str, live: dict, drift: dict, degradation_ratio: float, cfg: MonitorConfig):
    try:
        import mlflow

        if cfg.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_monitoring_experiment_name)

        with mlflow.start_run(run_name=f"{ticker}-{datetime.today().strftime('%Y-%m-%d')}"):
            mlflow.set_tags({"ticker": ticker, "type": "monitoring"})
            mlflow.log_metrics({
                "live_rmse": live["rmse"],
                "live_mae": live["mae"],
                "live_directional_accuracy": live["directional_accuracy"],
                "degradation_ratio": degradation_ratio,
                **({"drift_share": drift["drift_share"]} if drift["drift_share"] is not None else {}),
            })
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] Не удалось залогировать мониторинг в MLflow: %s", ticker, e)


def _push_to_prometheus(all_results: dict, cfg: MonitorConfig):
    if not cfg.pushgateway_url:
        logger.info("PUSHGATEWAY_URL не задан -- пропускаю push метрик мониторинга")
        return

    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()
        g_rmse = Gauge("cnyrub_monitor_live_rmse", "Живой RMSE на реализованном окне", ["ticker"], registry=registry)
        g_dir_acc = Gauge(
            "cnyrub_monitor_live_directional_accuracy", "Живая directional accuracy", ["ticker"], registry=registry
        )
        g_degradation = Gauge(
            "cnyrub_monitor_degradation_ratio", "live_rmse / train_rmse", ["ticker"], registry=registry
        )
        g_drift = Gauge("cnyrub_monitor_drift_share", "Доля признаков с дрифтом", ["ticker"], registry=registry)

        for ticker, r in all_results.items():
            if "error" in r:
                continue
            g_rmse.labels(ticker=ticker).set(r["live_rmse"])
            g_dir_acc.labels(ticker=ticker).set(r["live_directional_accuracy"])
            g_degradation.labels(ticker=ticker).set(r["degradation_ratio"])
            if r.get("drift_share") is not None:
                g_drift.labels(ticker=ticker).set(r["drift_share"])

        push_to_gateway(cfg.pushgateway_url, job="cnyrub-monitor", registry=registry)
        logger.info("Метрики мониторинга запушены в Pushgateway: %s", cfg.pushgateway_url)
    except Exception as e:  # noqa: BLE001
        logger.warning("Не удалось запушить метрики в Pushgateway: %s", e)


def monitor_one(ticker: str, cfg: MonitorConfig) -> dict:
    ticker_dir = os.path.join(cfg.artifacts_dir, ticker)
    model = joblib.load(os.path.join(ticker_dir, cfg.model_filename))
    with open(os.path.join(ticker_dir, cfg.feature_list_filename)) as f:
        meta = json.load(f)
    with open(os.path.join(ticker_dir, cfg.metrics_filename)) as f:
        train_metrics = json.load(f)

    live = _live_quality(ticker, cfg, model, meta)
    degradation_ratio = live["rmse"] / train_metrics["rmse"] if train_metrics["rmse"] > 0 else float("inf")

    reference_path = os.path.join(ticker_dir, cfg.reference_filename)
    drift = {"drift_share": None, "dataset_drift": None}
    if os.path.exists(reference_path):
        reference = pd.read_csv(reference_path)
        drift = _data_drift(reference, live["current_features"])
    else:
        logger.warning("[%s] Нет reference_features.csv -- drift-проверка пропущена", ticker)

    result = {
        "ticker": ticker,
        "live_rmse": live["rmse"],
        "live_mae": live["mae"],
        "live_directional_accuracy": live["directional_accuracy"],
        "n_samples": live["n_samples"],
        "train_rmse": train_metrics["rmse"],
        "degradation_ratio": degradation_ratio,
        "drift_share": drift["drift_share"],
        "dataset_drift": drift["dataset_drift"],
    }

    alerts = []
    if degradation_ratio > cfg.max_degradation_ratio:
        alerts.append(
            f"деградация качества: live_rmse/train_rmse={degradation_ratio:.2f} > {cfg.max_degradation_ratio}"
        )
    if drift["drift_share"] is not None and drift["drift_share"] > cfg.max_drift_share:
        alerts.append(f"data drift: доля дрифтующих признаков={drift['drift_share']:.2f} > {cfg.max_drift_share}")
    result["alerts"] = alerts

    logger.info("[%s] %s", ticker, result)
    _log_to_mlflow(ticker, live, drift, degradation_ratio, cfg)

    return result


def main(cfg: MonitorConfig, tickers=None) -> int:
    tickers = tickers or _discover_tickers(cfg)
    if not tickers:
        logger.warning("Не найдено ни одной обученной модели в %s", cfg.artifacts_dir)
        return 0

    all_results = {}
    any_alert = False
    for ticker in tickers:
        try:
            all_results[ticker] = monitor_one(ticker, cfg)
            if all_results[ticker]["alerts"]:
                any_alert = True
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] Мониторинг упал: %s", ticker, e)
            all_results[ticker] = {"error": str(e)}

    _push_to_prometheus(all_results, cfg)

    print(json.dumps(all_results, indent=2, ensure_ascii=False, default=lambda o: None))

    # Не роняем job аварийно с ненулевым кодом на первом же алерте по
    # умолчанию -- в отличие от evaluate.py (quality gate ПЕРЕД деплоем),
    # это уже наблюдение за тем, что УЖЕ в проде, и должно долетать до
    # Prometheus/Grafana как метрика, а не молча падать. Но код возврата
    # всё равно ненулевой при алертах -- удобно смотреть в истории CronJob'а.
    # return 1 if any_alert else 0
    return 1 if any_alert and not os.getenv("AIRFLOW_TASK_ID") else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Список тикеров через запятую (по умолчанию -- все найденные в ARTIFACTS_DIR)")
    args = parser.parse_args()

    cfg = MonitorConfig()
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    sys.exit(main(cfg, tickers))
