"""Обучение LightGBM-моделей предсказания log-return для набора валютных пар.

Для каждого тикера из cfg.tickers (по умолчанию CNYRUB_TOM, USD000UTSTOM,
EUR_RUB__TOM) обучается ОТДЕЛЬНАЯ модель -- разные валютные пары имеют разную
волатильность, ликвидность и зависимость от разных внешних факторов, поэтому
общая модель на всех сразу давала бы усреднённое и менее точное качество.

Каждый запуск логируется в MLflow (параметры, метрики, сама модель,
регистрация версии в Model Registry) -- если MLFLOW_TRACKING_URI не задан,
MLflow просто пишет всё в локальную папку ./mlruns, ничего не ломая.

Использование:
    python src/train.py                                  # обучить все тикеры из TICKERS
    python src/train.py --tickers CNYRUB_TOM              # обучить только один
    python src/train.py --tickers CNYRUB_TOM,USD000UTSTOM  # обучить несколько

Сохраняет для каждого тикера в ARTIFACTS_DIR/<ticker>/:
    - model.joblib               обученная модель (тот же файл serve.py читает напрямую)
    - features.json                список признаков + метаданные (тикер, источник, horizon)
    - metrics.json                  метрики качества на hold-out
    - reference_features.csv       сэмпл обучающих признаков -- baseline для
                                     data drift мониторинга (см. src/monitor.py)
"""
import argparse
import json
import logging
import os

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import TrainConfig
from data_ingestion import load_ohlcv
from feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def time_series_split(df, test_size: float):
    """Разбивка по времени: последние test_size% строк — hold-out.

    В отличие от случайного train_test_split, здесь модель никогда не видит
    будущее относительно теста — критично для корректной оценки на time series.
    """
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _setup_mlflow(cfg: TrainConfig) -> None:
    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    # Если tracking_uri не задан -- mlflow сам создаст локальную ./mlruns
    mlflow.set_experiment(cfg.mlflow_experiment_name)


def train_one(ticker: str, cfg: TrainConfig) -> dict:
    """Обучает и сохраняет модель для одного тикера. Возвращает метрики."""
    logger.info("=== Training ticker: %s ===", ticker)

    raw = load_ohlcv(ticker, cfg.start_date, cfg.end_date, source=cfg.source, board=cfg.board)
    featured, feature_cols = build_features(
        raw, cfg.lag_windows, cfg.ma_windows, cfg.rsi_window, cfg.horizon_days
    )

    train_df, test_df = time_series_split(featured, cfg.test_size)
    logger.info("[%s] Train rows: %d, Test rows: %d", ticker, len(train_df), len(test_df))

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    with mlflow.start_run(run_name=ticker) as run:
        mlflow.set_tags({
            "ticker": ticker,
            "source": cfg.source,
            "board": cfg.board,
            "horizon_days": cfg.horizon_days,
        })
        mlflow.log_params(cfg.lgb_params)
        mlflow.log_params({
            "num_boost_round": cfg.num_boost_round,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "test_size": cfg.test_size,
            "start_date": cfg.start_date,
            "n_features": len(feature_cols),
        })

        train_set = lgb.Dataset(X_train, label=y_train)
        valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

        model = lgb.train(
            cfg.lgb_params,
            train_set,
            num_boost_round=cfg.num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        preds = model.predict(X_test, num_iteration=model.best_iteration)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        direction_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))

        metrics = {
            "ticker": ticker,
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": direction_acc,
            "best_iteration": model.best_iteration,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }
        logger.info("[%s] Metrics: %s", ticker, metrics)

        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": direction_acc,
            "best_iteration": float(model.best_iteration),
        })

        # Модель + регистрация версии в Model Registry. try/except -- если
        # registry недоступен (например, backend store без поддержки), просто
        # логируем предупреждение, но не роняем обучение из-за телеметрии.
        try:
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                registered_model_name=cfg.registry_model_name(ticker),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] MLflow model logging/registration failed: %s", ticker, e)

        mlflow_run_id = run.info.run_id

    out_dir = cfg.artifacts_dir_for(ticker)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(model, os.path.join(out_dir, cfg.model_filename))

    with open(os.path.join(out_dir, cfg.feature_list_filename), "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "ticker": ticker,
                "source": cfg.source,
                "board": cfg.board,
                "horizon_days": cfg.horizon_days,
                "mlflow_run_id": mlflow_run_id,
                "mlflow_registered_model": cfg.registry_model_name(ticker),
            },
            f,
            indent=2,
        )

    with open(os.path.join(out_dir, cfg.metrics_filename), "w") as f:
        json.dump(metrics, f, indent=2)

    # Baseline для data drift-мониторинга: сэмпл признаков ИМЕННО из train-части
    # (не из hold-out и уж точно не из будущих данных) -- это распределение,
    # с которым Evidently потом будет сравнивать свежие "боевые" данные.
    reference_sample = X_train.sample(n=min(1000, len(X_train)), random_state=42)
    reference_sample.to_csv(os.path.join(out_dir, cfg.reference_filename), index=False)

    return metrics


def train_all(cfg: TrainConfig) -> dict:
    """Обучает модели для всех тикеров из cfg.tickers.

    Ошибка на одном тикере не должна останавливать обучение остальных --
    например, если у одной из пар временно недоступны данные на MOEX.
    Список успешных/упавших тикеров попадает в summary и в лог, а evaluate.py
    затем проверяет качество только успешно обученных моделей.
    """
    _setup_mlflow(cfg)

    summary = {"tickers": {}, "failed": []}
    for ticker in cfg.tickers:
        try:
            summary["tickers"][ticker] = train_one(ticker, cfg)
        except Exception as e:  # noqa: BLE001
            logger.error("Training failed for %s: %s", ticker, e)
            summary["failed"].append({"ticker": ticker, "error": str(e)})

    # Общий summary тоже полезно сохранить -- пригодится, например, для дашборда
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    with open(os.path.join(cfg.artifacts_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Список тикеров через запятую, например CNYRUB_TOM,USD000UTSTOM")
    parser.add_argument("--start")
    parser.add_argument("--end", default="")
    parser.add_argument("--source", choices=["moex", "yahoo"])
    parser.add_argument("--board")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.tickers:
        cfg.tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end
    if args.source:
        cfg.source = args.source
    if args.board:
        cfg.board = args.board

    result = train_all(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["failed"] and len(result["failed"]) == len(cfg.tickers):
        # Если обучение упало для ВСЕХ тикеров -- это точно провал пайплайна
        raise SystemExit(1)
