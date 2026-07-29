"""FastAPI-сервис инференса для набора валютных пар MOEX.

На старте процесса сканирует artifacts_dir/<ticker>/ и загружает В ПАМЯТЬ
модель + метаданные признаков для КАЖДОГО найденного тикера (CNYRUB_TOM,
USD000UTSTOM, EUR_RUB__TOM и т.д. -- ровно те, что были обучены train.py).
Модели не перечитываются на каждый запрос -- это важно для latency.

Эндпоинты:
    GET /health    -- liveness/readiness проба для Kubernetes
    GET /predict    -- прогноз log-return на следующий торговый день СРАЗУ
                        по всем загруженным тикерам, одним JSON-объектом:

    {
      "CNYRUB_TOM": {
        "predicted_log_return": 0.0031,
        "predicted_direction": "up",
        "last_rate": 12.45,
        "predicted_rate_estimate": 12.4887
      },
      "USD000UTSTOM": { ... },
      "EUR_RUB__TOM": { ... }
    }

    Опционально можно ограничить ответ подмножеством тикеров:
    GET /predict?tickers=CNYRUB_TOM,USD000UTSTOM

    Если для конкретного тикера не удалось получить свежие данные (например,
    MOEX временно недоступен), в ответе по этому тикеру будет поле "error",
    а остальные тикеры всё равно вернутся нормально -- один упавший источник
    не должен обрушивать весь ответ.

    GET /metrics -- метрики в формате Prometheus:
        - стандартные HTTP-метрики (латентность, RPS, коды ответа) через
          prometheus-fastapi-instrumentator;
        - кастомные ML-гейджи: cnyrub_predictor_last_log_return{ticker=...},
          cnyrub_predictor_last_rate{ticker=...}, cnyrub_predictor_predict_errors_total{ticker=...} --
          обновляются при каждом вызове /predict, чтобы в Grafana можно было
          построить график "что сервис предсказывал во времени" и алертить
          на всплеск ошибок инференса по конкретному тикеру.
"""
import glob
import json
import logging
import os
from typing import Dict, Optional

import joblib
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel

from config import ServeConfig
from data_ingestion import load_ohlcv
from feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MOEX FX Rate Predictor", version="2.0.0")

cfg = ServeConfig()

# ticker -> {"model": ..., "meta": {...}}
_registry: Dict[str, dict] = {}

if cfg.metrics_enabled:
    from prometheus_client import Counter, Gauge
    from prometheus_fastapi_instrumentator import Instrumentator

    # Instrumentator сам вешает /metrics и стандартные HTTP-метрики
    # (request duration, requests total по коду ответа и т.д.)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    LAST_LOG_RETURN = Gauge(
        "cnyrub_predictor_last_log_return", "Последний предсказанный log-return", ["ticker"]
    )
    LAST_RATE = Gauge(
        "cnyrub_predictor_last_rate", "Последний известный курс на момент предсказания", ["ticker"]
    )
    PREDICT_ERRORS = Counter(
        "cnyrub_predictor_predict_errors_total", "Количество неудачных попыток предсказания", ["ticker"]
    )
else:
    LAST_LOG_RETURN = LAST_RATE = PREDICT_ERRORS = None


class TickerPrediction(BaseModel):
    predicted_log_return: Optional[float] = None
    predicted_direction: Optional[str] = None
    last_rate: Optional[float] = None
    predicted_rate_estimate: Optional[float] = None
    error: Optional[str] = None


@app.on_event("startup")
def load_all_models():
    """Находит все обученные модели под artifacts_dir/<ticker>/ и грузит их."""
    pattern = os.path.join(cfg.artifacts_dir, "*", cfg.model_filename)
    model_paths = sorted(glob.glob(pattern))

    if not model_paths:
        logger.warning("Не найдено ни одной модели по пути %s -- сервис поднимется, но /predict будет пустым", pattern)

    for model_path in model_paths:
        ticker_dir = os.path.dirname(model_path)
        ticker = os.path.basename(ticker_dir)
        features_path = os.path.join(ticker_dir, cfg.feature_list_filename)

        try:
            model = joblib.load(model_path)
            with open(features_path) as f:
                meta = json.load(f)
            _registry[ticker] = {"model": model, "meta": meta}
            logger.info("Loaded model for %s (source=%s)", ticker, meta.get("source", cfg.default_source))
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load model for %s: %s", ticker, e)

    logger.info("Total tickers loaded: %d -> %s", len(_registry), list(_registry.keys()))


@app.get("/health")
def health():
    return {"status": "ok" if _registry else "loading", "tickers_loaded": list(_registry.keys())}


@app.get("/predict", response_model=Dict[str, TickerPrediction])
def predict(
    tickers: Optional[str] = Query(
        None, description="Список тикеров через запятую. Если не указан -- отдаются все загруженные."
    ),
    lookback_days: int = Query(400, description="Сколько дней истории тянуть для расчёта индикаторов"),
):
    requested = [t.strip() for t in tickers.split(",")] if tickers else list(_registry.keys())

    response: Dict[str, TickerPrediction] = {}
    for ticker in requested:
        response[ticker] = _predict_one(ticker, lookback_days)

    return response


def _predict_one(ticker: str, lookback_days: int) -> TickerPrediction:
    entry = _registry.get(ticker)
    if entry is None:
        return TickerPrediction(error=f"Модель для тикера '{ticker}' не загружена на этом сервисе")

    meta = entry["meta"]
    model = entry["model"]
    source = meta.get("source", cfg.default_source)
    board = meta.get("board", cfg.default_board)

    try:
        raw = load_ohlcv(
            ticker,
            start_date=_start_date_for_lookback(lookback_days),
            source=source,
            board=board,
        )
        latest_row = _latest_features_for_inference(raw, meta["feature_cols"])
    except Exception as e:  # noqa: BLE001
        logger.error("Predict failed for %s: %s", ticker, e)
        if PREDICT_ERRORS is not None:
            PREDICT_ERRORS.labels(ticker=ticker).inc()
        return TickerPrediction(error=str(e))

    pred = float(model.predict(latest_row)[0])
    last_rate = float(raw["Close"].iloc[-1])
    direction = "up" if pred > 0 else "down"
    predicted_rate_estimate = float(last_rate * np.exp(pred))

    if LAST_LOG_RETURN is not None:
        LAST_LOG_RETURN.labels(ticker=ticker).set(pred)
        LAST_RATE.labels(ticker=ticker).set(last_rate)

    return TickerPrediction(
        predicted_log_return=pred,
        predicted_direction=direction,
        last_rate=last_rate,
        predicted_rate_estimate=predicted_rate_estimate,
    )


def _start_date_for_lookback(lookback_days: int) -> str:
    from datetime import datetime, timedelta

    start = datetime.today() - timedelta(days=int(lookback_days * 1.6))
    return start.strftime("%Y-%m-%d")


def _latest_features_for_inference(raw, feature_cols):
    """Строит признаки без dropna по target -- нужна именно последняя строка.

    horizon_days=0 эквивалентно "таргет = текущий день": таргет не NaN,
    поэтому dropna не убирает последнюю (самую свежую) строку.
    """
    featured, _ = build_features(raw, [1, 2, 3, 5, 10, 20], [5, 10, 20, 50], 14, horizon_days=0)
    return featured[feature_cols].iloc[[-1]]
