"""FastAPI-сервис инференса.

Эндпоинты:
    GET  /health           -- liveness/readiness проба для Kubernetes
    POST /predict           -- предсказание log-return на следующий торговый день

Модель и список фичей загружаются один раз при старте процесса (не на
каждый запрос) — важно для latency и для количества обращений к диску.
"""
import json
import logging

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import ServeConfig
from data_ingestion import load_ohlcv
from feature_engineering import build_features, get_latest_feature_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Price Predictor", version="1.0.0")

cfg = ServeConfig()
_model = None
_feature_meta = None


class PredictRequest(BaseModel):
    ticker: str | None = None
    lookback_days: int = 400  # сколько дней истории тянуть для расчёта индикаторов


class PredictResponse(BaseModel):
    ticker: str
    predicted_log_return: float
    predicted_direction: str
    last_close: float
    predicted_close_estimate: float


@app.on_event("startup")
def load_artifacts():
    global _model, _feature_meta
    logger.info("Loading model from %s", cfg.model_path)
    _model = joblib.load(cfg.model_path)
    with open(cfg.features_path) as f:
        _feature_meta = json.load(f)
    logger.info("Model loaded. Trained for ticker=%s horizon=%s", _feature_meta["ticker"], _feature_meta["horizon_days"])


@app.get("/health")
def health():
    ready = _model is not None
    return {"status": "ok" if ready else "loading"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Модель ещё не загружена")

    ticker = req.ticker or cfg.default_ticker
    try:
        raw = load_ohlcv(ticker, start_date=_start_date_for_lookback(req.lookback_days))
        featured, _ = build_features(
            raw,
            lag_windows=[1, 2, 3, 5, 10, 20],
            ma_windows=[5, 10, 20, 50],
            rsi_window=14,
            horizon_days=_feature_meta["horizon_days"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    feature_cols = _feature_meta["feature_cols"]
    # build_features дропает последние horizon_days строк (нет таргета) —
    # для инференса на "сегодня" нам нужна строка ДО такого дропа, поэтому
    # пересчитываем признаки отдельно на полном сырье без завязки на target-dropna.
    latest_row = _latest_features_for_inference(raw, feature_cols)

    pred = float(_model.predict(latest_row)[0])
    last_close = float(raw["Close"].iloc[-1])
    direction = "up" if pred > 0 else "down"
    predicted_close_estimate = float(last_close * np.exp(pred))

    return PredictResponse(
        ticker=ticker,
        predicted_log_return=pred,
        predicted_direction=direction,
        last_close=last_close,
        predicted_close_estimate=predicted_close_estimate,
    )


def _start_date_for_lookback(lookback_days: int) -> str:
    from datetime import datetime, timedelta

    # Берём с запасом (х2.5), т.к. lookback_days — торговые дни, а не календарные
    start = datetime.today() - timedelta(days=int(lookback_days * 2.5))
    return start.strftime("%Y-%m-%d")


def _latest_features_for_inference(raw, feature_cols):
    """Строит признаки без dropna по target — нужна именно последняя строка."""
    from feature_engineering import build_features as _bf

    # horizon_days=0 эквивалентно "таргет = текущий день", target не NaN => dropna не убирает последнюю строку
    featured, _ = _bf(raw, [1, 2, 3, 5, 10, 20], [5, 10, 20, 50], 14, horizon_days=0)
    return featured[feature_cols].iloc[[-1]]
