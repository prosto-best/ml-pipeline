"""FastAPI-сервис инференса для курса CNY/RUB (юань/рубль).

Эндпоинты:
    GET  /health           -- liveness/readiness проба для Kubernetes
    POST /predict           -- предсказание log-return курса на следующий торговый день

Модель и список фичей загружаются один раз при старте процесса (не на
каждый запрос) -- важно для latency и для количества обращений к MOEX ISS.
"""
import json
import logging

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import ServeConfig
from data_ingestion import load_ohlcv
from feature_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="CNY/RUB Predictor", version="1.0.0")

cfg = ServeConfig()
_model = None
_feature_meta = None


class PredictRequest(BaseModel):
    ticker: str | None = None  # secid на MOEX, по умолчанию CNYRUB_TOM
    lookback_days: int = 400  # сколько дней истории тянуть для расчёта индикаторов


class PredictResponse(BaseModel):
    ticker: str
    predicted_log_return: float
    predicted_direction: str
    last_rate: float
    predicted_rate_estimate: float


@app.on_event("startup")
def load_artifacts():
    global _model, _feature_meta
    logger.info("Loading model from %s", cfg.model_path)
    _model = joblib.load(cfg.model_path)
    with open(cfg.features_path) as f:
        _feature_meta = json.load(f)
    logger.info(
        "Model loaded. Trained for ticker=%s source=%s horizon=%s",
        _feature_meta["ticker"],
        _feature_meta.get("source", cfg.default_source),
        _feature_meta["horizon_days"],
    )


@app.get("/health")
def health():
    ready = _model is not None
    return {"status": "ok" if ready else "loading"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Модель ещё не загружена")

    ticker = req.ticker or cfg.default_ticker
    source = _feature_meta.get("source", cfg.default_source)
    board = _feature_meta.get("board", cfg.default_board)

    try:
        raw = load_ohlcv(
            ticker,
            start_date=_start_date_for_lookback(req.lookback_days),
            source=source,
            board=board,
        )
        # build_features нужен только чтобы получить feature_cols; сам расчёт
        # для инференса делаем ниже отдельно, чтобы не терять последнюю строку
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    feature_cols = _feature_meta["feature_cols"]
    latest_row = _latest_features_for_inference(raw, feature_cols)

    pred = float(_model.predict(latest_row)[0])
    last_rate = float(raw["Close"].iloc[-1])
    direction = "up" if pred > 0 else "down"
    predicted_rate_estimate = float(last_rate * np.exp(pred))

    return PredictResponse(
        ticker=ticker,
        predicted_log_return=pred,
        predicted_direction=direction,
        last_rate=last_rate,
        predicted_rate_estimate=predicted_rate_estimate,
    )


def _start_date_for_lookback(lookback_days: int) -> str:
    from datetime import datetime, timedelta

    # Берём с запасом (х1.6), т.к. lookback_days -- торговые дни, а валютный
    # рынок MOEX торгует почти все будние дни (меньше праздничных пропусков,
    # чем на фондовом рынке)
    start = datetime.today() - timedelta(days=int(lookback_days * 1.6))
    return start.strftime("%Y-%m-%d")


def _latest_features_for_inference(raw, feature_cols):
    """Строит признаки без dropna по target -- нужна именно последняя строка."""
    from feature_engineering import build_features as _bf

    # horizon_days=0 эквивалентно "таргет = текущий день": target не NaN,
    # поэтому dropna не убирает последнюю (самую свежую) строку
    featured, _ = _bf(raw, [1, 2, 3, 5, 10, 20], [5, 10, 20, 50], 14, horizon_days=0)
    return featured[feature_cols].iloc[[-1]]
