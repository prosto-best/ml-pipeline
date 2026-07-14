"""Обучение LightGBM-модели предсказания log-return цены акции.

Использование:
    python src/train.py --ticker AAPL --start 2015-01-01

Сохраняет в ARTIFACTS_DIR:
    - model.joblib      обученная модель
    - features.json     список признаков + метаданные (тикер, horizon)
    - metrics.json       метрики качества на hold-out
"""
import argparse
import json
import logging
import os

import joblib
import lightgbm as lgb
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


def train(cfg: TrainConfig) -> dict:
    raw = load_ohlcv(cfg.ticker, cfg.start_date, cfg.end_date)
    featured, feature_cols = build_features(
        raw, cfg.lag_windows, cfg.ma_windows, cfg.rsi_window, cfg.horizon_days
    )

    train_df, test_df = time_series_split(featured, cfg.test_size)
    logger.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

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

    # Directional accuracy: часто важнее абсолютной ошибки для трейдинг-сигналов
    direction_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))

    metrics = {
        "ticker": cfg.ticker,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": direction_acc,
        "best_iteration": model.best_iteration,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    logger.info("Metrics: %s", metrics)

    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    joblib.dump(model, os.path.join(cfg.artifacts_dir, cfg.model_filename))

    with open(os.path.join(cfg.artifacts_dir, cfg.feature_list_filename), "w") as f:
        json.dump(
            {
                "feature_cols": feature_cols,
                "ticker": cfg.ticker,
                "horizon_days": cfg.horizon_days,
            },
            f,
            indent=2,
        )

    with open(os.path.join(cfg.artifacts_dir, cfg.metrics_filename), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker")
    parser.add_argument("--start")
    parser.add_argument("--end", default="")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.ticker:
        cfg.ticker = args.ticker
    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end

    result = train(cfg)
    print(json.dumps(result, indent=2))
