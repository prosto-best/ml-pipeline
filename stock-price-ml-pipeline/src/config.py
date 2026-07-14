"""Централизованная конфигурация пайплайна.

Значения можно переопределить переменными окружения — это удобно, когда
конфиг приходит из Kubernetes ConfigMap/Secret.
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainConfig:
    ticker: str = os.getenv("TICKER", "AAPL")
    start_date: str = os.getenv("START_DATE", "2012-01-01")
    end_date: str = os.getenv("END_DATE", "")  # пусто = до сегодня

    # На сколько торговых дней вперёд предсказываем return
    horizon_days: int = int(os.getenv("HORIZON_DAYS", "1"))

    # Доля данных на hold-out (по времени, не случайно — важно для time series)
    test_size: float = float(os.getenv("TEST_SIZE", "0.15"))

    # Гиперпараметры LightGBM
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
    })
    num_boost_round: int = int(os.getenv("NUM_BOOST_ROUND", "2000"))
    early_stopping_rounds: int = int(os.getenv("EARLY_STOPPING_ROUNDS", "100"))

    # Пороговое качество, ниже которого CI не собирает образ (см. evaluate.py)
    max_acceptable_rmse: float = float(os.getenv("MAX_ACCEPTABLE_RMSE", "0.05"))

    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    model_filename: str = "model.joblib"
    feature_list_filename: str = "features.json"
    metrics_filename: str = "metrics.json"

    lag_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    rsi_window: int = 14


@dataclass
class ServeConfig:
    model_path: str = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    features_path: str = os.getenv("FEATURES_PATH", "artifacts/features.json")
    default_ticker: str = os.getenv("TICKER", "AAPL")
    log_level: str = os.getenv("LOG_LEVEL", "info")
