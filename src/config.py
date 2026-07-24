"""Централизованная конфигурация пайплайна.

Значения можно переопределить переменными окружения — это удобно, когда
конфиг приходит из Kubernetes ConfigMap/Secret.

Пайплайн умеет работать сразу с несколькими валютными парами MOEX
(мультитикерный режим): для каждого тикера из списка `TICKERS` обучается
своя отдельная модель, а serving-сервис на старте подгружает все обученные
модели и по запросу к /predict отдаёт прогнозы по всем тикерам разом.
"""
import os
from dataclasses import dataclass, field
from typing import List


def _parse_tickers(raw: str) -> List[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


# Список валютных пар MOEX (борд CETS) по умолчанию:
#   CNYRUB_TOM     -- юань/рубль
#   USD000UTSTOM   -- доллар/рубль
#   EUR_RUB__TOM   -- евро/рубль
DEFAULT_TICKERS = "CNYRUB_TOM,USD000UTSTOM,EUR_RUB__TOM"


@dataclass
class TrainConfig:
    tickers: List[str] = field(default_factory=lambda: _parse_tickers(os.getenv("TICKERS", DEFAULT_TICKERS)))
    source: str = os.getenv("DATA_SOURCE", "moex")  # "moex" (рекомендуется) или "yahoo"
    board: str = os.getenv("MOEX_BOARD", "CETS")
    start_date: str = os.getenv("START_DATE", "2018-01-01")
    end_date: str = os.getenv("END_DATE", "")  # пусто = до сегодня

    # На сколько торговых дней вперёд предсказываем return
    horizon_days: int = int(os.getenv("HORIZON_DAYS", "1"))

    # Доля данных на hold-out (по времени, не случайно — важно для time series)
    test_size: float = float(os.getenv("TEST_SIZE", "0.15"))

    # Гиперпараметры LightGBM (общие для всех тикеров; вынесены отдельно от
    # тикера, потому что технические индикаторы и природа таргета одинаковы)
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

    # Базовая директория артефактов. Модель каждого тикера кладётся в
    # подпапку artifacts_dir/<ticker>/ — так несколько моделей не
    # перезаписывают друг друга.
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    model_filename: str = "model.joblib"
    feature_list_filename: str = "features.json"
    metrics_filename: str = "metrics.json"
    reference_filename: str = "reference_features.csv"  # baseline для data drift (Evidently)

    lag_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    rsi_window: int = 14

    # --- MLflow ---
    # Если MLFLOW_TRACKING_URI не задан, MLflow пишет всё в локальную папку
    # ./mlruns -- это рабочий fallback для локальной разработки и для CI
    # (можно приложить mlruns/ как build-артефакт), но для полноценного
    # трекинга и Model Registry нужен поднятый MLflow tracking server
    # (см. helm/cnyrub-predictor/templates/mlflow-*.yaml).
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "cnyrub-rate-predictor")
    mlflow_registry_model_name_prefix: str = os.getenv("MLFLOW_MODEL_PREFIX", "cnyrub-predictor")

    def registry_model_name(self, ticker: str) -> str:
        return f"{self.mlflow_registry_model_name_prefix}-{ticker}"

    def artifacts_dir_for(self, ticker: str) -> str:
        return os.path.join(self.artifacts_dir, ticker)


@dataclass
class MonitorConfig:
    """Настройки для src/monitor.py -- мониторинг живого качества и дрифта."""

    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    model_filename: str = "model.joblib"
    feature_list_filename: str = "features.json"
    metrics_filename: str = "metrics.json"
    reference_filename: str = "reference_features.csv"

    # Сколько последних ПОЛНОСТЬЮ реализованных дней брать для расчёта
    # "живого" качества (walk-forward): для каждого дня из этого окна мы уже
    # знаем и предсказание модели, и фактический результат.
    monitoring_window_days: int = int(os.getenv("MONITORING_WINDOW_DAYS", "20"))

    # Во сколько раз живой RMSE может быть хуже тренировочного, прежде чем
    # считаем это деградацией модели (не гейт "упало/не упало", а сигнал
    # для алерта -- реальный распад качества на рынке это нормально ожидать,
    # важно вовремя это заметить, а не сразу выключать сервис).
    max_degradation_ratio: float = float(os.getenv("MAX_DEGRADATION_RATIO", "1.8"))

    # Доля признаков с задетектированным дрифтом (Evidently), выше которой
    # считаем, что распределение входных данных значимо уехало от того, на
    # чём модель обучалась.
    max_drift_share: float = float(os.getenv("MAX_DRIFT_SHARE", "0.5"))

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_monitoring_experiment_name: str = os.getenv(
        "MLFLOW_MONITORING_EXPERIMENT_NAME", "cnyrub-rate-predictor-monitoring"
    )

    # Прометей Pushgateway: CronJob завершается и исчезает, поэтому Prometheus
    # не может сам "прийти и забрать" метрики -- job обязан их пушнуть сам,
    # пока жив. Пусто -- значит push пропускается (например, если Pushgateway
    # ещё не развёрнут).
    pushgateway_url: str = os.getenv("PUSHGATEWAY_URL", "")

    source: str = os.getenv("DATA_SOURCE", "moex")
    board: str = os.getenv("MOEX_BOARD", "CETS")


@dataclass
class ServeConfig:
    # Базовая директория, где serve.py на старте ищет подпапки с моделями
    # (по одной на тикер) -- та же самая artifacts_dir, что и при обучении.
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    model_filename: str = "model.joblib"
    feature_list_filename: str = "features.json"

    default_source: str = os.getenv("DATA_SOURCE", "moex")
    default_board: str = os.getenv("MOEX_BOARD", "CETS")
    log_level: str = os.getenv("LOG_LEVEL", "info")

    # Экспорт метрик для Prometheus (латентность, RPS, ошибки + кастомные
    # ML-гейджи по последнему предсказанию на тикер) -- см. src/serve.py
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
