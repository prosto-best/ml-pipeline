"""Проверяет метрики всех обученных моделей против порога качества.

Автоматически находит все подпапки artifacts_dir/<ticker>/metrics.json
(сколько бы тикеров ни было обучено train.py) и для каждой сверяет RMSE
с порогом. Используется в CI как gate: если хотя бы одна модель хуже
порога -- pipeline падает и новый образ не собирается/не деплоится, чтобы
деградировавшая модель не попала в прод автоматически.

Если модель прошла гейт и в её features.json есть mlflow_run_id -- скрипт
дополнительно переводит СООТВЕТСТВУЮЩУЮ версию модели в MLflow Model
Registry в стадию "Production" (через MlflowClient.transition_model_version_stage).
Так квалити-гейт становится единственным местом, откуда модель попадает в
прод-стадию реестра -- вручную "нажать кнопку" в MLflow UI не нужно, и
случайно продвинуть непрошедшую гейт модель тоже нельзя.
"""
import argparse
import glob
import json
import logging
import os
import sys

from config import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _promote_to_production(ticker: str, meta: dict, cfg: TrainConfig) -> None:
    """Переводит версию модели, обученную в данном run'е, в стадию Production."""
    run_id = meta.get("mlflow_run_id")
    model_name = meta.get("mlflow_registered_model")
    if not run_id or not model_name:
        logger.info("[%s] Нет данных о MLflow-регистрации в features.json -- пропускаю promote", ticker)
        return

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        if cfg.mlflow_tracking_uri:
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

        client = MlflowClient()
        # Находим версию модели, созданную именно этим run_id
        versions = client.search_model_versions(f"name='{model_name}'")
        matching = [v for v in versions if v.run_id == run_id]
        if not matching:
            logger.warning("[%s] Не найдена версия модели '%s' для run_id=%s", ticker, model_name, run_id)
            return

        version = matching[0].version
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("[%s] MLflow: %s v%s -> Production", ticker, model_name, version)
    except Exception as e:  # noqa: BLE001
        # Промоушен в реестре -- дополнительная телеметрия, а не критичный
        # для деплоя шаг (сама serving-часть всё равно читает локальные
        # артефакты), поэтому падать из-за недоступности MLflow не нужно.
        logger.warning("[%s] Не удалось выполнить promote в MLflow: %s", ticker, e)


def main(artifacts_dir: str, promote: bool) -> int:
    cfg = TrainConfig()
    pattern = os.path.join(artifacts_dir, "*", "metrics.json")
    metrics_paths = sorted(glob.glob(pattern))

    if not metrics_paths:
        print(f"FAIL: не найдено ни одного metrics.json по пути {pattern}")
        return 1

    all_passed = True
    for path in metrics_paths:
        ticker_dir = os.path.dirname(path)
        with open(path) as f:
            metrics = json.load(f)

        ticker = metrics.get("ticker", os.path.basename(ticker_dir))
        rmse = metrics["rmse"]
        passed = rmse <= cfg.max_acceptable_rmse

        status = "OK" if passed else "FAIL"
        print(
            f"[{status}] {ticker}: RMSE={rmse:.5f} "
            f"(threshold={cfg.max_acceptable_rmse:.5f}), "
            f"directional_accuracy={metrics['directional_accuracy']:.3f}"
        )
        all_passed = all_passed and passed

        if passed and promote:
            features_path = os.path.join(ticker_dir, cfg.feature_list_filename)
            if os.path.exists(features_path):
                with open(features_path) as f:
                    meta = json.load(f)
                _promote_to_production(ticker, meta, cfg)

    if not all_passed:
        print("FAIL: хотя бы одна модель хуже порогового качества, деплой останавливается.")
        return 1

    print(f"OK: все {len(metrics_paths)} модел{'ь' if len(metrics_paths) == 1 else 'и'} прошли gate по качеству.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Не переводить прошедшие гейт модели в стадию Production в MLflow Registry",
    )
    args = parser.parse_args()
    sys.exit(main(args.artifacts_dir, promote=not args.no_promote))
