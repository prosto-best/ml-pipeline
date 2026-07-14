"""Проверяет метрики обученной модели против порога качества.

Используется в CI как gate: если модель хуже порога — pipeline падает и
новый образ не собирается / не деплоится. Так деградировавшая модель не
попадает в прод автоматически.
"""
import argparse
import json
import sys

from config import TrainConfig


def main(metrics_path: str) -> int:
    with open(metrics_path) as f:
        metrics = json.load(f)

    cfg = TrainConfig()
    rmse = metrics["rmse"]

    print(f"RMSE={rmse:.5f}  threshold={cfg.max_acceptable_rmse:.5f}")
    print(f"Directional accuracy={metrics['directional_accuracy']:.3f}")

    if rmse > cfg.max_acceptable_rmse:
        print("FAIL: модель хуже порогового качества, деплой останавливается.")
        return 1

    print("OK: модель прошла gate по качеству.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", default="artifacts/metrics.json")
    args = parser.parse_args()
    sys.exit(main(args.metrics_path))
