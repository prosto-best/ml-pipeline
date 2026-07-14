"""Загрузка исторических OHLCV-данных по тикеру.

Источник — yfinance (Yahoo Finance). В проде на месте этого модуля обычно
стоит платный провайдер (Polygon, IEX, Bloomberg) — интерфейс функции
специально сделан так, чтобы источник можно было заменить без изменений
в остальном пайплайне.
"""
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_ohlcv(ticker: str, start_date: str, end_date: str = "") -> pd.DataFrame:
    """Скачивает дневные OHLCV-данные для тикера.

    Returns:
        DataFrame с колонками [Open, High, Low, Close, Adj Close, Volume],
        индекс — DatetimeIndex, отсортирован по возрастанию даты.
    """
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    logger.info("Loading %s from %s to %s", ticker, start_date, end_date)

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"Не удалось загрузить данные для тикера '{ticker}'. "
            "Проверьте название тикера и доступность сети."
        )

    # yfinance иногда возвращает MultiIndex колонки при мульти-тикере — на всякий случай flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    df.index.name = "Date"
    logger.info("Loaded %d rows for %s", len(df), ticker)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()

    data = load_ohlcv(args.ticker, args.start)
    print(data.tail())
