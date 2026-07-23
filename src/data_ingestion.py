"""Загрузка исторических данных по валютной паре CNY/RUB (юань/рубль).

Основной источник — **Московская биржа (MOEX ISS API)**, а не Yahoo Finance.
Причина: с 2022 года CNY/RUB торгуется на MOEX напрямую (инструмент
`CNYRUB_TOM`, режим `CETS`, engine=currency, market=selt) и это первичный
рынок для данной пары. Синтетический кросс-курс, который отдал бы Yahoo
Finance (`CNYRUB=X`, посчитанный через доллар как мост), для реально
торгуемой на MOEX пары был бы менее точным и мог бы не совпадать с
фактическими котировками, по которым исполняются сделки.

MOEX ISS не требует токена/регистрации для дневных свечей.
Документация: https://iss.moex.com/iss/reference/
"""
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MOEX_BASE_URL = (
    "https://iss.moex.com/iss/engines/currency/markets/selt/boards/{board}"
    "/securities/{secid}/candles.json"
)


def load_ohlcv_moex(
    secid: str = "CNYRUB_TOM",
    board: str = "CETS",
    start_date: str = "2015-01-01",
    end_date: str = "",
    interval: int = 24,  # 24 = дневные свечи в MOEX ISS
) -> pd.DataFrame:
    """Скачивает дневные свечи валютной пары с Московской биржи.

    Args:
        secid: код инструмента на MOEX (по умолчанию CNYRUB_TOM — юань/рубль,
            расчёты "завтра").
        board: режим торгов (CETS — основной режим валютного рынка MOEX).
        start_date, end_date: границы периода в формате YYYY-MM-DD.
        interval: 24 = сутки. MOEX также поддерживает 1/10/60 (мин/часы) —
            не используются в этом пайплайне, т.к. модель работает на
            дневных данных.

    Returns:
        DataFrame с колонками [Open, High, Low, Close, Adj Close, Volume],
        индекс — DatetimeIndex, отсортирован по возрастанию даты.
        Volume = оборот в рублях за свечу, из поля `value` MOEX ISS.
    """
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    url = MOEX_BASE_URL.format(board=board, secid=secid)

    all_rows = []
    columns = []
    start_cursor = 0
    page_size = 500  # MOEX ISS отдаёт ограниченное число строк за запрос, пагинация через `start`

    logger.info("Loading MOEX %s (%s) from %s to %s", secid, board, start_date, end_date)

    while True:
        params = {
            "from": start_date,
            "till": end_date,
            "interval": interval,
            "start": start_cursor,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        candles_block = payload.get("candles", {})
        columns = candles_block.get("columns", columns)
        rows = candles_block.get("data", [])

        if not rows:
            break

        all_rows.extend(rows)
        start_cursor += len(rows)

        if len(rows) < page_size:
            break

    if not all_rows:
        raise ValueError(
            f"MOEX ISS не вернул данных для '{secid}' на борде '{board}' "
            f"за период {start_date}..{end_date}. Проверьте secid/board и "
            "доступность сети до iss.moex.com."
        )

    df = pd.DataFrame(all_rows, columns=columns)

    # Типовые колонки MOEX candles: open, close, high, low, value, volume, begin, end
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.set_index("begin").sort_index()
    df.index.name = "Date"

    out = pd.DataFrame(
        {
            "Open": df["open"].astype(float),
            "High": df["high"].astype(float),
            "Low": df["low"].astype(float),
            "Close": df["close"].astype(float),
            "Adj Close": df["close"].astype(float),
            # `value` — оборот в рублях за свечу; более показателен для FX,
            # чем `volume` (объём в лотах базовой валюты)
            "Volume": df["value"].astype(float) if "value" in df.columns else 0.0,
        }
    )
    logger.info("Loaded %d rows for %s from MOEX", len(out), secid)
    return out


def load_ohlcv_yahoo(ticker: str, start_date: str, end_date: str = "") -> pd.DataFrame:
    """Резервный источник через Yahoo Finance (синтетический кросс-курс).

    Полезен как fallback, если MOEX ISS временно недоступен, либо для
    сравнения/бэктеста на других валютных парах. Для боевого использования
    по CNY/RUB предпочтителен `load_ohlcv_moex`.
    """
    import yfinance as yf

    end_date = end_date or datetime.today().strftime("%Y-%m-%d")
    logger.info("Loading %s from Yahoo Finance: %s to %s", ticker, start_date, end_date)

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"Не удалось загрузить данные для тикера '{ticker}' с Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    df.index.name = "Date"
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df


def load_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str = "",
    source: str = "moex",
    board: Optional[str] = "CETS",
) -> pd.DataFrame:
    """Единая точка входа для остального пайплайна.

    `source` управляет провайдером данных: "moex" (по умолчанию, рекомендуется
    для CNY/RUB) или "yahoo" (резервный вариант / другие инструменты).
    """
    if source == "moex":
        return load_ohlcv_moex(secid=ticker, board=board or "CETS", start_date=start_date, end_date=end_date)
    if source == "yahoo":
        return load_ohlcv_yahoo(ticker, start_date, end_date)
    raise ValueError(f"Неизвестный источник данных: {source!r}. Ожидается 'moex' или 'yahoo'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="CNYRUB_TOM", help="secid на MOEX или тикер Yahoo")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--source", default="moex", choices=["moex", "yahoo"])
    args = parser.parse_args()

    data = load_ohlcv(args.ticker, args.start, source=args.source)
    print(data.tail())
