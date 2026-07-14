"""Построение признаков (технические индикаторы) и таргета для модели.

Таргет = логарифмический return цены закрытия через `horizon_days` торговых дней.
Работа с log-return вместо абсолютной цены даёт стационарный ряд, с которым
модель обучается стабильнее и который не "уезжает" при росте/падении цены
за пределы диапазона, виденного при обучении.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def build_features(
    df: pd.DataFrame,
    lag_windows: List[int],
    ma_windows: List[int],
    rsi_window: int,
    horizon_days: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Строит матрицу признаков и таргет из сырых OHLCV-данных.

    Returns:
        (df_with_features_and_target, feature_column_names)
    """
    out = df.copy()
    close = out["Close"]
    log_close = np.log(close)
    log_return_1d = log_close.diff()

    feature_cols: List[str] = []

    # Лаги дневного log-return
    for lag in lag_windows:
        col = f"return_lag_{lag}"
        out[col] = log_return_1d.shift(lag - 1) if lag == 1 else log_return_1d.rolling(lag).sum().shift(1)
        feature_cols.append(col)

    # Скользящие средние (отношение цены к MA — масштабо-независимый признак)
    for w in ma_windows:
        col = f"close_to_ma_{w}"
        out[col] = close / close.rolling(w).mean() - 1
        feature_cols.append(col)

    # Волатильность (std дневных доходностей за окно)
    for w in [5, 10, 20]:
        col = f"volatility_{w}"
        out[col] = log_return_1d.rolling(w).std()
        feature_cols.append(col)

    # RSI
    out["rsi"] = _rsi(close, rsi_window)
    feature_cols.append("rsi")

    # MACD
    macd, signal = _macd(close)
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal
    feature_cols += ["macd", "macd_signal", "macd_hist"]

    # Объём (для CNY/RUB это оборот в рублях с MOEX, поле `value`): относительное
    # изменение к среднему за 20 дней. На валютных парах Volume иногда отсутствует
    # или равен нулю (например, при использовании резервного источника Yahoo) —
    # в этом случае признак аккуратно зануляется, а не взрывается в +-inf.
    volume_ma_20 = out["Volume"].rolling(20).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_rel_20 = out["Volume"] / volume_ma_20 - 1
    out["volume_rel_20"] = volume_rel_20.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_cols.append("volume_rel_20")

    # High-Low spread как доля цены закрытия (прокси внутридневной волатильности)
    out["hl_spread"] = (out["High"] - out["Low"]) / close
    feature_cols.append("hl_spread")

    # Таргет: суммарный log-return за horizon_days вперёд
    out["target"] = log_close.shift(-horizon_days) - log_close

    out = out.dropna(subset=feature_cols + ["target"])
    return out, feature_cols


def get_latest_feature_row(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Возвращает последнюю строку признаков — для инференса на "сегодня"."""
    return df[feature_cols].iloc[[-1]]
