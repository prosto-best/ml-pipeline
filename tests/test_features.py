import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import build_features  # noqa: E402


def _fake_ohlcv(n=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    price = np.clip(price, 1, None)
    df = pd.DataFrame(
        {
            "Open": price + rng.normal(0, 0.5, n),
            "High": price + np.abs(rng.normal(0, 1, n)),
            "Low": price - np.abs(rng.normal(0, 1, n)),
            "Close": price,
            "Adj Close": price,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )
    return df


def test_build_features_shapes():
    df = _fake_ohlcv()
    featured, feature_cols = build_features(
        df, lag_windows=[1, 2, 3], ma_windows=[5, 10], rsi_window=14, horizon_days=1
    )
    assert len(feature_cols) > 0
    assert "target" in featured.columns
    assert not featured[feature_cols].isna().any().any()
    assert not featured["target"].isna().any()


def test_no_lookahead_in_target():
    """Таргет на дату t должен зависеть только от цены на t и t+horizon, не раньше."""
    df = _fake_ohlcv()
    featured, _ = build_features(df, [1], [5], 14, horizon_days=1)
    close = df["Close"]
    log_close = np.log(close)
    expected_target = (log_close.shift(-1) - log_close).reindex(featured.index)
    pd.testing.assert_series_equal(
        featured["target"], expected_target, check_names=False
    )


def test_rsi_bounds():
    df = _fake_ohlcv()
    featured, _ = build_features(df, [1], [5], 14, horizon_days=1)
    assert featured["rsi"].between(0, 100).all()
