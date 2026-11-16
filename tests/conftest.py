"""Shared synthetic data: 120 days of hourly values with a weekly pattern."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_df() -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=120 * 24, freq="h")
    rng = np.random.default_rng(0)
    aqi = 50 + 10 * np.sin(2 * np.pi * times.hour / 24) + 5 * (times.day_of_week >= 5) + rng.normal(0, 2, len(times))
    return pd.DataFrame({"time": times, "pm25": aqi / 3, "aqi": aqi})
