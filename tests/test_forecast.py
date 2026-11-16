import pytest

import forecast


def test_forecast_shape_and_corridor(hourly_df):
    result = forecast.forecast_aqi(hourly_df, days=7)
    assert len(result) == 7
    assert list(result.columns) == ["date", "yhat", "lower", "upper"]
    assert ((result["lower"] <= result["yhat"]) & (result["yhat"] <= result["upper"])).all()
    assert (result["lower"] >= 0).all()


def test_forecast_without_seasonality_on_short_history(hourly_df):
    short = hourly_df[hourly_df["time"] < "2025-01-21"]  # 20 days < 28
    assert len(forecast.forecast_aqi(short, days=5)) == 5


def test_too_little_data_raises(hourly_df):
    with pytest.raises(ValueError):
        forecast.forecast_aqi(hourly_df.head(24 * 3))


def test_backtest_mae_is_small_on_regular_data(hourly_df):
    assert 0 <= forecast.backtest_mae(hourly_df, days=7) < 5
