"""AQI forecast: Holt-Winters exponential smoothing on daily means."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from analysis import daily

SEASONAL_PERIOD = 7           # weekly seasonality
MIN_DAYS_SEASONAL = 28        # need >= 4 weeks to use seasonality
MIN_DAYS_TRAIN = 10           # below this the model is meaningless
HISTORY_DAYS = 365            # train only on the last year (recent behaviour matters)
MAX_GAP_DAYS = 3              # only gaps up to this length are interpolated
MIN_HOURS_PER_DAY = 12        # days with fewer measurements are too noisy / partial
Z_95 = 1.96
FORECAST_COLUMNS = ["date", "yhat", "lower", "upper"]


def _daily_series(df: pd.DataFrame) -> pd.Series:
    """Daily mean AQI as a gap-free series with a daily frequency.

    Partial days are dropped, small gaps are interpolated. If a larger gap remains,
    we keep only the latest continuous stretch (simplest honest choice).
    """
    valid = df.dropna(subset=["aqi"])
    hours_per_day = valid.groupby(valid["time"].dt.normalize()).size()
    series = daily(valid).set_index("date")["aqi"]
    series = series[hours_per_day.reindex(series.index) >= MIN_HOURS_PER_DAY]
    if series.empty:
        return series
    series = series.asfreq("D").interpolate(limit=MAX_GAP_DAYS, limit_area="inside")
    series = series.loc[series.first_valid_index():series.last_valid_index()]
    gap_id = series.isna().cumsum()
    return series[gap_id == gap_id.iloc[-1]].dropna().tail(HISTORY_DAYS)


def _forecast_series(series: pd.Series, days: int) -> pd.DataFrame:
    """Fit Holt-Winters on ``series`` and forecast ``days`` days ahead."""
    if len(series) < MIN_DAYS_TRAIN:
        raise ValueError(
            f"Замало даних для прогнозу: потрібно хоча б {MIN_DAYS_TRAIN} днів, є {len(series)}."
        )
    seasonal = "add" if len(series) >= MIN_DAYS_SEASONAL else None
    model = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,  # damped trend: long forecasts do not run away
        seasonal=seasonal,
        seasonal_periods=SEASONAL_PERIOD if seasonal else None,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels convergence chatter
        fit = model.fit()
    yhat = np.clip(np.asarray(fit.forecast(days)), 0, None)
    margin = Z_95 * float(np.std(fit.resid, ddof=1))
    dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    return pd.DataFrame(
        {"date": dates, "yhat": yhat, "lower": np.clip(yhat - margin, 0, None), "upper": yhat + margin}
    )[FORECAST_COLUMNS]


def forecast_aqi(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """Forecast daily AQI: columns ``date, yhat, lower, upper`` (lower >= 0).

    Uses weekly seasonality when there are >= 28 days of data, otherwise a plain
    damped trend. The corridor is ``yhat +- 1.96 * std(residuals)``.
    """
    return _forecast_series(_daily_series(df), days)


def backtest_mae(df: pd.DataFrame, days: int = 7) -> float:
    """Hold out the last ``days`` days, forecast them, return the mean absolute error."""
    series = _daily_series(df)
    train, actual = series.iloc[:-days], series.iloc[-days:]
    predicted = _forecast_series(train, days)["yhat"].to_numpy()
    return float(np.mean(np.abs(predicted - actual.to_numpy())))
