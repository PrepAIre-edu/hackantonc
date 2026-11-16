"""AQI forecast: Holt-Winters exponential smoothing on daily means."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from analysis import daily

SEASONAL_PERIOD = 7           # weekly seasonality
MIN_DAYS_SEASONAL = 28        # need >= 4 weeks to use seasonality
MIN_DAYS_TRAIN = 10           # below this the model is meaningless
HISTORY_DAYS = 365            # train only on the last year (recent behaviour matters)
MAX_GAP_DAYS = 3              # only gaps up to this length are interpolated
MIN_HOURS_PER_DAY = 12        # days with fewer measurements are too noisy / partial
Z_95 = 1.96
FORECAST_COLUMNS = ["date", "yhat", "lower", "upper"]
MODELS = ["naive", "seasonal_naive", "holt_winters"]
BACKTEST_FOLDS = 8            # rolling windows used to measure the error
YEAR_PERIOD = 365             # annual seasonality (heating season)


def _daily_series(df: pd.DataFrame, max_days: int | None = HISTORY_DAYS) -> pd.Series:
    """Daily mean AQI as a gap-free series with a daily frequency.

    Partial days are dropped, small gaps are interpolated. If a larger gap remains,
    we keep only the latest continuous stretch (simplest honest choice).
    ``max_days`` limits the history (``None`` = everything).
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
    series = series[gap_id == gap_id.iloc[-1]].dropna()
    return series if max_days is None else series.tail(max_days)


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


def _baseline_forecasts(train: pd.Series, days: int) -> dict[str, np.ndarray]:
    """Two trivial benchmarks: repeat the last day / repeat the last week."""
    last_week = train.to_numpy()[-SEASONAL_PERIOD:]
    return {
        "naive": np.full(days, train.iloc[-1]),
        "seasonal_naive": np.resize(last_week, days),
    }


def rolling_backtest(df: pd.DataFrame, days: int = 7, folds: int = BACKTEST_FOLDS) -> pd.DataFrame:
    """Forecast the last ``folds`` windows of ``days`` days with every model.

    Columns: ``date, actual, naive, seasonal_naive, holt_winters``. Each window is
    predicted using only the data before it.
    """
    series = _daily_series(df)
    parts = []
    for fold in range(folds):
        cutoff = len(series) - days * (fold + 1)
        train, actual = series.iloc[:cutoff], series.iloc[cutoff:cutoff + days]
        if len(train) < MIN_DAYS_TRAIN:
            break
        predictions = _baseline_forecasts(train, days)
        predictions["holt_winters"] = _forecast_series(train, days)["yhat"].to_numpy()
        parts.append(pd.DataFrame({"date": actual.index, "actual": actual.to_numpy(), **predictions}))
    if not parts:
        raise ValueError("Замало даних для перевірки прогнозу.")
    return pd.concat(parts, ignore_index=True).sort_values("date", ignore_index=True)


def model_errors(backtest: pd.DataFrame) -> pd.DataFrame:
    """MAE of each model from ``rolling_backtest``: columns ``model, mae`` (best first)."""
    errors = {m: float((backtest[m] - backtest["actual"]).abs().mean()) for m in MODELS}
    return pd.DataFrame({"model": list(errors), "mae": list(errors.values())}).sort_values(
        "mae", ignore_index=True
    )


def backtest_mae(df: pd.DataFrame, days: int = 7) -> float:
    """Hold out the last ``days`` days, forecast them, return the mean absolute error."""
    backtest = rolling_backtest(df, days, folds=1)
    return float((backtest["holt_winters"] - backtest["actual"]).abs().mean())


def decompose_daily(df: pd.DataFrame, period: int = YEAR_PERIOD) -> pd.DataFrame:
    """Split daily AQI into trend + seasonality + residual (additive).

    Columns: ``date, observed, trend, seasonal, resid``. ``period=365`` shows the
    yearly (heating season) cycle, ``period=7`` the weekly one.
    """
    series = _daily_series(df, max_days=None)
    if len(series) < 2 * period:
        raise ValueError(f"Для розкладу з періодом {period} днів потрібно щонайменше {2 * period} днів даних.")
    parts = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="period")
    return pd.DataFrame({
        "date": series.index, "observed": series.to_numpy(), "trend": parts.trend.to_numpy(),
        "seasonal": parts.seasonal.to_numpy(), "resid": parts.resid.to_numpy(),
    })
