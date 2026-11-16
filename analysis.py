"""Pandas aggregations over the hourly table ``time, pm25, aqi``."""
import pandas as pd

METRICS = ["pm25", "aqi"]
SUMMARY_WINDOW_DAYS = 30


def daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily means: ``date`` (midnight timestamp), ``pm25``, ``aqi``."""
    grouped = df.groupby(df["time"].dt.normalize())[METRICS].mean()
    return grouped.rename_axis("date").reset_index()


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pm25/aqi by hour of day: ``hour`` 0-23, ``pm25``, ``aqi``."""
    return df.groupby(df["time"].dt.hour)[METRICS].mean().rename_axis("hour").reset_index()


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pm25/aqi by weekday: ``weekday`` 0=Mon..6=Sun, ``pm25``, ``aqi``."""
    return (
        df.groupby(df["time"].dt.weekday)[METRICS].mean().rename_axis("weekday").reset_index()
    )


def hour_weekday_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Mean AQI: rows = weekday (0-6), columns = hour (0-23)."""
    matrix = df.pivot_table(
        index=df["time"].dt.weekday,
        columns=df["time"].dt.hour,
        values="aqi",
        aggfunc="mean",
        dropna=False,
    )
    return matrix.reindex(index=range(7), columns=range(24)).rename_axis(
        index="weekday", columns="hour"
    )


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly means: ``year``, ``month``, ``pm25``, ``aqi``."""
    grouped = df.groupby([df["time"].dt.year, df["time"].dt.month])[METRICS].mean()
    return grouped.rename_axis(["year", "month"]).reset_index()


def summary(df: pd.DataFrame) -> dict:
    """Key numbers for the "now" card.

    Keys: ``last_time``, ``last_pm25``, ``last_aqi`` (latest row with a valid AQI),
    ``mean_pm25_30d``, ``mean_aqi_30d``, ``worst_day``, ``worst_day_aqi``.
    The 30-day window ends at the latest measurement (data may be older than today);
    the worst day is also searched inside this window. Empty data gives ``None`` values.
    """
    result = dict.fromkeys(
        ["last_time", "last_pm25", "last_aqi", "mean_pm25_30d", "mean_aqi_30d",
         "worst_day", "worst_day_aqi"]
    )
    valid = df.dropna(subset=["aqi"])
    if valid.empty:
        return result
    last = valid.iloc[-1]
    window = df[df["time"] > last["time"] - pd.Timedelta(days=SUMMARY_WINDOW_DAYS)]
    days = daily(window).dropna(subset=["aqi"])
    result.update(
        last_time=last["time"],
        last_pm25=last["pm25"],
        last_aqi=last["aqi"],
        mean_pm25_30d=window["pm25"].mean(),
        mean_aqi_30d=window["aqi"].mean(),
    )
    if not days.empty:
        worst = days.loc[days["aqi"].idxmax()]
        result.update(worst_day=worst["date"], worst_day_aqi=worst["aqi"])
    return result
