"""Static report: PNG charts and a text summary for one city plus the city comparison.

Usage: ``python report.py [city]`` (default: kyiv). Output goes to ``outputs/``.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import analysis
import charts
import forecast
import insights
import loader

OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_CITY = "kyiv"
SHORT_FORECAST_DAYS = 7
LONG_FORECAST_DAYS = 30
IMAGE_SIZE = (1100, 600)  # width, height of PNG files
IMAGE_SCALE = 1.5
DISCLAIMER = (
    "Прогноз — статистична оцінка за історією вимірювань, а не медична порада. "
    "Дані: SaveEcoBot (CC BY 4.0), https://www.saveecobot.com/"
)


def labelled_frames(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Same data keyed by the Ukrainian city names (for chart and table labels)."""
    return {loader.city_label(name): df for name, df in dfs.items()}


def build_figures(city: str, dfs: dict[str, pd.DataFrame]) -> dict[str, go.Figure]:
    """All report charts, keyed by file name (without extension), in display order."""
    df, labelled = dfs[city], labelled_frames(dfs)
    backtest = forecast.rolling_backtest(df, SHORT_FORECAST_DAYS)
    daily = analysis.daily(df)
    return {
        "01_weekly_trend": charts.weekly_trend(analysis.weekly(df)),
        "02_heatmap_hour_weekday": charts.heatmap_hour_weekday(analysis.hour_weekday_matrix(df)),
        "03_hourly_profile": charts.hourly_profile_chart(analysis.hourly_profile(df)),
        "04_weekday_profile": charts.weekday_profile_chart(analysis.weekday_profile(df)),
        "05_heatmap_month_year": charts.heatmap_month_year(analysis.month_year_matrix(df)),
        "06_category_share": charts.category_share_chart(insights.category_share_by_year(df)),
        "07_decomposition": charts.decomposition_chart(forecast.decompose_daily(df)),
        "08_forecast_short": charts.line_with_forecast(daily, forecast.forecast_aqi(df, SHORT_FORECAST_DAYS)),
        "09_forecast_long": charts.line_with_forecast(daily, forecast.forecast_aqi(df, LONG_FORECAST_DAYS)),
        "10_model_comparison": charts.model_comparison(forecast.model_errors(backtest)),
        "11_predicted_vs_actual": charts.predicted_vs_actual(backtest),
        "12_compare_cities": charts.compare_cities(labelled),
        "13_city_boxplot": charts.city_boxplot(analysis.daily_by_city(labelled)),
        "14_city_hour_heatmap": charts.heatmap_city_hour(analysis.city_hour_matrix(labelled)),
    }


def save_figures(figures: dict[str, go.Figure], plots_dir: Path) -> None:
    """Write every figure as PNG into ``plots_dir`` (one browser session for all files)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    width, height = IMAGE_SIZE
    for fig in figures.values():
        fig.update_layout(width=width, height=fig.layout.height or height)
    pio.write_images(
        fig=list(figures.values()),
        file=[plots_dir / f"{name}.png" for name in figures],
        scale=IMAGE_SCALE,
    )


def models_by_city(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MAE of every forecasting model for every city (rows = city, columns = model)."""
    rows = {}
    for city, df in dfs.items():
        try:
            errors = forecast.model_errors(forecast.rolling_backtest(df, SHORT_FORECAST_DAYS))
        except ValueError:
            continue
        rows[city] = errors.set_index("model")["mae"]
    return pd.DataFrame(rows).T[forecast.MODELS].rename(index=loader.city_label)


def _section(title: str, lines: list[str]) -> str:
    """Text block with an underlined title."""
    return "\n".join([title, "-" * len(title), *lines, ""])


def build_summary(city: str, dfs: dict[str, pd.DataFrame]) -> str:
    """Text report: key numbers, findings, forecast quality and city ranking."""
    df, info = dfs[city], analysis.summary(dfs[city])
    errors = models_by_city(dfs)
    wins = int((errors.idxmin(axis=1) == "holt_winters").sum())
    stats = analysis.city_stats(labelled_frames(dfs)).sort_values("mean_aqi")
    forecast_table = forecast.forecast_aqi(df, SHORT_FORECAST_DAYS).round({"yhat": 1, "lower": 1, "upper": 1})
    forecast_table = forecast_table.to_string(index=False)
    return "\n".join([
        f"ЧИСТЕ НЕБО: звіт по місту {loader.city_label(city)}",
        f"Сформовано: {datetime.now():%Y-%m-%d %H:%M}. Період даних: {df['time'].min():%Y-%m-%d} — "
        f"{df['time'].max():%Y-%m-%d}, годинних записів: {len(df):,}",
        DISCLAIMER,
        "",
        _section("1. Зараз", [
            f"Останній AQI: {info['last_aqi']:.0f} ({insights.aqi_category(info['last_aqi'])}) "
            f"на {info['last_time']:%Y-%m-%d %H:%M}",
            f"Порада: {insights.advice(info['last_aqi'])}",
            f"Середній AQI за 30 днів: {info['mean_aqi_30d']:.1f}; найгірший день: "
            f"{info['worst_day']:%Y-%m-%d} (AQI {info['worst_day_aqi']:.0f})",
        ]),
        _section("2. Закономірності", [f"- {text}" for text in insights.key_findings(df)]),
        _section(f"3. Прогноз на {SHORT_FORECAST_DAYS} днів (AQI, коридор 95%)", [forecast_table]),
        _section("4. Якість прогнозу (MAE на 8 останніх тижнях, менше — краще)", [
            errors.round(2).rename(columns=charts.MODEL_LABELS).to_string(),
            f"Модель Хольта–Вінтерса найкраща у {wins} з {len(errors)} міст; різниця з наївним "
            f"еталоном «як учора» невелика, бо добовий AQI дуже інерційний.",
        ]),
        _section("5. Порівняння міст за останній рік", [stats.round(1).to_string(index=False)]),
    ])


def main(city: str = DEFAULT_CITY) -> None:
    """Build all charts and the summary for ``city`` into ``outputs/``."""
    dfs = {name: loader.load_city(name) for name in loader.list_cities()}
    city = city.lower()
    if city not in dfs:
        raise SystemExit(f"Невідоме місто '{city}'. Доступні: {', '.join(dfs)}")
    save_figures(build_figures(city, dfs), OUTPUT_DIR / "plots")
    reports_dir = OUTPUT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "summary.txt").write_text(build_summary(city, dfs), encoding="utf-8")
    analysis.city_stats(labelled_frames(dfs)).round(2).to_csv(reports_dir / "city_stats.csv", index=False)
    print(f"Готово: графіки в {OUTPUT_DIR / 'plots'}, звіт у {reports_dir / 'summary.txt'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:2])
