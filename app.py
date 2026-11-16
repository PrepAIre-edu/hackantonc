"""Streamlit UI "Чисте небо": air quality history, patterns, forecast and city comparison."""
import pandas as pd
import streamlit as st

import analysis
import charts
import forecast
import insights
import loader

DATA_SOURCE_URL = "https://www.saveecobot.com/"
DEFAULT_CITY = "kyiv"
FORECAST_DAYS_RANGE = (3, 14, 7)  # min, max, default
MIN_DAYS_FOR_ANALYSIS = 7
STATS_COLUMNS = {
    "city": "Місто", "mean_aqi": "Середній AQI", "median_aqi": "Медіана", "p95_aqi": "95-й перцентиль",
    "unhealthy_share": "Шкідливих годин, %", "best_hour": "Найчистіша година", "worst_hour": "Найбрудніша година",
}
MODEL_EXPLANATION = (
    "Щоб чесно оцінити прогноз, ми кілька разів «повертаємось у минуле»: прогнозуємо тиждень, "
    "який уже відомий, і порівнюємо з реальністю. Наївні моделі — прості еталони: «буде як учора» "
    "і «буде як тиждень тому». Якщо складна модель не краща за еталон, це теж корисна відповідь."
)


@st.cache_data(show_spinner="Завантажуємо дані…")
def get_city_data(city: str) -> pd.DataFrame:
    """Cached hourly data of one city."""
    return loader.load_city(city)


@st.cache_data(show_spinner="Рахуємо прогноз…")
def get_forecast(city: str, days: int) -> pd.DataFrame:
    """Cached forecast for one city."""
    return forecast.forecast_aqi(get_city_data(city), days)


@st.cache_data(show_spinner="Перевіряємо прогноз на минулому…")
def get_backtest(city: str, days: int) -> pd.DataFrame:
    """Cached rolling backtest of all models."""
    return forecast.rolling_backtest(get_city_data(city), days)


@st.cache_data(show_spinner="Розкладаємо ряд на тренд і сезонність…")
def get_decomposition(city: str) -> pd.DataFrame:
    """Cached yearly decomposition of daily AQI."""
    return forecast.decompose_daily(get_city_data(city))


def render_sidebar(cities: list[str]) -> tuple[str, int]:
    """Sidebar controls: city and forecast length."""
    st.sidebar.header("Налаштування")
    index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0
    city = st.sidebar.selectbox("Місто", cities, index=index, format_func=loader.city_label)
    low, high, default = FORECAST_DAYS_RANGE
    days = st.sidebar.slider("Днів прогнозу", low, high, default)
    return city, days


def render_now_card(df: pd.DataFrame) -> None:
    """Top card: latest AQI, category, advice and best hour."""
    info = analysis.summary(df)
    aqi = float("nan") if info["last_aqi"] is None else info["last_aqi"]
    best = insights.best_hour(analysis.hourly_profile(df))
    st.subheader("Зараз")
    col_aqi, col_category, col_hour = st.columns(3)
    col_aqi.metric("AQI", "—" if pd.isna(aqi) else f"{aqi:.0f}")
    col_category.markdown(
        f"**Категорія**<br><span style='color:{charts.color_for_aqi(aqi)};"
        f"font-size:1.6rem;font-weight:700'>{insights.aqi_category(aqi)}</span>",
        unsafe_allow_html=True,
    )
    col_hour.metric("Найкраща година для вулиці", best)
    st.info(insights.advice(aqi))
    if info["last_time"] is not None:
        st.caption(f"Останнє вимірювання: {info['last_time']:%d.%m.%Y %H:%M}. "
                   f"Середній AQI за 30 днів: {info['mean_aqi_30d']:.0f}.")


def render_forecast_tab(city: str, df: pd.DataFrame, days: int) -> None:
    """Forecast chart, its error and the comparison with simple benchmarks."""
    try:
        result = get_forecast(city, days)
        backtest = get_backtest(city, days)
    except ValueError as exc:
        st.warning(str(exc))
        st.plotly_chart(charts.line_with_forecast(analysis.daily(df), pd.DataFrame()), width="stretch")
        return
    errors = forecast.model_errors(backtest)
    mae = errors.set_index("model").loc["holt_winters", "mae"]
    st.plotly_chart(charts.line_with_forecast(analysis.daily(df), result), width="stretch")
    st.success(f"Перевірка на минулому: у середньому ми помиляємось на {mae:.0f} одиниць AQI.")
    st.subheader("Наскільки можна довіряти прогнозу")
    st.caption(MODEL_EXPLANATION)
    left, right = st.columns(2)
    left.plotly_chart(charts.model_comparison(errors), width="stretch")
    right.plotly_chart(charts.predicted_vs_actual(backtest), width="stretch")


def render_patterns_tab(city: str, df: pd.DataFrame) -> None:
    """Hourly/weekly/seasonal patterns and plain-language findings."""
    st.subheader("Головне про це місто")
    for finding in insights.key_findings(df):
        st.markdown(f"- {finding}")
    month = int(df["time"].iloc[-1].month)
    st.markdown(f"- У {insights.MONTH_NAMES[month - 1]}: {insights.compare_to_last_year(df, month)}.")
    left, right = st.columns(2)
    left.plotly_chart(charts.hourly_profile_chart(analysis.hourly_profile(df)), width="stretch")
    right.plotly_chart(charts.weekday_profile_chart(analysis.weekday_profile(df)), width="stretch")
    st.plotly_chart(charts.heatmap_hour_weekday(analysis.hour_weekday_matrix(df)), width="stretch")
    st.plotly_chart(charts.heatmap_month_year(analysis.month_year_matrix(df)), width="stretch")
    st.plotly_chart(charts.category_share_chart(insights.category_share_by_year(df)), width="stretch")
    st.plotly_chart(charts.weekly_trend(analysis.weekly(df)), width="stretch")
    try:
        st.plotly_chart(charts.decomposition_chart(get_decomposition(city)), width="stretch")
    except ValueError as exc:
        st.info(str(exc))


def render_cities_tab(cities: list[str]) -> None:
    """Comparison of all cities."""
    frames = {loader.city_label(city): get_city_data(city) for city in cities}
    stats = analysis.city_stats(frames)
    st.plotly_chart(charts.compare_cities(frames), width="stretch")
    st.subheader("Порівняння за останній рік")
    table = stats.rename(columns=STATS_COLUMNS).round(1)
    for column in ("Найчистіша година", "Найбрудніша година"):
        table[column] = table[column].map(lambda hour: f"{int(hour):02d}:00")
    st.dataframe(table, hide_index=True, width="stretch")
    st.plotly_chart(charts.city_boxplot(analysis.daily_by_city(frames)), width="stretch")
    st.plotly_chart(charts.heatmap_city_hour(analysis.city_hour_matrix(frames)), width="stretch")


def render_footer() -> None:
    """License attribution required by CC BY 4.0."""
    st.divider()
    st.markdown(f"Дані: [SaveEcoBot]({DATA_SOURCE_URL}) (CC BY 4.0)")


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Чисте небо", page_icon="🌤️", layout="wide")
    st.title("🌤️ Чисте небо")
    st.caption("Якість повітря в містах України: історія, закономірності та прогноз.")
    cities = loader.list_cities()
    if not cities:
        st.error("У папці data/ немає CSV-файлів із даними.")
        render_footer()
        return
    city, days = render_sidebar(cities)
    try:
        df = get_city_data(city)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Не вдалося завантажити дані: {exc}")
        render_footer()
        return
    if df["aqi"].notna().sum() < MIN_DAYS_FOR_ANALYSIS * 24:
        st.warning("Для цього міста замало даних, щоб показати аналіз.")
        render_footer()
        return
    render_now_card(df)
    forecast_tab, patterns_tab, cities_tab = st.tabs(["Прогноз", "Закономірності", "Порівняння міст"])
    with forecast_tab:
        render_forecast_tab(city, df, days)
    with patterns_tab:
        render_patterns_tab(city, df)
    with cities_tab:
        render_cities_tab(cities)
    render_footer()


main()
