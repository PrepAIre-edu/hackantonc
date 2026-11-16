"""Streamlit UI "Чисте небо": air quality history, patterns and forecast."""
import pandas as pd
import streamlit as st

import analysis
import charts
import forecast
import insights
import loader

DATA_SOURCE_URL = "https://www.saveecobot.com/"
CITY_LABELS = {
    "dnipro": "Дніпро", "kharkiv": "Харків", "kyiv": "Київ",
    "lutsk": "Луцьк", "lviv": "Львів", "odessa": "Одеса",
}
DEFAULT_CITY = "kyiv"
FORECAST_DAYS_RANGE = (3, 14, 7)  # min, max, default
MIN_DAYS_FOR_ANALYSIS = 7


@st.cache_data(show_spinner="Завантажуємо дані…")
def get_city_data(city: str) -> pd.DataFrame:
    """Cached hourly data of one city."""
    return loader.load_city(city)


@st.cache_data(show_spinner="Рахуємо прогноз…")
def get_forecast(city: str, days: int) -> tuple[pd.DataFrame, float | None]:
    """Cached forecast and its backtest MAE (``None`` if the backtest is impossible)."""
    df = get_city_data(city)
    result = forecast.forecast_aqi(df, days)
    try:
        mae = forecast.backtest_mae(df, days)
    except ValueError:
        mae = None
    return result, mae


def city_label(city: str) -> str:
    """Ukrainian display name of a city (falls back to the file name)."""
    return CITY_LABELS.get(city, city.title())


def render_sidebar(cities: list[str]) -> tuple[str, int, bool]:
    """Sidebar controls: city, forecast days, compare switch."""
    st.sidebar.header("Налаштування")
    index = cities.index(DEFAULT_CITY) if DEFAULT_CITY in cities else 0
    city = st.sidebar.selectbox("Місто", cities, index=index, format_func=city_label)
    low, high, default = FORECAST_DAYS_RANGE
    days = st.sidebar.slider("Днів прогнозу", low, high, default)
    compare = st.sidebar.toggle("Порівняти міста")
    return city, days, compare


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


def render_forecast(city: str, df: pd.DataFrame, days: int) -> None:
    """History + forecast chart and the backtest error explained in plain words."""
    st.subheader("Історія та прогноз")
    try:
        result, mae = get_forecast(city, days)
    except ValueError as exc:
        st.warning(str(exc))
        st.plotly_chart(charts.line_with_forecast(analysis.daily(df), pd.DataFrame()), width="stretch")
        return
    st.plotly_chart(charts.line_with_forecast(analysis.daily(df), result), width="stretch")
    if mae is None:
        st.caption("Точність прогнозу оцінити не вдалося: замало даних.")
    else:
        st.caption(f"Перевірка на минулому: у середньому ми помиляємось на {mae:.0f} одиниць AQI.")


def render_patterns(df: pd.DataFrame) -> None:
    """Heatmap and year-over-year comparison."""
    st.subheader("Коли повітря найчистіше")
    st.plotly_chart(charts.heatmap_hour_weekday(analysis.hour_weekday_matrix(df)), width="stretch")
    month = int(df["time"].iloc[-1].month)
    st.subheader("Порівняння з минулим роком")
    st.write(f"У {insights.MONTH_NAMES[month - 1]}: {insights.compare_to_last_year(df, month)}.")


def render_comparison(cities: list[str]) -> None:
    """Bar chart of all cities."""
    st.subheader("Порівняння міст")
    frames = {city_label(city): get_city_data(city) for city in cities}
    st.plotly_chart(charts.compare_cities(frames), width="stretch")


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
    city, days, compare = render_sidebar(cities)
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
    render_forecast(city, df, days)
    render_patterns(df)
    if compare:
        render_comparison(cities)
    render_footer()


main()
