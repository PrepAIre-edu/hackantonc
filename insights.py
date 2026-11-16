"""Human-readable texts: AQI categories, advice, best hour, year-over-year change."""
import bisect
import math

import pandas as pd

from analysis import monthly

# Upper AQI bound (inclusive) of each band, US EPA style. The last band is open-ended.
AQI_UPPER_BOUNDS = [50, 100, 150, 200, 300]
AQI_CATEGORIES = [
    "Добре",
    "Помірно",
    "Шкідливо для чутливих",
    "Шкідливо",
    "Дуже шкідливо",
    "Небезпечно",
]
AQI_ADVICE = [
    "Повітря чисте. Гуляйте, провітрюйте квартиру, займайтеся спортом надворі.",
    "Прийнятно для більшості. Дуже чутливим людям краще скоротити тривале навантаження надворі.",
    "Закрийте вікна, якщо ви чутливі до забруднення. Дітям, літнім людям і астматикам краще менше бувати надворі.",
    "Закрийте вікна. Обмежте прогулянки та спорт надворі, на вулиці бажана маска FFP2.",
    "Закрийте вікна, увімкніть очисник повітря. Не виходьте надовго на вулицю.",
    "Небезпечне повітря. Залишайтеся вдома з закритими вікнами, на вулиці тільки в респіраторі.",
]
NO_DATA = "Немає даних"
MONTH_NAMES = [
    "січні", "лютому", "березні", "квітні", "травні", "червні",
    "липні", "серпні", "вересні", "жовтні", "листопаді", "грудні",
]
SAME_LEVEL_PERCENT = 1.0  # changes below this are reported as "about the same"


def aqi_band(aqi: float) -> int | None:
    """Index (0-5) of the AQI band, or ``None`` for a missing value."""
    if aqi is None or math.isnan(aqi):
        return None
    return bisect.bisect_left(AQI_UPPER_BOUNDS, aqi)


def aqi_category(aqi: float) -> str:
    """Category name for an AQI value, e.g. 42 -> "Добре"."""
    band = aqi_band(aqi)
    return NO_DATA if band is None else AQI_CATEGORIES[band]


def advice(aqi: float) -> str:
    """Short advice for a person at the given AQI."""
    band = aqi_band(aqi)
    return NO_DATA if band is None else AQI_ADVICE[band]


def best_hour(hourly: pd.DataFrame) -> str:
    """Hour of day with the lowest AQI, e.g. "06:00" (input: ``hourly_profile``)."""
    column = "aqi" if hourly["aqi"].notna().any() else "pm25"
    if hourly.empty or hourly[column].notna().sum() == 0:
        return NO_DATA
    hour = int(hourly.loc[hourly[column].idxmin(), "hour"])
    return f"{hour:02d}:00"


def compare_to_last_year(df: pd.DataFrame, month: int) -> str:
    """Compare the latest available ``month`` with the same month a year earlier.

    Example: "на 12% чистіше, ніж торік". The latest month may be incomplete
    (e.g. the current one); we still compare month means, which is good enough.
    """
    table = monthly(df).dropna(subset=["aqi"])
    table = table[table["month"] == month].set_index("year")["aqi"]
    if table.empty:
        return f"Даних у {MONTH_NAMES[month - 1]} немає."
    year = int(table.index.max())
    if year - 1 not in table.index or table[year - 1] <= 0:
        return f"Даних у {MONTH_NAMES[month - 1]} минулого року немає, порівняти неможливо."
    change = (table[year] - table[year - 1]) / table[year - 1] * 100
    if abs(change) < SAME_LEVEL_PERCENT:
        return "приблизно так само, як торік"
    word = "чистіше" if change < 0 else "брудніше"
    return f"на {abs(change):.0f}% {word}, ніж торік"
