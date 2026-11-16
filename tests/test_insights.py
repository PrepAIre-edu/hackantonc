import math

import analysis
import insights


def test_category_boundaries():
    cases = {0: "Добре", 50: "Добре", 51: "Помірно", 100: "Помірно", 101: "Шкідливо для чутливих",
             150: "Шкідливо для чутливих", 151: "Шкідливо", 200: "Шкідливо", 201: "Дуже шкідливо",
             300: "Дуже шкідливо", 301: "Небезпечно"}
    for aqi, expected in cases.items():
        assert insights.aqi_category(aqi) == expected


def test_missing_aqi_is_handled():
    assert insights.aqi_category(math.nan) == insights.NO_DATA
    assert insights.advice(math.nan) == insights.NO_DATA
    assert "вікна" in insights.advice(160)


def test_best_hour_format(hourly_df):
    result = insights.best_hour(analysis.hourly_profile(hourly_df))
    assert result == "18:00"  # sin is minimal at hour 18


def test_compare_to_last_year(hourly_df):
    assert "минулого року немає" in insights.compare_to_last_year(hourly_df, 1)
    older = hourly_df.copy()
    older["time"] = older["time"] - __import__("pandas").DateOffset(years=1)
    older["aqi"] = older["aqi"] * 2  # last year was twice as dirty
    both = __import__("pandas").concat([older, hourly_df], ignore_index=True)
    assert insights.compare_to_last_year(both, 1) == "на 50% чистіше, ніж торік"
