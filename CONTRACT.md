# Контракт функцій

`df` — погодинна таблиця з `loader.load_city`: колонки `time` (datetime), `pm25`, `aqi` (пропуски = NaN).

## loader.py
| Функція | Вхід | Вихід |
|---|---|---|
| `list_cities()` | — | `list[str]`, імена файлів `data/*.csv` без розширення |
| `load_city(city)` | `city: str` (регістр неважливий) | `DataFrame[time, pm25, aqi]`, відсортований, без дублів. `FileNotFoundError` зі списком міст, `ValueError` для битого файлу |
| `pm25_to_aqi(pm25)` | `pd.Series` | `pd.Series` AQI за таблицею US EPA (`EPA_PM25_BREAKPOINTS`) |

## analysis.py
| Функція | Вхід | Вихід |
|---|---|---|
| `daily(df)` | `df` | `DataFrame[date, pm25, aqi]` |
| `hourly_profile(df)` | `df` | `DataFrame[hour 0–23, pm25, aqi]` |
| `weekday_profile(df)` | `df` | `DataFrame[weekday 0=Пн…6=Нд, pm25, aqi]` |
| `hour_weekday_matrix(df)` | `df` | `DataFrame` 7×24: рядки — день тижня, колонки — година, значення — середній AQI |
| `monthly(df)` | `df` | `DataFrame[year, month, pm25, aqi]` |
| `summary(df)` | `df` | `dict`: `last_time`, `last_pm25`, `last_aqi`, `mean_pm25_30d`, `mean_aqi_30d`, `worst_day`, `worst_day_aqi` (`None`, якщо даних немає) |

## forecast.py
| Функція | Вхід | Вихід |
|---|---|---|
| `forecast_aqi(df, days=7)` | `df`, `days: int` | `DataFrame[date, yhat, lower, upper]`, `days` рядків, `0 ≤ lower ≤ yhat ≤ upper`. `ValueError`, якщо даних менше 10 днів |
| `backtest_mae(df, days=7)` | `df`, `days: int` | `float`, MAE прогнозу на відкладеному хвості |

## insights.py
| Функція | Вхід | Вихід |
|---|---|---|
| `aqi_band(aqi)` | `float` | `int` 0–5 або `None` для NaN |
| `aqi_category(aqi)` | `float` | `str`: «Добре» … «Небезпечно» або «Немає даних» |
| `advice(aqi)` | `float` | `str`, коротка порада |
| `best_hour(hourly)` | результат `hourly_profile` | `str`, напр. `"06:00"` |
| `compare_to_last_year(df, month)` | `df`, `month: int` 1–12 | `str`, напр. «на 12% чистіше, ніж торік» |

## charts.py (повертають `plotly.graph_objects.Figure`; на порожніх даних — порожній Figure з підписом)
| Функція | Вхід | Вихід |
|---|---|---|
| `color_for_aqi(aqi)` | `float` | `str`, hex-колір |
| `line_with_forecast(history, forecast)` | `daily(df)` і `forecast_aqi(df)` | `Figure` |
| `heatmap_hour_weekday(matrix)` | `hour_weekday_matrix(df)` | `Figure` |
| `compare_cities(dfs)` | `dict[str, DataFrame]` | `Figure`, середній AQI за останні 30 днів |
