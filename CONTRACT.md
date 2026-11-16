# Контракт функцій

`df` — погодинна таблиця з `loader.load_city`: колонки `time` (datetime), `pm25`, `aqi` (пропуски = NaN).

## loader.py
| Функція | Вхід | Вихід |
|---|---|---|
| `list_cities()` | — | `list[str]`, імена файлів `data/*.csv` без розширення |
| `load_city(city)` | `city: str` (регістр неважливий) | `DataFrame[time, pm25, aqi]`, відсортований, без дублів. `FileNotFoundError` зі списком міст, `ValueError` для битого файлу |
| `city_label(city)` | `str` | `str`, українська назва («kyiv» → «Київ») |
| `pm25_to_aqi(pm25)` | `pd.Series` | `pd.Series` AQI за таблицею US EPA (`EPA_PM25_BREAKPOINTS`) |

## analysis.py
| Функція | Вхід | Вихід |
|---|---|---|
| `daily(df)` | `df` | `DataFrame[date, pm25, aqi]` |
| `hourly_profile(df)` | `df` | `DataFrame[hour 0–23, pm25, aqi]` |
| `weekday_profile(df)` | `df` | `DataFrame[weekday 0=Пн…6=Нд, pm25, aqi]` |
| `hour_weekday_matrix(df)` | `df` | `DataFrame` 7×24: рядки — день тижня, колонки — година, значення — середній AQI |
| `monthly(df)` | `df` | `DataFrame[year, month, pm25, aqi]` |
| `last_days(df, days)` | `df`, `days: int` | рядки останніх `days` днів (вікно закінчується на останньому вимірюванні) |
| `weekly(df)` | `df` | `DataFrame[week, pm25, aqi]`, тижні з понеділка |
| `yearly(df)` | `df` | `DataFrame[year, pm25, aqi, days]` |
| `month_year_matrix(df)` | `df` | `DataFrame`: рядки — рік, колонки — місяць 1–12, значення — середній AQI |
| `city_stats(dfs)` | `dict[str, DataFrame]` | `DataFrame[city, mean_aqi, median_aqi, p95_aqi, unhealthy_share, best_hour, worst_hour]` за останній рік |
| `city_hour_matrix(dfs)` | `dict[str, DataFrame]` | `DataFrame`: рядки — місто, колонки — година 0–23 |
| `daily_by_city(dfs)` | `dict[str, DataFrame]` | `DataFrame[city, date, aqi]` за останній рік |
| `summary(df)` | `df` | `dict`: `last_time`, `last_pm25`, `last_aqi`, `mean_pm25_30d`, `mean_aqi_30d`, `worst_day`, `worst_day_aqi` (`None`, якщо даних немає) |

## forecast.py
| Функція | Вхід | Вихід |
|---|---|---|
| `forecast_aqi(df, days=7)` | `df`, `days: int` | `DataFrame[date, yhat, lower, upper]`, `days` рядків, `0 ≤ lower ≤ yhat ≤ upper`. `ValueError`, якщо даних менше 10 днів |
| `backtest_mae(df, days=7)` | `df`, `days: int` | `float`, MAE прогнозу на відкладеному хвості |
| `rolling_backtest(df, days=7, folds=8)` | `df` | `DataFrame[date, actual, naive, seasonal_naive, holt_winters]` |
| `model_errors(backtest)` | результат `rolling_backtest` | `DataFrame[model, mae]`, найкраща модель першою |
| `decompose_daily(df, period=365)` | `df`, `period: int` | `DataFrame[date, observed, trend, seasonal, resid]`. `ValueError`, якщо даних менше ніж `2 × period` днів |

## insights.py
| Функція | Вхід | Вихід |
|---|---|---|
| `aqi_band(aqi)` | `float` | `int` 0–5 або `None` для NaN |
| `aqi_category(aqi)` | `float` | `str`: «Добре» … «Небезпечно» або «Немає даних» |
| `advice(aqi)` | `float` | `str`, коротка порада |
| `best_hour(hourly)` | результат `hourly_profile` | `str`, напр. `"06:00"` |
| `category_share_by_year(df)` | `df` | `DataFrame`: рядки — рік, колонки — категорія 0–5, значення — % днів |
| `key_findings(df)` | `df` | `list[str]`, прості висновки про місто |
| `compare_to_last_year(df, month)` | `df`, `month: int` 1–12 | `str`, напр. «на 12% чистіше, ніж торік» |

## charts.py (повертають `plotly.graph_objects.Figure`; на порожніх даних — порожній Figure з підписом)
| Функція | Вхід | Вихід |
|---|---|---|
| `color_for_aqi(aqi)` | `float` | `str`, hex-колір |
| `line_with_forecast(history, forecast)` | `daily(df)` і `forecast_aqi(df)` | `Figure` |
| `heatmap_hour_weekday(matrix)` | `hour_weekday_matrix(df)` | `Figure` |
| `compare_cities(dfs)` | `dict[str, DataFrame]` | `Figure`, середній AQI за останні 30 днів |
| `weekly_trend(weekly)` | `weekly(df)` | `Figure`, тижневий AQI і ковзне середнє |
| `hourly_profile_chart(hourly)` | `hourly_profile(df)` | `Figure` |
| `weekday_profile_chart(weekday)` | `weekday_profile(df)` | `Figure` |
| `heatmap_month_year(matrix)` | `month_year_matrix(df)` | `Figure` |
| `heatmap_city_hour(matrix)` | `city_hour_matrix(dfs)` | `Figure` |
| `category_share_chart(shares)` | `category_share_by_year(df)` | `Figure`, стовпчики з накопиченням |
| `city_boxplot(daily_by_city)` | `daily_by_city(dfs)` | `Figure`, розкид добового AQI |
| `decomposition_chart(parts)` | `decompose_daily(df)` | `Figure` із 4 панелями |
| `predicted_vs_actual(backtest)` | `rolling_backtest(df)` | `Figure`, прогноз проти реальності |
| `model_comparison(errors)` | `model_errors(backtest)` | `Figure`, MAE по моделях |

## report.py
| Функція | Вхід | Вихід |
|---|---|---|
| `build_figures(city, dfs)` | назва міста, `dict[str, DataFrame]` | `dict[str, Figure]`, ім'я файлу → графік |
| `save_figures(figures, plots_dir)` | графіки, папка | PNG-файли |
| `models_by_city(dfs)` | `dict[str, DataFrame]` | `DataFrame` MAE: рядки — міста, колонки — моделі |
| `build_summary(city, dfs)` | назва міста, `dict[str, DataFrame]` | `str`, текстовий звіт |
| `main(city="kyiv")` | назва міста | пише `outputs/plots/*.png`, `outputs/reports/*` |
