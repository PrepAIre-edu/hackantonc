import pandas as pd
import plotly.graph_objects as go

import analysis
import charts
import forecast


def test_charts_return_figures(hourly_df):
    daily = analysis.daily(hourly_df)
    figures = [
        charts.line_with_forecast(daily, forecast.forecast_aqi(hourly_df)),
        charts.heatmap_hour_weekday(analysis.hour_weekday_matrix(hourly_df)),
        charts.compare_cities({"a": hourly_df, "b": hourly_df}),
    ]
    assert all(isinstance(fig, go.Figure) and len(fig.data) > 0 for fig in figures)


def test_charts_survive_empty_data(hourly_df):
    empty = hourly_df.iloc[:0]
    figures = [
        charts.line_with_forecast(analysis.daily(empty), pd.DataFrame()),
        charts.heatmap_hour_weekday(pd.DataFrame()),
        charts.compare_cities({}),
        charts.compare_cities({"a": empty}),
    ]
    assert all(isinstance(fig, go.Figure) and len(fig.data) == 0 for fig in figures)


def test_color_for_aqi_is_consistent():
    assert charts.color_for_aqi(10) != charts.color_for_aqi(250)
    assert charts.color_for_aqi(float("nan")) == charts.NO_DATA_COLOR
