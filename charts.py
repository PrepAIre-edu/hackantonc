"""Plotly figures. Every chart takes its colours from ``color_for_aqi``."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis import last_days
from insights import AQI_CATEGORIES, AQI_UPPER_BOUNDS, aqi_band

# One colour per AQI band (green -> red -> maroon), same order as insights.AQI_CATEGORIES.
AQI_COLORS = ["#2e9e5b", "#e3c21f", "#f08a24", "#d64545", "#8e3a9d", "#7a1f2e"]
NO_DATA_COLOR = "#9aa0a6"
HISTORY_COLOR = "#4c78a8"
ACCENT_COLOR = "#e05c5c"
HISTORY_DAYS_SHOWN = 120
HEATMAP_AQI_MAX = 200
COMPARE_DAYS = 30
SMOOTHING_WEEKS = 4
WEEKDAY_LABELS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
MONTH_LABELS = ["Січ", "Лют", "Бер", "Кві", "Тра", "Чер", "Лип", "Сер", "Вер", "Жов", "Лис", "Гру"]
MODEL_LABELS = {
    "naive": "Наївна (як учора)",
    "seasonal_naive": "Сезонна наївна (як тиждень тому)",
    "holt_winters": "Хольта–Вінтерса",
}
EMPTY_TEXT = "Немає даних для відображення"
LAYOUT = {"template": "plotly_white", "margin": {"l": 40, "r": 20, "t": 50, "b": 40}}


def color_for_aqi(aqi: float) -> str:
    """Hex colour for an AQI value (grey when missing)."""
    band = aqi_band(aqi)
    return NO_DATA_COLOR if band is None else AQI_COLORS[band]


def _colors(values: pd.Series) -> list[str]:
    """AQI colour of every value (used for markers and bars)."""
    return [color_for_aqi(value) for value in values]


def _empty_figure(text: str = EMPTY_TEXT) -> go.Figure:
    """Empty figure with a centered caption (used instead of raising on empty data)."""
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"size": 16})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.update_layout(**LAYOUT)


def _aqi_colorscale() -> list[list]:
    """Plotly colorscale with hard colour steps at the AQI band bounds."""
    edges = [0, *[bound for bound in AQI_UPPER_BOUNDS if bound < HEATMAP_AQI_MAX], HEATMAP_AQI_MAX]
    scale: list[list] = []
    for color, low, high in zip(AQI_COLORS, edges, edges[1:]):
        scale += [[low / HEATMAP_AQI_MAX, color], [high / HEATMAP_AQI_MAX, color]]
    return scale


def _aqi_heatmap(matrix: pd.DataFrame, x_labels: list[str], y_labels: list[str], title: str,
                 xaxis: str, yaxis: str) -> go.Figure:
    """Shared AQI heatmap (rows = y, columns = x) with the common colour scale."""
    if matrix.empty or matrix.isna().all().all():
        return _empty_figure()
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=x_labels, y=y_labels, colorscale=_aqi_colorscale(),
        zmin=0, zmax=HEATMAP_AQI_MAX, colorbar={"title": "AQI"},
        hovertemplate="%{y}, %{x}<br>AQI: %{z:.0f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return fig.update_layout(title=title, xaxis_title=xaxis, yaxis_title=yaxis, **LAYOUT)


def line_with_forecast(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Daily AQI history (``date, aqi``) plus forecast (``date, yhat, lower, upper``)."""
    history = history.dropna(subset=["aqi"]).tail(HISTORY_DAYS_SHOWN)
    if history.empty and forecast.empty:
        return _empty_figure()
    fig = go.Figure()
    if not history.empty:
        fig.add_scatter(x=history["date"], y=history["aqi"], name="Історія (AQI)", mode="lines+markers",
                        line={"color": HISTORY_COLOR}, marker={"color": _colors(history["aqi"]), "size": 6})
    if not forecast.empty:
        fig.add_scatter(x=forecast["date"], y=forecast["upper"], mode="lines", line={"width": 0},
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=forecast["date"], y=forecast["lower"], mode="lines", line={"width": 0},
                        fill="tonexty", fillcolor="rgba(76,120,168,0.2)", name="Коридор невизначеності")
        fig.add_scatter(x=forecast["date"], y=forecast["yhat"], name="Прогноз", mode="lines+markers",
                        line={"color": HISTORY_COLOR, "dash": "dash"},
                        marker={"color": _colors(forecast["yhat"]), "size": 7})
    return fig.update_layout(title="AQI: історія та прогноз", xaxis_title="Дата", yaxis_title="AQI",
                             hovermode="x unified", **LAYOUT)


def weekly_trend(weekly: pd.DataFrame) -> go.Figure:
    """Weekly mean AQI with a 4-week moving average (``week, aqi``)."""
    weekly = weekly.dropna(subset=["aqi"])
    if weekly.empty:
        return _empty_figure()
    smooth = weekly["aqi"].rolling(SMOOTHING_WEEKS, min_periods=1).mean()
    fig = go.Figure()
    fig.add_scatter(x=weekly["week"], y=weekly["aqi"], name="AQI за тиждень", mode="lines",
                    line={"color": HISTORY_COLOR, "width": 1}, opacity=0.5)
    fig.add_scatter(x=weekly["week"], y=smooth, name=f"Середнє за {SMOOTHING_WEEKS} тижні", mode="lines",
                    line={"color": ACCENT_COLOR, "width": 3})
    return fig.update_layout(title="Тижневий AQI за весь період", xaxis_title="Тиждень", yaxis_title="AQI",
                             hovermode="x unified", **LAYOUT)


def hourly_profile_chart(hourly: pd.DataFrame) -> go.Figure:
    """Mean AQI by hour of day (``hour, aqi``)."""
    hourly = hourly.dropna(subset=["aqi"])
    if hourly.empty:
        return _empty_figure()
    fig = go.Figure(go.Scatter(x=[f"{h:02d}:00" for h in hourly["hour"]], y=hourly["aqi"], mode="lines+markers",
                               line={"color": HISTORY_COLOR}, marker={"color": _colors(hourly["aqi"]), "size": 8}))
    return fig.update_layout(title="Середній AQI за годинами доби", xaxis_title="Година", yaxis_title="AQI",
                             **LAYOUT)


def weekday_profile_chart(weekday: pd.DataFrame) -> go.Figure:
    """Mean AQI by weekday (``weekday, aqi``)."""
    weekday = weekday.dropna(subset=["aqi"])
    if weekday.empty:
        return _empty_figure()
    fig = go.Figure(go.Bar(x=[WEEKDAY_LABELS[d] for d in weekday["weekday"]], y=weekday["aqi"],
                           marker_color=_colors(weekday["aqi"]), text=[f"{v:.0f}" for v in weekday["aqi"]],
                           textposition="outside"))
    return fig.update_layout(title="Середній AQI за днями тижня", xaxis_title="День тижня", yaxis_title="AQI",
                             **LAYOUT)


def heatmap_hour_weekday(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of mean AQI: weekday (rows) by hour (columns)."""
    return _aqi_heatmap(matrix, [f"{h:02d}:00" for h in matrix.columns],
                        [WEEKDAY_LABELS[d] for d in matrix.index],
                        "Середній AQI: година × день тижня", "Година доби", "День тижня")


def heatmap_month_year(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of mean AQI: year (rows) by month (columns)."""
    return _aqi_heatmap(matrix, [MONTH_LABELS[m - 1] for m in matrix.columns], [str(y) for y in matrix.index],
                        "Середній AQI: рік × місяць (опалювальний сезон)", "Місяць", "Рік")


def heatmap_city_hour(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of mean AQI: city (rows) by hour (columns)."""
    return _aqi_heatmap(matrix, [f"{h:02d}:00" for h in matrix.columns], list(matrix.index),
                        "Середній AQI за останній рік: місто × година", "Година доби", "Місто")


def category_share_chart(shares: pd.DataFrame) -> go.Figure:
    """Stacked bars: % of days in each AQI category per year (rows = year, columns = band)."""
    if shares.empty:
        return _empty_figure()
    fig = go.Figure()
    for band, name in enumerate(AQI_CATEGORIES):
        if band in shares.columns and shares[band].sum() > 0:
            fig.add_bar(x=[str(y) for y in shares.index], y=shares[band], name=name, marker_color=AQI_COLORS[band])
    return fig.update_layout(barmode="stack", title="Частка днів за категоріями якості повітря",
                             xaxis_title="Рік", yaxis_title="% днів", **LAYOUT)


def compare_cities(dfs: dict[str, pd.DataFrame]) -> go.Figure:
    """Bars of mean AQI over the last 30 days of each city's data."""
    means = {city: last_days(df, COMPARE_DAYS)["aqi"].mean() for city, df in dfs.items() if not df.empty}
    means = {city: value for city, value in means.items() if pd.notna(value)}
    if not means:
        return _empty_figure()
    cities, values = zip(*sorted(means.items(), key=lambda item: item[1], reverse=True))
    fig = go.Figure(go.Bar(x=list(cities), y=list(values), marker_color=[color_for_aqi(v) for v in values],
                           text=[f"{v:.0f}" for v in values], textposition="outside"))
    return fig.update_layout(title=f"Середній AQI за останні {COMPARE_DAYS} днів", yaxis_title="AQI",
                             xaxis_title="Місто", **LAYOUT)


def city_boxplot(daily_by_city: pd.DataFrame) -> go.Figure:
    """Distribution of daily AQI per city over the last year (``city, date, aqi``)."""
    if daily_by_city.empty:
        return _empty_figure()
    order = daily_by_city.groupby("city")["aqi"].median().sort_values(ascending=False).index
    fig = go.Figure()
    for city in order:
        values = daily_by_city.loc[daily_by_city["city"] == city, "aqi"]
        fig.add_box(y=values, name=city, marker_color=color_for_aqi(values.median()), boxmean=True)
    return fig.update_layout(title="Розкид добового AQI за останній рік", yaxis_title="AQI", showlegend=False,
                             **LAYOUT)


def decomposition_chart(parts: pd.DataFrame) -> go.Figure:
    """Four stacked panels: observed, trend, seasonal, residual (``date, observed, trend, seasonal, resid``)."""
    if parts.empty:
        return _empty_figure()
    titles = {"observed": "Спостереження", "trend": "Тренд", "seasonal": "Сезонність", "resid": "Залишок"}
    fig = make_subplots(rows=len(titles), cols=1, shared_xaxes=True, subplot_titles=list(titles.values()),
                        vertical_spacing=0.06)
    for row, column in enumerate(titles, start=1):
        fig.add_scatter(x=parts["date"], y=parts[column], mode="lines", showlegend=False,
                        line={"color": ACCENT_COLOR if column == "trend" else HISTORY_COLOR, "width": 1.5},
                        row=row, col=1)
    return fig.update_layout(title="Розклад добового AQI: тренд і сезонність", height=700, **LAYOUT)


def predicted_vs_actual(backtest: pd.DataFrame) -> go.Figure:
    """Scatter of forecast vs real daily AQI from ``rolling_backtest`` (perfect forecast = diagonal)."""
    if backtest.empty:
        return _empty_figure()
    top = float(max(backtest["actual"].max(), backtest["holt_winters"].max()))
    fig = go.Figure()
    fig.add_scatter(x=[0, top], y=[0, top], mode="lines", name="Ідеальний прогноз",
                    line={"color": NO_DATA_COLOR, "dash": "dash"})
    fig.add_scatter(x=backtest["actual"], y=backtest["holt_winters"], mode="markers", name="Прогноз",
                    marker={"color": _colors(backtest["actual"]), "size": 7, "opacity": 0.8})
    return fig.update_layout(title="Прогноз проти реальності (перевірка на минулому)",
                             xaxis_title="Реальний AQI", yaxis_title="Прогнозований AQI", **LAYOUT)


def model_comparison(errors: pd.DataFrame) -> go.Figure:
    """Bars of MAE per model (``model, mae``); the lower the better."""
    if errors.empty:
        return _empty_figure()
    labels = [MODEL_LABELS.get(model, model) for model in errors["model"]]
    fig = go.Figure(go.Bar(x=labels, y=errors["mae"], marker_color=HISTORY_COLOR,
                           text=[f"{v:.1f}" for v in errors["mae"]], textposition="outside"))
    return fig.update_layout(title="Помилка моделей (MAE, менше — краще)", yaxis_title="MAE, одиниць AQI",
                             **LAYOUT)
