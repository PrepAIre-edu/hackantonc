"""Plotly figures. Every chart takes its colours from ``color_for_aqi``."""
import pandas as pd
import plotly.graph_objects as go

from insights import AQI_UPPER_BOUNDS, aqi_band

# One colour per AQI band (green -> red -> maroon), same order as insights.AQI_CATEGORIES.
AQI_COLORS = ["#2e9e5b", "#e3c21f", "#f08a24", "#d64545", "#8e3a9d", "#7a1f2e"]
NO_DATA_COLOR = "#9aa0a6"
HISTORY_COLOR = "#4c78a8"
HISTORY_DAYS_SHOWN = 120
HEATMAP_AQI_MAX = 200
COMPARE_DAYS = 30
WEEKDAY_LABELS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
EMPTY_TEXT = "Немає даних для відображення"
LAYOUT = {"template": "plotly_white", "margin": {"l": 40, "r": 20, "t": 50, "b": 40}}


def color_for_aqi(aqi: float) -> str:
    """Hex colour for an AQI value (grey when missing)."""
    band = aqi_band(aqi)
    return NO_DATA_COLOR if band is None else AQI_COLORS[band]


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


def line_with_forecast(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Daily AQI history (``date, aqi``) plus forecast (``date, yhat, lower, upper``)."""
    history = history.dropna(subset=["aqi"]).tail(HISTORY_DAYS_SHOWN)
    if history.empty and forecast.empty:
        return _empty_figure()
    fig = go.Figure()
    if not history.empty:
        fig.add_scatter(x=history["date"], y=history["aqi"], name="Історія (AQI)", mode="lines+markers",
                        line={"color": HISTORY_COLOR},
                        marker={"color": [color_for_aqi(v) for v in history["aqi"]], "size": 6})
    if not forecast.empty:
        fig.add_scatter(x=forecast["date"], y=forecast["upper"], mode="lines", line={"width": 0},
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=forecast["date"], y=forecast["lower"], mode="lines", line={"width": 0},
                        fill="tonexty", fillcolor="rgba(76,120,168,0.2)", name="Коридор невизначеності")
        fig.add_scatter(x=forecast["date"], y=forecast["yhat"], name="Прогноз", mode="lines+markers",
                        line={"color": HISTORY_COLOR, "dash": "dash"},
                        marker={"color": [color_for_aqi(v) for v in forecast["yhat"]], "size": 7})
    return fig.update_layout(title="AQI: історія та прогноз", xaxis_title="Дата", yaxis_title="AQI",
                             hovermode="x unified", **LAYOUT)


def heatmap_hour_weekday(matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of mean AQI: weekday (rows) by hour (columns)."""
    if matrix.empty or matrix.isna().all().all():
        return _empty_figure()
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=[f"{h:02d}:00" for h in matrix.columns],
        y=[WEEKDAY_LABELS[d] for d in matrix.index], colorscale=_aqi_colorscale(),
        zmin=0, zmax=HEATMAP_AQI_MAX, colorbar={"title": "AQI"},
        hovertemplate="%{y}, %{x}<br>AQI: %{z:.0f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return fig.update_layout(title="Середній AQI: година × день тижня", xaxis_title="Година доби",
                             yaxis_title="День тижня", **LAYOUT)


def compare_cities(dfs: dict[str, pd.DataFrame]) -> go.Figure:
    """Bars of mean AQI over the last 30 days of each city's data."""
    means = {}
    for city, df in dfs.items():
        if df.empty or df["aqi"].notna().sum() == 0:
            continue
        recent = df[df["time"] > df["time"].max() - pd.Timedelta(days=COMPARE_DAYS)]
        means[city] = recent["aqi"].mean()
    means = {city: value for city, value in means.items() if pd.notna(value)}
    if not means:
        return _empty_figure()
    ordered = sorted(means.items(), key=lambda item: item[1], reverse=True)
    cities, values = zip(*ordered)
    fig = go.Figure(go.Bar(x=list(cities), y=list(values), marker_color=[color_for_aqi(v) for v in values],
                           text=[f"{v:.0f}" for v in values], textposition="outside"))
    return fig.update_layout(title=f"Середній AQI за останні {COMPARE_DAYS} днів", yaxis_title="AQI",
                             xaxis_title="Місто", **LAYOUT)
