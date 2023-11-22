import plotly.graph_objs as go
from models.models import QueryEntry, TrendType


def generate_trend_chart(entry: QueryEntry):
    # Extracting the x and y values from the data
    x_values = list(range(entry.start_year, entry.end_year + 1))
    y_values = entry.results["search_results"]["adjusted"]

    # Creating the line chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines",
                  line=dict(color="gray"), name="Publications"))

    # Highlighting the trends
    for trend in entry.results["trend_results"]["sub_trends"]:
        if trend["type"] == TrendType.NONE:
            continue

        color = "green" if trend["type"] == TrendType.INCREASING else "red"

        start = trend["start"] - entry.start_year
        end = trend["end"] - entry.start_year

        fig.add_trace(go.Scatter(
            x=(list(range(trend["start"], trend["end"] + 1))),
            y=y_values[start:end + 1],
            mode="lines",
            line=dict(color=color), name="Publications")
        )

    # Breakpoints
    for bp in entry.results["trend_results"]["breakpoints"]:
        fig.add_shape(type="line", x0=bp, y0=0, x1=bp, y1=max(y_values),
                      line=dict(color="red", width=1, dash="dash"))

    fig.update_layout(showlegend=False, template="plotly_white",
                      margin=dict(l=10, r=10, b=10, t=10))

    return fig.to_image(format="png", width=1200, height=500, scale=1)
