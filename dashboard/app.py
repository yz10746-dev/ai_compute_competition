"""Interactive dashboard of Epoch AI compute build-out and model frontier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, callback, ctx, dcc, html, no_update

from dashboard.data import (
    AS_OF,
    CHIP_PATTERNS,
    CHIPS,
    COLORS,
    FOCUS_OWNERS,
    LABS,
    MONTHS,
    QUARTERS,
    REVENUE_COMPANIES,
    company_order,
    format_h100,
    load,
    parse_bound,
    quarter_date,
    quarter_index,
    view_state,
)

EMPTY = {
    "companies": [],
    "campus": None,
    "model_companies": None,
    "date_start": None,
    "date_end": None,
    "role": "owner",
    "chips": [],
    "status": "all",
}
COMPANIES = company_order()
STACK_ORDER = COMPANIES + ["Other", "Unattributed"]
LAST_QUARTER = len(QUARTERS) - 1
FULL_RANGE = [MONTHS[0].date().isoformat(), MONTHS[-1].date().isoformat()]
TODAY = AS_OF.isoformat()
TOP_CAMPUSES = 15

FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif'
INK = "#111113"
MUTED = "#6b6f76"
GRID = "#eef0f3"
MARGIN = dict(l=64, r=16, t=12, b=40)
GRAPH_CONFIG = {
    "displaylogo": False,
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

app = Dash(
    __name__,
    assets_folder=str(Path(__file__).resolve().parents[1] / "assets"),
    title="AI Compute Competition by 2030",
)
server = app.server


def _base_layout(height: int, **extra) -> dict:
    layout = dict(
        template="plotly_white",
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=MARGIN,
        hovermode="closest",
        showlegend=False,
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    )
    layout.update(extra)
    return layout


def _axis(title: str, **extra) -> dict:
    axis = dict(
        title=dict(text=title, font=dict(size=11, color=MUTED)),
        tickfont=dict(size=11, color=MUTED),
        gridcolor=GRID,
        zeroline=False,
        showline=False,
    )
    axis.update(extra)
    return axis


def _empty_figure(message: str, height: int) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        **_base_layout(height),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                 font=dict(size=13, color=MUTED))
        ],
    )
    return figure


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _window(state: dict) -> list[str]:
    if state["date_start"] and state["date_end"]:
        return [state["date_start"], state["date_end"]]
    return FULL_RANGE


def _si(value) -> str:
    if pd.isna(value):
        return "—"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.3g}"


def _today_rule(figure: go.Figure) -> None:
    figure.add_vline(x=TODAY, line=dict(color=INK, width=1, dash="dot"), opacity=0.45)
    figure.add_annotation(x=TODAY, y=1, yref="paper", yanchor="bottom", text="Today", showarrow=False,
                          font=dict(size=10, color=MUTED), xshift=18)


def area_figure(state: dict) -> go.Figure:
    series = state["series"]
    if series.empty or series.to_numpy().sum() == 0:
        return _empty_figure("No compute", 330)

    window = _window(state)
    series = series.loc[window[0]:window[1]]
    campus_series = state["campus_series"]
    if campus_series is not None:
        campus_series = campus_series.loc[window[0]:window[1]]
    dim = campus_series is not None
    figure = go.Figure()
    for company in [name for name in STACK_ORDER if name in series.columns]:
        color = COLORS.get(company, COLORS["Other"])
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series[company],
                name=company,
                mode="lines",
                stackgroup="compute",
                line=dict(width=0.6, color=_rgba(color, 0.35 if dim else 1)),
                fillcolor=_rgba(color, 0.25 if dim else 0.88),
                customdata=[company] * len(series),
                hovertemplate="%{customdata} %{y:.3s}<extra></extra>",
            )
        )
    if dim:
        figure.add_trace(
            go.Scatter(
                x=campus_series.index,
                y=campus_series.to_numpy(),
                name=state["campus"],
                mode="lines",
                line=dict(width=2.4, color=INK, shape="hv"),
                customdata=[state["campus"]] * len(campus_series),
                hovertemplate="%{customdata} %{y:.3s}<extra></extra>",
            )
        )
    _today_rule(figure)
    figure.update_layout(
        **_base_layout(330, hovermode="x unified", uirevision="|".join(_window(state))),
        xaxis=_axis("", range=_window(state), showgrid=False),
        yaxis=_axis("H100e Equivalent", tickformat=".2s", fixedrange=True, rangemode="tozero"),
    )
    return figure


def _campus_color(row: pd.Series, state: dict) -> str:
    if state["role"] == "user":
        users = row["users"]
        picked = [user for user in users if user in state["companies"]]
        if picked:
            return picked[0]
        if not users:
            return "Unattributed"
        return users[0] if users[0] in COLORS else "Other"
    return row["owner"] if row["owner"] in FOCUS_OWNERS else "Other"


def campus_bars_figure(state: dict) -> go.Figure:
    frame = state["campuses"]
    frame = frame.loc[frame["h100e_end"] > 0].nlargest(TOP_CAMPUSES, "h100e_end").copy()
    if frame.empty:
        return _empty_figure("No campuses", 660)

    frame["color"] = frame.apply(lambda row: _campus_color(row, state), axis=1)
    order = frame["name"].tolist()[::-1]
    figure = go.Figure()
    for color_name, group in frame.groupby("color", sort=False):
        color = COLORS.get(color_name, COLORS["Other"])
        selected = [name == state["campus"] for name in group["name"]]
        figure.add_trace(
            go.Bar(
                x=group["h100e_end"],
                y=group["name"],
                orientation="h",
                name=color_name,
                marker=dict(
                    color=color,
                    line=dict(color=INK, width=[2 if flag else 0 for flag in selected]),
                    pattern=dict(
                        shape=[CHIP_PATTERNS[family] for family in group["chip_family"]],
                        fgcolor="rgba(255,255,255,0.75)",
                        size=6,
                        solidity=0.35,
                    ),
                ),
                customdata=list(
                    zip(
                        group["name"],
                        group["owner"].replace("", "—"),
                        group["users"].map(lambda users: " · ".join(users) if users else "—"),
                        group["chip_family"],
                        group["country"],
                    )
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{x:.3s} H100e<br>"
                    "%{customdata[1]} → %{customdata[2]}<br>%{customdata[3]} · %{customdata[4]}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    for name in order:
        figure.add_annotation(x=0, y=name, xref="x", yref="y", xanchor="left", yanchor="bottom", yshift=8,
                              text=name, showarrow=False, font=dict(size=11, color=INK))
    figure.update_layout(
        **_base_layout(
            660,
            barmode="overlay",
            bargap=0.58,
            margin=dict(l=4, r=12, t=24, b=8),
        ),
        xaxis=_axis("", tickformat=".2s", side="top", fixedrange=True, rangemode="tozero"),
        yaxis=_axis("", categoryorder="array", categoryarray=order, fixedrange=True, showticklabels=False,
                    showgrid=False),
    )
    return figure


def frontier_figure(state: dict) -> go.Figure:
    models = state["models"]
    if models.empty:
        return _empty_figure("No models", 300)

    figure = go.Figure()
    for company in [name for name in STACK_ORDER if name in set(models["company"])]:
        group = models.loc[models["company"] == company]
        color = COLORS.get(company, COLORS["Other"])
        line_x = list(group["date"]) + [pd.Timestamp(AS_OF)]
        line_y = list(group["best"]) + [group["best"].iloc[-1]]
        figure.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color=color, width=2, shape="hv"),
                hoverinfo="skip",
                name=company,
            )
        )
        params = group["params"].map(_si)
        figure.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["compute"],
                mode="markers",
                name=company,
                marker=dict(
                    color=color,
                    size=[8 if flag else 5 for flag in group["record"]],
                    opacity=[1 if flag else 0.22 for flag in group["record"]],
                    line=dict(color="white", width=[1 if flag else 0 for flag in group["record"]]),
                ),
                customdata=list(zip(group["model"], [company] * len(group), params)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]} · %{x|%Y-%m-%d}<br>"
                    "%{y:.2e} FLOP · %{customdata[2]} params<extra></extra>"
                ),
            )
        )
    _today_rule(figure)
    figure.update_layout(
        **_base_layout(300, uirevision="|".join(_window(state))),
        xaxis=_axis("", range=_window(state), showgrid=False),
        yaxis=_axis("FLOP", type="log", exponentformat="power", dtick=2),
    )
    return figure


MAX_COMPARE = 3


def _latest_staff(staff: pd.DataFrame, company: str):
    rows = staff.loc[staff["company"] == company]
    return None if rows.empty else float(rows.iloc[-1]["staff"])


SUMMARY_COLUMNS = [
    ("Campuses", "campuses", lambda value: f"{value}"),
    ("H100e", "h100e", format_h100),
    ("$B", "cost_b", lambda value: f"{value:.0f}"),
]


def _compare_table(rows: list[dict], labels: list, columns: list):
    leaders = {}
    if len(rows) > 1:
        for _, key, _ in columns:
            values = [row[key] for row in rows if row.get(key) is not None]
            leaders[key] = max(values) if values else None

    def cell(row, key, fmt):
        lead = leaders.get(key) is not None and row.get(key) == leaders[key]
        return html.Td(fmt(row.get(key)), className="lead" if lead else None)

    header = html.Tr([html.Th("")] + [html.Th(label) for label, _, _ in columns])
    body = [
        html.Tr([html.Td(label, className="name")] + [cell(row, key, fmt) for _, key, fmt in columns])
        for row, label in zip(rows, labels)
    ]
    return html.Table([html.Thead(header), html.Tbody(body)], className="compare")


def _company_label(name: str):
    return html.Span([html.Span(className="swatch", style={"background": COLORS[name]}), name], className="row-label")


def _chip_label(family: str):
    return html.Span([html.Span(className=f"pattern pattern-{family.lower()}"), family], className="row-label")


def chip_detail_children(state: dict):
    rows = state["chip_summary"]
    if not rows or len(rows) > MAX_COMPARE:
        return []
    return [_compare_table(rows, [_chip_label(row["chip"]) for row in rows], SUMMARY_COLUMNS)]


def detail_children(state: dict):
    summary = state["summary"]
    if not summary or len(summary) > MAX_COMPARE:
        return []
    labs = [row["company"] for row in summary if row["company"] in REVENUE_COMPANIES]
    columns = list(SUMMARY_COLUMNS)
    if labs:
        for row in summary:
            row["staff"] = _latest_staff(state["staff"], row["company"])
        columns.append(("Staff", "staff", lambda value: "—" if value is None else format_h100(value)))

    return [_compare_table(summary, [_company_label(row["company"]) for row in summary], columns)]


def kpi_children(state: dict):
    capacity_label = "Capacity (H100 equivalent)"
    if state["date_end"]:
        capacity_label += f" · {state['end_month']:%b %Y}"
    items = [
        (f"{int((state['campuses']['h100e_end'] > 0).sum())}", "Campuses"),
        (format_h100(state["h100e_end"]), capacity_label),
        (f"{state['models_in_window']['model'].nunique()}", "Notable models"),
    ]
    return [
        html.Div([html.Div(value, className="value"), html.Div(label, className="label")], className="kpi")
        for value, label in items
    ]


def status_text(state: dict) -> str:
    parts = []
    if state["campus"]:
        parts.append(state["campus"])
    if state["date_start"] and state["date_end"]:
        parts.append(f"{state['date_start']} – {state['date_end']}")
    return " · ".join(parts)


def _slider_value(selection: dict) -> list[int]:
    if selection.get("date_start") and selection.get("date_end"):
        return [quarter_index(selection["date_start"]), quarter_index(selection["date_end"], round_up=True)]
    return [0, LAST_QUARTER]


def _dates_from_quarters(low: int, high: int) -> tuple[str | None, str | None]:
    if low <= 0 and high >= LAST_QUARTER:
        return None, None
    return quarter_date(low), quarter_date(high)


def _range_from_relayout(relayout: dict):
    if not relayout:
        return None
    if relayout.get("xaxis.autorange"):
        return "reset"
    start = relayout.get("xaxis.range[0]")
    end = relayout.get("xaxis.range[1]")
    if start is None and isinstance(relayout.get("xaxis.range"), (list, tuple)):
        start, end = relayout["xaxis.range"]
    start_date, end_date = parse_bound(start), parse_bound(end)
    if not start_date or not end_date or start_date >= end_date:
        return None
    return _dates_from_quarters(quarter_index(start_date), quarter_index(end_date, round_up=True))


def _pick_company(selection: dict, name: str) -> dict:
    chosen = list(selection.get("companies") or [])
    if name in chosen:
        chosen.remove(name)
        role = selection.get("role")
    else:
        chosen.append(name)
        role = "user" if name in LABS and selection.get("role") == "owner" else selection.get("role")
    chosen = [company for company in COMPANIES if company in chosen]
    return {**selection, "companies": chosen, "campus": None, "model_companies": None, "role": role}


def _pick_chip(selection: dict, family: str) -> dict:
    chosen = list(selection.get("chips") or [])
    chosen = [item for item in chosen if item != family] if family in chosen else chosen + [family]
    return {**selection, "chips": [item for item in CHIPS if item in chosen], "campus": None, "model_companies": None}


def _pick_campus(selection: dict, point: dict) -> dict:
    campus, owner, users_label = point["customdata"][:3]
    if selection.get("campus") == campus:
        return {**selection, "campus": None, "model_companies": None}
    users = [] if users_label == "—" else users_label.split(" · ")
    model_companies = [owner] if owner in FOCUS_OWNERS else []
    model_companies += [user for user in users if user in COLORS and user not in model_companies]
    return {**selection, "campus": campus, "model_companies": model_companies or None}


app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            className="mast",
            children=[
                html.Div(
                    [
                        html.H1("AI Compute Competition by 2030"),
                        html.P(
                            "Who builds and who uses AI compute: capacity, campuses, and models of the major players.",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(id="kpis", className="kpis"),
            ],
        ),
        html.Div(
            className="controls",
            children=[
                html.Div(
                    className="control control-range",
                    children=dcc.RangeSlider(
                        id="date-slider",
                        min=0,
                        max=LAST_QUARTER,
                        step=1,
                        value=[0, LAST_QUARTER],
                        allowCross=False,
                        allow_direct_input=False,
                        updatemode="drag",
                        marks={index: str(QUARTERS[index].year) for index in range(0, LAST_QUARTER + 1, 8)},
                        tooltip=None,
                    ),
                ),
                dcc.RadioItems(
                    id="status",
                    className="segmented",
                    options=[
                        {"label": "All", "value": "all"},
                        {"label": "Operating", "value": "operating"},
                        {"label": "Planned", "value": "planned"},
                    ],
                    value="all",
                    inline=True,
                ),
                html.Div(
                    className="criteria",
                    children=[
                        html.Span(
                            [
                                "Stat Criteria",
                                html.Span("?", className="hint-icon"),
                                html.Span(
                                    [
                                        html.B("Owner"),
                                        " counts campuses a company owns. ",
                                        html.B("User"),
                                        " counts campuses it trains on, split evenly among users.",
                                        html.Br(),
                                        html.Br(),
                                        "OpenAI owns none, so Owner shows 0. As a user of Stargate "
                                        "and Microsoft Fairwater, it gets 12 campuses and 24M H100e.",
                                    ],
                                    className="hint-pop",
                                ),
                            ],
                            className="criteria-label",
                            tabIndex="0",
                        ),
                        dcc.RadioItems(
                            id="role",
                            className="segmented",
                            options=[{"label": "Owner", "value": "owner"}, {"label": "User", "value": "user"}],
                            value="owner",
                            inline=True,
                        ),
                    ],
                ),
                html.Button("Reset", id="reset", n_clicks=0, className="reset"),
            ],
        ),
        html.P(id="status-line", className="status-line"),
        html.Div(
            className="grid",
            children=[
                html.Aside(
                    className="card",
                    children=[
                        html.H2("Companies"),
                        html.Div(
                            className="company-list",
                            children=[
                                html.Button(
                                    [html.Span(className="swatch", style={"background": COLORS[name]}), name],
                                    id={"type": "company", "name": name},
                                    n_clicks=0,
                                    className="company-btn",
                                )
                                for name in COMPANIES
                            ],
                        ),
                        html.Div(id="lab-detail", className="lab-detail"),
                        html.H2("Chips", className="sub"),
                        html.Div(
                            className="company-list",
                            children=[
                                html.Button(
                                    [html.Span(className=f"pattern pattern-{family.lower()}"), family],
                                    id={"type": "chip", "name": family},
                                    n_clicks=0,
                                    className="company-btn",
                                )
                                for family in CHIPS
                            ],
                        ),
                        html.Div(id="chip-detail", className="lab-detail"),
                    ],
                ),
                html.Div(
                    className="card time-card",
                    children=[
                        html.H2("Compute Capacity"),
                        dcc.Graph(id="area", config=GRAPH_CONFIG, style={"height": "330px"}),
                        html.H2("Model frontier", className="sub"),
                        dcc.Graph(id="frontier", config=GRAPH_CONFIG, style={"height": "300px"}),
                    ],
                ),
                html.Section(
                    className="card",
                    children=[
                        html.H2("Largest campuses"),
                        dcc.Graph(id="bars", config=GRAPH_CONFIG, style={"height": "660px"}),
                    ],
                ),
            ],
        ),
        html.P(
            ["Data: ", html.A("Epoch AI", href="https://epoch.ai/data/ai-companies", target="_blank")],
            className="credit",
        ),
        dcc.Store(id="selection", data=EMPTY),
    ],
)


@callback(
    Output("selection", "data"),
    Output("date-slider", "value"),
    Output("role", "value"),
    Output("status", "value"),
    Input({"type": "company", "name": ALL}, "n_clicks"),
    Input({"type": "chip", "name": ALL}, "n_clicks"),
    Input("date-slider", "value"),
    Input("role", "value"),
    Input("status", "value"),
    Input("area", "relayoutData"),
    Input("area", "clickData"),
    Input("bars", "clickData"),
    Input("frontier", "clickData"),
    Input("reset", "n_clicks"),
    State("selection", "data"),
    prevent_initial_call=True,
)
def update_selection(_company_clicks, _chip_clicks, slider, role, status, relayout, area_click, bar_click, frontier_click,
                     _reset, selection):
    current = {**EMPTY, **(selection or {})}
    triggered = ctx.triggered_id
    updated = None

    if triggered == "reset":
        updated = dict(EMPTY)
    elif isinstance(triggered, dict) and triggered.get("type") == "company":
        updated = _pick_company(current, triggered["name"])
    elif isinstance(triggered, dict) and triggered.get("type") == "chip":
        updated = _pick_chip(current, triggered["name"])
    elif triggered == "date-slider" and slider:
        start, end = _dates_from_quarters(*slider)
        updated = {**current, "date_start": start, "date_end": end}
    elif triggered == "role":
        updated = {**current, "role": role}
    elif triggered == "status":
        updated = {**current, "status": status}
    elif triggered == "area" and ctx.triggered[0]["prop_id"].endswith("relayoutData"):
        parsed = _range_from_relayout(relayout or {})
        if parsed == "reset":
            updated = {**current, "date_start": None, "date_end": None}
        elif parsed:
            updated = {**current, "date_start": parsed[0], "date_end": parsed[1]}
    elif triggered == "area" and area_click and area_click.get("points"):
        name = area_click["points"][0].get("customdata")
        if name in COMPANIES:
            updated = _pick_company(current, name)
    elif triggered == "bars" and bar_click and bar_click.get("points"):
        point = bar_click["points"][0]
        if point.get("customdata"):
            updated = _pick_campus(current, point)
    elif triggered == "frontier" and frontier_click and frontier_click.get("points"):
        data = frontier_click["points"][0].get("customdata")
        if data and data[1] in COMPANIES:
            updated = _pick_company(current, data[1])

    if updated is None or updated == current:
        return no_update, no_update, no_update, no_update
    slider_value = _slider_value(updated)
    return (
        updated,
        no_update if slider_value == slider else slider_value,
        no_update if updated["role"] == role else updated["role"],
        no_update if updated["status"] == status else updated["status"],
    )


@callback(
    Output("area", "figure"),
    Output("frontier", "figure"),
    Output("bars", "figure"),
    Output("kpis", "children"),
    Output("lab-detail", "children"),
    Output("chip-detail", "children"),
    Output("status-line", "children"),
    Output({"type": "company", "name": ALL}, "className"),
    Output({"type": "chip", "name": ALL}, "className"),
    Input("selection", "data"),
)
def render(selection):
    state = view_state(selection)
    classes = ["company-btn active" if name in state["companies"] else "company-btn" for name in COMPANIES]
    chip_classes = ["company-btn active" if family in state["chips"] else "company-btn" for family in CHIPS]
    return (
        area_figure(state),
        frontier_figure(state),
        campus_bars_figure(state),
        kpi_children(state),
        detail_children(state),
        chip_detail_children(state),
        status_text(state),
        classes,
        chip_classes,
    )


if __name__ == "__main__":
    load()
    app.run(debug=False, port=8050, host="127.0.0.1")
