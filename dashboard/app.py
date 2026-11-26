"""AI Compute Competition by 2030.

Yuehua Zhuang. Plotly Dash views of Epoch AI data centers and models.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.dash_table.Format import Format, Scheme

from dashboard.data import (
    AS_OF,
    CHIP_PATTERNS,
    CHIPS,
    COLORS,
    LABS,
    MODEL_FOCUS,
    MONTHS,
    REVENUE_COMPANIES,
    company_order,
    format_h100,
    load,
    view_state,
)

# One dict drives both cards. The company buttons toggle membership.
# A click inside a graph keeps that one company; the same company again
# clears the list. An empty list is the default and means every company.
EMPTY = {
    "companies": [],
    "role": "owner",
    "chips": [],
    "model_tier": "all",
    "model_scope": "focus",
    "raw_metric": "compute",
    "model_stat": "max",
    "model_year": None,
    "capacity_metric": "h100e",
}
YEAR_CHOICES = [str(year) for year in range(2018, 2031)]
BAR_LIMIT = 24
AREA_HEIGHT = 264
SHARE_HEIGHT = 64
COMPANIES = company_order()
FULL_RANGE = [MONTHS[0].date().isoformat(), MONTHS[-1].date().isoformat()]
TODAY = AS_OF.isoformat()

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
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#d9dbe0",
            font=dict(family=FONT, size=13, color=INK),
            grouptitlefont=dict(family=FONT, size=13, color=INK),
            align="left",
            namelength=-1,
        ),
    )
    layout.update(extra)
    return layout


def _hint(body, wide: bool = False):
    pop = "hint-pop hint-wide" if wide else "hint-pop"
    return html.Span(
        [html.Span("?", className="hint-icon"), html.Span(body, className=pop)],
        className="hint",
        tabIndex="0",
    )


def _measure_hint():
    lines = [
        ("H100e", "Chip capacity counted in equivalents of an NVIDIA H100."),
        ("IT MW", "Power drawn by the IT equipment, in megawatts."),
        ("Power", "Total facility power, in megawatts."),
        ("Buildings", "Buildings reported as operational."),
        ("8-bit", "Peak 8-bit operations per second."),
        ("Capital", "Total capital cost, in 2025 USD billions."),
        ("Compute $", "Cost of the compute hardware, in 2025 USD billions."),
        ("Build $", "Construction cost, in 2025 USD billions."),
        ("Operating $", "Annual operating cost, in 2025 USD billions."),
    ]
    body = []
    for name, sentence in lines:
        if body:
            body.append(html.Br())
        body.extend([html.B(name), " " + sentence])
    return _hint(body, wide=True)


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


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _tint(hex_color: str, amount: float) -> str:
    amount = min(1.0, max(0.0, amount))
    return _hex(_mix(_rgb(hex_color), (255, 255, 255), amount))


def _text_on(hex_color: str) -> str:
    return "#ffffff" if _luma(_rgb(hex_color)) < 0.5 else INK


def _company_color(name: str) -> str:
    return COLORS.get(name, COLORS["Other"])


def _series_color(company: str, active: bool, _future: bool) -> str:
    """Hue is the company. A company that is not selected is a lighter step of that hue."""
    color = _company_color(company)
    return color if active else _tint(color, 0.72)


def _campus_colors(segments: list[dict]) -> list[str]:
    """Largest campus keeps the company color. Smaller ones step lighter in the same hue."""
    ranks: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        ranks.setdefault(segment.get("company") or segment["label"], []).append(index)
    colors = [COLORS["Other"]] * len(segments)
    for company, indexes in ranks.items():
        ordered = sorted(indexes, key=lambda index: segments[index]["value"], reverse=True)
        count = len(ordered)
        base = _company_color(company)
        for rank, index in enumerate(ordered):
            amount = 0.0 if count == 1 else 0.42 * rank / (count - 1)
            colors[index] = _tint(base, amount)
    return colors


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _today_rule(figure: go.Figure) -> None:
    figure.add_vline(x=TODAY, line=dict(color=INK, width=1, dash="dot"), opacity=0.45)
    figure.add_annotation(x=TODAY, y=1, yref="paper", yanchor="bottom", text="Today", showarrow=False,
                          font=dict(size=10, color=MUTED), xshift=18)


CAPACITY_OPTIONS = [
    ("h100e", "H100e"),
    ("it_power_mw", "IT MW"),
    ("power_mw", "Power"),
    ("buildings", "Buildings"),
    ("performance", "8-bit"),
    ("cost_b", "Capital"),
    ("compute_cost_b", "Compute $"),
    ("construction_cost_b", "Build $"),
    ("operating_cost_b", "Operating $"),
]
MEASURE_LABEL = dict(CAPACITY_OPTIONS)
RAW_LABEL = {"compute": "FLOP", "params": "Parameters", "data": "Data"}
CAPACITY_AXIS = {
    "h100e": ("H100e", ".2s"),
    "it_power_mw": ("IT power (MW)", ".2s"),
    "power_mw": ("Power (MW)", ".2s"),
    "buildings": ("Buildings", ".0f"),
    "performance": ("8-bit OP/s", ".2s"),
    "cost_b": ("Capital cost ($B)", ".2f"),
    "compute_cost_b": ("Compute cost ($B)", ".2f"),
    "construction_cost_b": ("Construction cost ($B)", ".2f"),
    "operating_cost_b": ("Operating cost ($B)", ".2f"),
}
HOVER_VALUE = {
    "h100e": (",.0f", "H100e"),
    "it_power_mw": (",.1f", "MW"),
    "power_mw": (",.1f", "MW"),
    "buildings": (",.0f", "buildings"),
    "performance": (".3e", "8-bit OP/s"),
    "cost_b": (",.2f", "$B"),
    "compute_cost_b": (",.2f", "$B"),
    "construction_cost_b": (",.2f", "$B"),
    "operating_cost_b": (",.2f", "$B"),
}




def area_figure(state: dict) -> go.Figure:
    """Stacked capacity. Planned months add a hatch on the same hue.

    Trace opacity does not reach a stacked fill, so a company that is not
    selected is drawn as a lighter step of its own hue.
    """
    series = state["series"]
    if series.empty or float(series.to_numpy().sum()) == 0:
        return _empty_figure("No compute", AREA_HEIGHT)
    boundary = pd.Timestamp(TODAY)
    present = series.index[series.index <= boundary]
    edge = present.max() if len(present) else None
    history = series.loc[:edge] if edge is not None else series.iloc[0:0]
    future = series.loc[series.index > edge] if edge is not None else series
    title, _tick = CAPACITY_AXIS[state["capacity_metric"]]
    value_format, unit = HOVER_VALUE[state["capacity_metric"]]
    chosen = set(state["highlight"])
    ordered = [name for name in COMPANIES if name in series.columns]
    ordered += [name for name in series.columns if name not in ordered]
    figure = go.Figure()

    def add(frame: pd.DataFrame, stack: str, future: bool) -> None:
        if frame.empty:
            return
        for company in ordered:
            if company not in frame.columns or float(frame[company].sum()) == 0:
                continue
            active = not chosen or company in chosen
            color = _series_color(company, active, future)
            figure.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame[company],
                    name=company,
                    mode="lines",
                    stackgroup=stack,
                    line=dict(width=0.6, color="#ffffff"),
                    fillcolor=color,
                    opacity=1,
                    fillpattern=dict(
                        shape="/" if future else "",
                        bgcolor=color,
                        fgcolor="rgba(255,255,255,0.55)",
                        size=8,
                        solidity=0.18,
                    ),
                    customdata=[company] * len(frame),
                    hovertemplate=(
                        f"<b>%{{customdata}}</b>  %{{y:{value_format}}} {unit}<extra></extra>"
                    ),
                    legendgroup=company,
                    showlegend=False,
                    selected=dict(marker=dict(opacity=1)),
                    unselected=dict(marker=dict(opacity=1)),
                )
            )

    add(history, "history", False)
    add(future, "future", True)
    year = int(state["model_year"])
    start_year = min(year, 2024)
    band_end = "2030-02-01" if year >= 2030 else f"{year + 1}-01-01"
    figure.add_vrect(
        x0=f"{year}-01-01",
        x1=band_end,
        fillcolor="rgba(17,17,19,0.06)",
        line_width=0,
        layer="below",
    )
    _today_rule(figure)
    _year_labels(figure, start_year, year)
    figure.update_layout(
        **_base_layout(
            AREA_HEIGHT,
            hovermode="x unified",
            dragmode="select",
            selectdirection="h",
            clickmode="event+select",
            margin=dict(l=52, r=16, t=12, b=36),
        ),
        xaxis=_axis(
            "",
            range=[f"{start_year}-01-01", FULL_RANGE[1]],
            fixedrange=True,
            showgrid=False,
            showticklabels=False,
            ticks="",
            hoverformat="%b %Y",
        ),
        yaxis=_axis(title, tickformat=_tick, fixedrange=True, rangemode="tozero"),
    )
    return figure


def _year_labels(figure: go.Figure, start_year: int, selected: int) -> None:
    """Bold year labels, centered in each year so the selected one sits in the band."""
    years = list(range(int(start_year), 2030))
    if int(selected) not in years:
        years.append(int(selected))
    for label in years:
        chosen = label == int(selected)
        at_end = label >= 2030
        figure.add_annotation(
            x=f"{label}-01-01" if at_end else f"{label}-07-01",
            y=0,
            yref="paper",
            xanchor="right" if at_end else "center",
            yanchor="top",
            yshift=-2,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=12, color=INK if chosen else MUTED, family=FONT),
            bgcolor="rgba(17,17,19,0.08)" if chosen else "rgba(0,0,0,0)",
            borderpad=3,
        )


def share_bar(state: dict) -> go.Figure:
    segments = state["segments"]
    if not segments:
        return _empty_figure("No capacity", 140)
    _title, tick = CAPACITY_AXIS[state["capacity_metric"]]
    value_format, unit = HOVER_VALUE[state["capacity_metric"]]
    kind = state["segment_kind"]
    year = state["model_year"]
    heading = f"{'Campuses' if kind == 'campus' else 'Companies'} · {year}"
    total = sum(segment["value"] for segment in segments)
    colors = (
        [_company_color(segment["company"]) for segment in segments]
        if kind == "company"
        else _campus_colors(segments)
    )
    figure = go.Figure()
    for segment, color in zip(segments, colors):
        custom = segment["company"] if segment.get("company") in COMPANIES else ""
        figure.add_trace(
            go.Bar(
                y=[heading],
                x=[segment["value"]],
                orientation="h",
                name=segment["label"],
                marker=dict(color=color, line=dict(color="#ffffff", width=1)),
                text=segment["label"] if total and segment["value"] / total >= 0.08 else "",
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=_text_on(color), size=11),
                customdata=[custom],
                hovertemplate=(
                    f"<b>{segment['label']}</b><br>%{{x:{value_format}}} {unit}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.update_layout(
        **_base_layout(
            SHARE_HEIGHT,
            barmode="stack",
            margin=dict(l=8, r=8, t=0, b=20),
            hovermode="closest",
        ),
        xaxis=_axis("", tickformat=tick, fixedrange=True, rangemode="tozero"),
        yaxis=_axis("", fixedrange=True, showgrid=False, automargin=True),
    )
    return figure


RAW_AXIS = {
    "compute": "10ⁿ FLOP",
    "params": "10ⁿ",
    "data": "10ⁿ",
}


def _color_org(orgs: list[str]) -> str:
    for name in orgs:
        if name in MODEL_FOCUS:
            return name
    return "Other"


def _tick_label(name: str) -> str:
    text = " ".join(str(name).split())
    return text if len(text) <= 28 else text[:27] + "…"


def _pow10(value) -> str:
    if value is None or pd.isna(value) or value <= 0:
        return "—"
    exponent = math.log10(float(value))
    power = math.floor(exponent)
    mantissa = 10 ** (exponent - power)
    if mantissa >= 9.95:
        mantissa /= 10
        power += 1
    superscript = str(power).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mantissa:.1f}×10{superscript}"


def _log10_summary(summary: dict) -> dict:
    rows = []
    for row in summary.get("rows") or []:
        converted = {"company": row["company"]}
        for year in summary.get("years") or []:
            value = row.get(year)
            converted[year] = None if value is None or value <= 0 else math.log10(float(value))
        rows.append(converted)
    return {"years": summary.get("years") or [], "rows": rows}


def _padded_range(values: pd.Series, scale: str) -> list[float]:
    """Start beside the data. Bar charts otherwise pull the axis back to zero."""
    finite = pd.to_numeric(values, errors="coerce").dropna()
    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    pad = span * 0.14 if span else max(abs(high) * 0.05, 0.04 if scale == "unit" else 0.5)
    start = low - pad
    if low > 0 and start <= 0:
        start = low * 0.92
    return [start, high + pad * 0.4]


def _year_bars(frame: pd.DataFrame, column: str, axis_title: str, scale: str, hover) -> go.Figure:
    if frame.empty:
        return _empty_figure("No models", 120)
    shown = frame.sort_values(column, ascending=False).head(BAR_LIMIT).copy()
    shown["color_org"] = shown["orgs"].map(_color_org)
    shown["label_org"] = shown.apply(
        lambda row: row["color_org"] if row["color_org"] in MODEL_FOCUS else row["point_org"],
        axis=1,
    )
    ticks = shown["model"].map(_tick_label)
    if ticks.duplicated().any():
        ticks = shown["model"].map(_tick_label) + " · " + shown["label_org"]
    shown["tick"] = ticks
    shown = shown.iloc[::-1]
    height = max(108, 20 + len(shown) * 12)
    figure = go.Figure(
        go.Bar(
            x=shown[column],
            y=shown["tick"],
            orientation="h",
            marker=dict(color=[COLORS.get(name, COLORS["Other"]) for name in shown["color_org"]]),
            customdata=list(zip(
                shown["model"],
                shown["color_org"].where(shown["color_org"].isin(COMPANIES), ""),
                shown["label_org"],
                [hover(row) for row in shown.to_dict(orient="records")],
            )),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>",
        )
    )
    xaxis = _axis(axis_title, fixedrange=True, range=_padded_range(shown[column], scale))
    if scale == "log":
        xaxis.update(type="log", exponentformat="power")
    elif scale == "unit":
        xaxis.update(tickformat=".1f")
    else:
        xaxis.update(tickformat=".0f")
        if float(shown[column].max()) - float(shown[column].min()) > 4:
            xaxis["dtick"] = 2
    figure.update_layout(
        **_base_layout(height, margin=dict(l=4, r=8, t=2, b=22), bargap=0.22),
        xaxis=xaxis,
        yaxis=_axis("", fixedrange=True, showgrid=False, automargin=True, tickfont=dict(size=10, color=MUTED)),
    )
    return figure


def score_figure(state: dict) -> go.Figure:
    return _year_bars(
        state["ranked_year"],
        "score",
        "Normalized rank",
        "unit",
        lambda row: f"{row['score']:.2f} · {row['used']}",
    )


def raw_figure(state: dict) -> go.Figure:
    frame = state["raw_year"].copy()
    if not frame.empty:
        frame["value"] = frame["value"].map(math.log10)
    return _year_bars(
        frame,
        "value",
        RAW_AXIS[state["raw_metric"]],
        "exponent",
        lambda row: _pow10(10 ** row["value"]),
    )


MAX_COMPARE = 3


def _latest_staff(staff: pd.DataFrame, company: str):
    rows = staff.loc[staff["company"] == company]
    return None if rows.empty else float(rows.iloc[-1]["staff"])


SUMMARY_COLUMNS = [
    ("Campuses", "campuses", lambda value: f"{value}"),
    ("H100e", "h100e", format_h100),
    ("MW", "power_mw", lambda value: f"{value:,.0f}"),
    ("$B", "cost_b", lambda value: f"{value:.2f}"),
]


def _compare_table(rows: list[dict], labels: list, columns: list):
    leaders = {}
    if len(rows) > 1:
        for _, key, _ in columns:
            values = [row[key] for row in rows if row.get(key) is not None]
            leaders[key] = max(values) if values else None

    def cell(row, key, fmt):
        value = row.get(key)
        lead = leaders.get(key) is not None and value is not None and value == leaders[key]
        shown = "—" if value is None else fmt(value)
        return html.Td(shown, className="lead" if lead else None)

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


SCORE_FORMAT = Format(precision=2, scheme=Scheme.fixed)
EXPONENT_FORMAT = Format(precision=1, scheme=Scheme.fixed)
GRADIENT_STEPS = 8


def _mix(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(start[i] + (end[i] - start[i]) * t) for i in range(3))


def _luma(rgb: tuple[int, int, int]) -> float:
    def channel(value: int) -> float:
        x = value / 255
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(part) for part in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _gradient_styles(rows: list[dict], years: list[str], logarithmic: bool) -> list[dict]:
    """Shade each company in its own color. Empty cells stay blank.

    Companies without their own hue share one ramp. A rule per company per
    shade freezes the table once every organization is included.
    """
    values = []
    for row in rows:
        for year in years:
            value = row.get(year)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            number = float(value)
            if logarithmic and number <= 0:
                continue
            values.append(number)
    if not values:
        return []
    scaled = [math.log10(value) for value in values] if logarithmic else list(values)
    low, high = min(scaled), max(scaled)
    if high == low:
        edges = [(None, None, 1.0)]
    else:
        edges = []
        for step in range(GRADIENT_STEPS):
            start = None if step == 0 else low + (high - low) * step / GRADIENT_STEPS
            end = None if step == GRADIENT_STEPS - 1 else low + (high - low) * (step + 1) / GRADIENT_STEPS
            edges.append((start, end, (step + 0.5) / GRADIENT_STEPS))

    def ramp(color: str, company: str | None) -> list[dict]:
        rgb_color = _rgb(color)
        styles = []
        for year in years:
            for start, end, shade in edges:
                rgb = _mix((255, 255, 255), rgb_color, 0.16 + 0.84 * shade)
                if logarithmic:
                    start_value = None if start is None else 10 ** start
                    end_value = None if end is None else 10 ** end
                else:
                    start_value, end_value = start, end
                if start_value is None and end_value is None:
                    clause = f"{{{year}}} >= {min(values):.8g}"
                elif start_value is None:
                    clause = f"{{{year}}} < {end_value:.8g}"
                elif end_value is None:
                    clause = f"{{{year}}} >= {start_value:.8g}"
                else:
                    clause = f"{{{year}}} >= {start_value:.8g} && {{{year}}} < {end_value:.8g}"
                query = f"{{{year}}} is num && {clause}"
                if company is not None:
                    query = f'{{company}} = "{company}" && {query}'
                styles.append(
                    {
                        "if": {"column_id": year, "filter_query": query},
                        "backgroundColor": _hex(rgb),
                        "color": "#ffffff" if _luma(rgb) < 0.30 else INK,
                    }
                )
        return styles

    present = []
    for row in rows:
        name = row.get("company")
        if name and name not in present:
            present.append(name)
    styles = ramp(COLORS["Other"], None)
    for name in present:
        color = COLORS.get(name)
        if color and color != COLORS["Other"]:
            styles.extend(ramp(color, name))
    return styles


def _year_table(summary: dict, number_format, sort_latest: bool, logarithmic: bool = False, selected_year=None):
    rows = summary.get("rows") or []
    years = summary.get("years") or []
    if not rows:
        return html.P("No companies in this window.", className="model-note")
    columns = [{"name": "Company", "id": "company"}]
    columns += [
        {"name": year, "id": year, "type": "numeric", "format": number_format}
        for year in years
    ]
    accents = []
    for name, color in COLORS.items():
        if name == "Other":
            continue
        accents.append(
            {
                "if": {"column_id": "company", "filter_query": f'{{company}} = "{name}"'},
                "borderLeft": f"3px solid {color}",
            }
        )
    sort_by = [{"column_id": years[-1], "direction": "desc"}] if sort_latest and years else []
    return dash_table.DataTable(
        columns=columns,
        data=rows,
        sort_action="native",
        sort_mode="single",
        sort_by=sort_by,
        page_action="native" if len(rows) > 12 else "none",
        page_size=12,
        fill_width=True,
        style_as_list_view=True,
        style_table={"overflowX": "auto", "width": "100%", "maxWidth": "100%"},
        style_cell={
            "fontFamily": FONT,
            "fontSize": "11px",
            "padding": "1px 4px",
            "height": "20px",
            "lineHeight": "16px",
            "textAlign": "right",
            "border": "none",
            "backgroundColor": "transparent",
            "minWidth": "36px",
            "width": "40px",
            "maxWidth": "44px",
        },
        style_cell_conditional=[
            {
                "if": {"column_id": "company"},
                "textAlign": "left",
                "minWidth": "78px",
                "width": "78px",
                "maxWidth": "78px",
                "paddingLeft": "10px",
                "backgroundColor": "#ffffff",
            }
        ],
        style_header={
            "fontFamily": FONT,
            "fontWeight": "500",
            "fontSize": "11px",
            "height": "22px",
            "lineHeight": "16px",
            "padding": "2px 4px",
            "color": MUTED,
            "backgroundColor": "transparent",
            "border": "none",
            "borderBottom": "1px solid #e7e8eb",
            "textAlign": "right",
        },
        style_header_conditional=[
            {"if": {"column_id": "company"}, "textAlign": "left", "backgroundColor": "#ffffff"},
            *(
                [{
                    "if": {"column_id": str(selected_year)},
                    "color": INK,
                    "fontWeight": "600",
                    "backgroundColor": "#ffffff",
                    "borderBottom": "2px solid #111113",
                }]
                if selected_year is not None
                else []
            ),
        ],
        style_data={"border": "none", "borderBottom": "1px solid #f3f4f6", "color": INK},
        style_data_conditional=accents + _gradient_styles(rows, years, logarithmic),
        css=[
            {"selector": ".dash-spreadsheet", "rule": "font-variant-numeric: tabular-nums;"},
            {"selector": ".previous-next-container", "rule": "font-size: 12px;"},
        ],
    )


def chip_detail_children(state: dict):
    rows = state["chip_summary"]
    if not rows or len(rows) > MAX_COMPARE:
        return []
    columns = [column for column in SUMMARY_COLUMNS if column[1] != "power_mw"]
    return [_compare_table(rows, [_chip_label(row["chip"]) for row in rows], columns)]


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
    items = [
        (f"{len(state['campuses'])}", "Campuses"),
        (format_h100(state["h100e_now"]), "H100 equivalent"),
        (f"{len(state['ranked'])}", "Ranked models"),
    ]
    return [
        html.Div([html.Div(value, className="value"), html.Div(label, className="label")], className="kpi")
        for value, label in items
    ]


def _company_classes(state: dict) -> list[str]:
    picks = set(state["companies"])
    return ["company-btn active" if name in picks else "company-btn" for name in COMPANIES]


def _select_all_class(state: dict) -> str:
    if not state["companies"]:
        return "select-all active"
    return "select-all"


def _year_classes(state: dict) -> list[str]:
    selected = str(state["model_year"])
    return ["year-btn active" if year == selected else "year-btn" for year in YEAR_CHOICES]


def _pick_company(selection: dict, name: str) -> dict:
    """Toggle one company from the left rail. Does not change the scope."""
    chosen = list(selection.get("companies") or [])
    if name in chosen:
        chosen.remove(name)
        role = selection.get("role")
    else:
        chosen.append(name)
        role = "user" if name in LABS and selection.get("role") == "owner" else selection.get("role")
    chosen = [company for company in COMPANIES if company in chosen]
    return {**selection, "companies": chosen, "role": role}


def _point_company(value, index: int = 0):
    if isinstance(value, (list, tuple)):
        value = value[index] if len(value) > index else None
    return value if value in COMPANIES else None


def _set_year(selection: dict, year) -> dict | None:
    try:
        chosen = int(str(year)[:4])
    except (TypeError, ValueError):
        return None
    if not 2018 <= chosen <= 2030 or chosen == int(selection.get("model_year") or 0):
        return None
    return {**selection, "model_year": chosen}


def _year_from_brush(selected) -> int | None:
    """Snap a horizontal drag to the year it covers most. A short drag is a click."""
    span = (selected or {}).get("range") or {}
    bounds = span.get("x") if isinstance(span, dict) else None
    if not bounds or len(bounds) < 2:
        return None
    start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
    if end < start:
        start, end = end, start
    if end - start < pd.Timedelta(days=21):
        return None
    best_year, best_overlap = None, pd.Timedelta(0)
    for year in range(2018, 2031):
        opening = pd.Timestamp(f"{year}-01-01")
        closing = pd.Timestamp("2030-02-01") if year >= 2030 else pd.Timestamp(f"{year + 1}-01-01")
        overlap = min(end, closing) - max(start, opening)
        if overlap > best_overlap:
            best_year, best_overlap = year, overlap
    return best_year


def _solo_company(selection: dict, name: str) -> dict:
    """Graph click keeps one company. The same company again clears the list."""
    chosen = [company for company in COMPANIES if company in (selection.get("companies") or [])]
    if chosen == [name]:
        return {**selection, "companies": []}
    role = selection.get("role")
    if name in LABS and role == "owner":
        role = "user"
    return {**selection, "companies": [name], "role": role}


def _choose_all(selection: dict) -> dict:
    """Select None. An empty list shows every company in the current scope."""
    return {**selection, "companies": [], "model_scope": selection.get("model_scope", "focus")}


def _pick_chip(selection: dict, family: str) -> dict:
    chosen = list(selection.get("chips") or [])
    chosen = [item for item in chosen if item != family] if family in chosen else chosen + [family]
    return {**selection, "chips": [item for item in CHIPS if item in chosen]}


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
                            "Who builds and who uses AI compute. The left rail sets the companies and year; capacity is above, models below.",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    className="kpi-block",
                    children=[
                        html.Div("Now", className="kpi-kicker"),
                        html.Div(id="kpis", className="kpis"),
                    ],
                ),
            ],
        ),
        html.Div(
            className="workspace",
            children=[
                html.Aside(
                    className="card rail",
                    children=[
                        html.H2("Filters & Controls", className="region-title"),
                        html.H3("Companies", className="group-label"),
                        dcc.RadioItems(
                            id="model-scope",
                            className="segmented scope-switch",
                            options=[
                                {"label": "Focus Group", "value": "focus"},
                                {"label": "All Companies", "value": "all"},
                            ],
                            value="focus",
                            inline=True,
                        ),
                        html.Div(
                            className="company-list",
                            children=[
                                html.Button(
                                    [html.Span(className="swatch swatch-empty"), "Select None"],
                                    id="select-all",
                                    n_clicks=0,
                                    className="select-all",
                                ),
                                *[
                                    html.Button(
                                        [html.Span(className="swatch", style={"background": COLORS[name]}), name],
                                        id={"type": "company", "name": name},
                                        n_clicks=0,
                                        className="company-btn",
                                    )
                                    for name in COMPANIES
                                ],
                            ],
                        ),
                        html.H3("Year", className="group-label"),
                        html.Div(
                            className="year-ruler",
                            children=[
                                html.Button(
                                    year,
                                    id={"type": "model-year", "year": year},
                                    n_clicks=0,
                                    className="year-btn",
                                )
                                for year in YEAR_CHOICES
                            ],
                        ),
                        html.Div(
                            className="rail-foot",
                            children=html.Button("Reset", id="reset", n_clicks=0, className="reset"),
                        ),
                    ],
                ),
                html.Div(
                    className="stack",
                    children=[
                        html.Section(
                            className="card campus-panel",
                            children=[
                                html.H2("Company & Campus", className="region-title"),
                                html.Div(
                                    className="filter-stack",
                                    children=[
                                        html.Div(
                                            className="filter-row",
                                            children=[
                                                html.Span(
                                                    [
                                                        "Count as",
                                                        html.Span("?", className="hint-icon"),
                                                        html.Span(
                                                            [
                                                                html.B("Owner"),
                                                                " counts a campus toward the company that owns it. ",
                                                                html.B("User"),
                                                                " splits a campus evenly across the companies that train there.",
                                                                html.Br(),
                                                                html.Br(),
                                                                "OpenAI owns none, so Owner shows 0. As a user it gets "
                                                                "12 campuses and 2.0M H100e.",
                                                            ],
                                                            className="hint-pop",
                                                        ),
                                                    ],
                                                    className="filter-label criteria-label",
                                                    tabIndex="0",
                                                ),
                                                dcc.RadioItems(
                                                    id="role",
                                                    className="segmented",
                                                    options=[
                                                        {"label": "Owner", "value": "owner"},
                                                        {"label": "User", "value": "user"},
                                                    ],
                                                    value="owner",
                                                    inline=True,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="filter-row",
                                            children=[
                                                html.Span("Chip", className="filter-label"),
                                                html.Div(
                                                    className="chip-row",
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
                                            ],
                                        ),
                                        html.Div(
                                            className="filter-row",
                                            children=[
                                                html.Span(
                                                    ["Measure", _measure_hint()],
                                                    className="filter-label",
                                                ),
                                                dcc.RadioItems(
                                                    id="capacity-metric",
                                                    className="segmented metric-switch",
                                                    options=[
                                                        {"label": label, "value": value}
                                                        for value, label in CAPACITY_OPTIONS
                                                    ],
                                                    value="h100e",
                                                    inline=True,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(id="lab-detail", className="lab-detail"),
                                html.Div(id="chip-detail", className="lab-detail"),
                                html.P(
                                    "H100e Time Series of Focus Group Companies (Owner)",
                                    id="area-title",
                                    className="chart-title",
                                ),
                                dcc.Graph(id="area", config=GRAPH_CONFIG, style={"height": f"{AREA_HEIGHT}px"}),
                                html.P(
                                    "Companies' H100e Proportion as of 2026",
                                    id="bar-title",
                                    className="chart-title",
                                ),
                                dcc.Graph(id="share-bar", config=GRAPH_CONFIG, style={"height": f"{SHARE_HEIGHT}px"}),
                            ],
                        ),
        html.Section(
            className="model-board",
            children=[
                html.H2("Models", className="region-title"),
                html.Div(
                    className="model-grid",
                    children=[
                        html.Div(
                            className="model-col",
                            children=[
                                html.Div(
                                    className="col-head",
                                    children=[
                                        html.H3(
                                            className="rank-title",
                                            children=[
                                                html.Span(
                                                    "Normalized Rank of All Models as of 2026",
                                                    id="score-title",
                                                ),
                                                _hint(
                                                    "Percentile of training compute, parameters, and training data "
                                                    "within this tier, averaged over the metrics a model reports."
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="filter-row",
                                            children=[
                                                html.Span(
                                                    [
                                                        "Tier",
                                                        _hint(
                                                            [
                                                                html.B("Notable"),
                                                                " models are the ones Epoch AI lists notability criteria for. ",
                                                                html.B("Frontier"),
                                                                " models are the ones Epoch AI marks as frontier.",
                                                            ]
                                                        ),
                                                    ],
                                                    className="filter-label",
                                                ),
                                                dcc.RadioItems(
                                                    id="model-tier",
                                                    className="segmented",
                                                    options=[
                                                        {"label": "All", "value": "all"},
                                                        {"label": "Notable", "value": "notable"},
                                                        {"label": "Frontier", "value": "frontier"},
                                                    ],
                                                    value="all",
                                                    inline=True,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(className="model-plot", children=dcc.Graph(id="score", config=GRAPH_CONFIG)),
                                html.P(
                                    "Compare Max Normalized Rank among companies",
                                    id="score-table-title",
                                    className="table-caption",
                                ),
                                html.Div(id="score-table", className="year-table"),
                            ],
                        ),
                        html.Div(
                            className="model-col",
                            children=[
                                html.Div(
                                    className="col-head",
                                    children=[
                                        html.H3(
                                            className="rank-title",
                                            children=[
                                                html.Span(
                                                    "FLOP of Frontier Models as of 2026",
                                                    id="raw-title",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="filter-row",
                                            children=[
                                                html.Span("Size", className="filter-label"),
                                                dcc.RadioItems(
                                                    id="raw-metric",
                                                    className="segmented",
                                                    options=[
                                                        {"label": "10ⁿ FLOP", "value": "compute"},
                                                        {"label": "Parameters", "value": "params"},
                                                        {"label": "Data", "value": "data"},
                                                    ],
                                                    value="compute",
                                                    inline=True,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(className="model-plot", children=dcc.Graph(id="raw", config=GRAPH_CONFIG)),
                                html.P(
                                    "Compare Max FLOP of Frontier Models among companies",
                                    id="raw-table-title",
                                    className="table-caption",
                                ),
                                html.Div(id="raw-table", className="year-table"),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="table-stat",
                    children=[
                        html.Span("In the tables", className="filter-label"),
                        dcc.RadioItems(
                            id="model-stat",
                            className="segmented",
                            options=[
                                {"label": "Max", "value": "max"},
                                {"label": "Median", "value": "median"},
                            ],
                            value="max",
                            inline=True,
                        ),
                    ],
                ),
            ],
        ),
                    ],
                ),
            ],
        ),
        html.P(
            [
                "Yuehua Zhuang. Data: Epoch AI ",
                html.A("AI data centers", href="https://epoch.ai/data/ai-data-centers", target="_blank"),
                " and ",
                html.A("AI models", href="https://epoch.ai/data/ai-models", target="_blank"),
                ", snapshot 23 Nov 2026. Rank scores use training compute, parameters, and training data.",
            ],
            className="credit",
        ),
        dcc.Store(id="selection", data={**EMPTY}),
        dcc.Store(id="year-pick"),
        dcc.Store(id="shown-year"),
        html.Div(id="year-anchor", style={"display": "none"}),
    ],
)


@callback(
    Output("selection", "data"),
    Output("role", "value"),
    Output("model-tier", "value"),
    Output("model-scope", "value"),
    Output("raw-metric", "value"),
    Output("model-stat", "value"),
    Output("capacity-metric", "value"),
    Output("area", "clickData"),
    Output("share-bar", "clickData"),
    Output("score", "clickData"),
    Output("raw", "clickData"),
    Output("area", "selectedData"),
    Input({"type": "company", "name": ALL}, "n_clicks"),
    Input({"type": "chip", "name": ALL}, "n_clicks"),
    Input({"type": "model-year", "year": ALL}, "n_clicks"),
    Input("role", "value"),
    Input("model-tier", "value"),
    Input("model-scope", "value"),
    Input("raw-metric", "value"),
    Input("model-stat", "value"),
    Input("capacity-metric", "value"),
    Input("area", "clickData"),
    Input("area", "selectedData"),
    Input("share-bar", "clickData"),
    Input("score", "clickData"),
    Input("raw", "clickData"),
    Input("select-all", "n_clicks"),
    Input("reset", "n_clicks"),
    Input("year-pick", "data"),
    State("selection", "data"),
    prevent_initial_call=True,
)
def update_selection(_company_clicks, _chip_clicks, _model_years, role, tier, scope, metric, stat,
                     capacity_metric, area_click, area_select, bar_click, score_click, raw_click,
                     _select_all, _reset, year_pick, selection):
    """Write the selection dict. ``area`` fires both click and brush, so the prop id separates them."""
    current = {**EMPTY, **(selection or {})}
    triggered = ctx.triggered_id
    fired = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    updated = None
    graph = None
    clear_brush = False

    if triggered == "reset":
        updated = {**EMPTY}
    elif triggered == "select-all":
        updated = _choose_all(current)
    elif isinstance(triggered, dict) and triggered.get("type") == "company":
        updated = _pick_company(current, triggered["name"])
    elif isinstance(triggered, dict) and triggered.get("type") == "chip":
        updated = _pick_chip(current, triggered["name"])
    elif isinstance(triggered, dict) and triggered.get("type") == "model-year":
        updated = {**current, "model_year": int(triggered["year"])}
    elif triggered == "year-pick":
        updated = _set_year(current, year_pick)
    elif triggered == "role":
        updated = {**current, "role": role}
    elif triggered == "model-tier":
        updated = {**current, "model_tier": tier}
    elif triggered == "model-scope":
        updated = {**current, "model_scope": scope}
    elif triggered == "raw-metric":
        updated = {**current, "raw_metric": metric}
    elif triggered == "model-stat":
        updated = {**current, "model_stat": stat}
    elif triggered == "capacity-metric":
        updated = {**current, "capacity_metric": capacity_metric}
    elif fired == "area.clickData" and area_click and area_click.get("points"):
        graph = "area"
        name = _point_company(area_click["points"][0].get("customdata"))
        if name:
            updated = _solo_company(current, name)
    elif fired == "area.selectedData":
        clear_brush = True
        updated = _set_year(current, _year_from_brush(area_select))
    elif triggered == "share-bar" and bar_click and bar_click.get("points"):
        graph = "share-bar"
        name = _point_company(bar_click["points"][0].get("customdata"))
        if name:
            updated = _solo_company(current, name)
    elif triggered in ("score", "raw"):
        click = score_click if triggered == "score" else raw_click
        if click and click.get("points"):
            graph = triggered
            name = _point_company(click["points"][0].get("customdata"), index=1)
            if name:
                updated = _solo_company(current, name)

    same = updated is None or updated == current
    if same and graph is None and not clear_brush:
        return (no_update,) * 12
    selection_out = no_update if same else updated
    role_out = no_update if same or updated["role"] == role else updated["role"]
    tier_out = no_update if same or updated.get("model_tier", "all") == tier else updated.get("model_tier", "all")
    scope_out = no_update if same or updated.get("model_scope", "focus") == scope else updated.get("model_scope", "focus")
    metric_out = no_update if same or updated.get("raw_metric", "compute") == metric else updated.get("raw_metric", "compute")
    stat_out = no_update if same or updated.get("model_stat", "max") == stat else updated.get("model_stat", "max")
    capacity_out = (
        no_update
        if same or updated.get("capacity_metric", "h100e") == capacity_metric
        else updated.get("capacity_metric", "h100e")
    )
    # Clearing clickData lets the same company be clicked again.
    cleared = tuple(None if name == graph else no_update for name in ("area", "share-bar", "score", "raw"))
    brush_out = None if clear_brush else no_update
    return (
        selection_out, role_out, tier_out, scope_out, metric_out, stat_out, capacity_out,
        *cleared, brush_out,
    )


@callback(
    Output("area", "figure"),
    Output("score", "figure"),
    Output("raw", "figure"),
    Output("kpis", "children"),
    Output("lab-detail", "children"),
    Output("chip-detail", "children"),
    Output("score-title", "children"),
    Output("score-table-title", "children"),
    Output("area-title", "children"),
    Output("bar-title", "children"),
    Output("raw-title", "children"),
    Output("raw-table-title", "children"),
    Output("score-table", "children"),
    Output("raw-table", "children"),
    Output("share-bar", "figure"),
    Output("select-all", "className"),
    Output({"type": "company", "name": ALL}, "className"),
    Output({"type": "chip", "name": ALL}, "className"),
    Output({"type": "model-year", "year": ALL}, "className"),
    Output("shown-year", "data"),
    Input("selection", "data"),
)
def render(selection):
    state = view_state(selection)
    chip_classes = ["company-btn active" if family in state["chips"] else "company-btn" for family in CHIPS]
    year = state["model_year"]
    tier = {"all": "All", "notable": "Notable", "frontier": "Frontier"}.get(state["model_tier"], "All")
    stat = {"max": "Max", "median": "Median"}.get(state["model_stat"], "Max")
    measure = MEASURE_LABEL.get(state["capacity_metric"], "H100e")
    role = "Owner" if state["role"] == "owner" else "User"
    if state["scope"] == "focus":
        group = "Focus Group Companies"
    elif not state["companies"] or set(state["companies"]) == set(COMPANIES):
        group = "All Companies"
    elif len(state["companies"]) == 1:
        group = state["companies"][0]
    else:
        group = ", ".join(state["companies"])
    share_noun = "Campuses'" if state["segment_kind"] == "campus" else "Companies'"
    size = RAW_LABEL.get(state["raw_metric"], "FLOP")
    return (
        area_figure(state),
        score_figure(state),
        raw_figure(state),
        kpi_children(state),
        detail_children(state),
        chip_detail_children(state),
        f"Normalized Rank of {tier} Models as of {year}",
        f"Compare {stat} Normalized Rank among companies",
        f"{measure} Time Series of {group} ({role})",
        f"{share_noun} {measure} Proportion as of {year}",
        f"{size} of Frontier Models as of {year}",
        f"Compare {stat} {size} of Frontier Models among companies",
        _year_table(state["rank_summary"], SCORE_FORMAT, state["org_names"] is None, selected_year=year),
        _year_table(
            _log10_summary(state["raw_summary"]),
            EXPONENT_FORMAT,
            state["org_names"] is None,
            selected_year=year,
        ),
        share_bar(state),
        _select_all_class(state),
        _company_classes(state),
        chip_classes,
        _year_classes(state),
        year,
    )


# Scroll each model table so the selected year column is in view.
clientside_callback(
    """
    function(year) {
        const scroll = () => {
            if (year === undefined || year === null || year === "") return;
            document.querySelectorAll('.year-table [data-dash-column="' + year + '"]').forEach((cell) => {
                const scroller = cell.closest('.dash-spreadsheet-container');
                if (!scroller) return;
                const edge = cell.offsetLeft + cell.offsetWidth;
                scroller.scrollLeft = Math.max(0, edge - scroller.clientWidth);
            });
        };
        requestAnimationFrame(scroll);
        setTimeout(scroll, 60);
        return "";
    }
    """,
    Output("year-anchor", "children"),
    Input("shown-year", "data"),
)


if __name__ == "__main__":
    load()
    app.run(debug=False, port=8060, host="127.0.0.1")
