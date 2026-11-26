"""Load Epoch AI tables for the dashboard.

Yuehua Zhuang. Raw CSVs stay untouched. Speculative campuses are dropped,
organization names are aligned, and chip types are grouped into a few families.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "epoch"
AS_OF = date(2026, 11, 23)

FOCUS_OWNERS = [
    "Google",
    "Meta",
    "Microsoft",
    "Amazon",
    "Oracle",
    "CoreWeave",
    "xAI",
]
LABS = ["OpenAI", "Anthropic"]
REVENUE_COMPANIES = ["OpenAI", "Anthropic", "xAI"]

# One hue per company, from the Okabe–Ito and Tol categorical palettes.
COLORS = {
    "Google": "#E69F00",
    "Meta": "#0072B2",
    "Microsoft": "#56B4E9",
    "Amazon": "#009E73",
    "Oracle": "#CC79A7",
    "CoreWeave": "#D55E00",
    "xAI": "#332288",
    "OpenAI": "#44AA99",
    "Anthropic": "#882255",
    "Other": "#B9BDC4",
    "Unknown": "#DCDFE3",
}

CHIP_PATTERNS = {
    "NVIDIA": "",
    "TPU": "/",
    "Trainium": "x",
    "Other": ".",
    "Unspecified": "-",
}
CHIPS = list(CHIP_PATTERNS)

MONTHS = pd.date_range("2018-01-01", "2030-01-01", freq="MS")

_ALIASES = {
    "google": "Google",
    "google deepmind": "Google",
    "deepmind": "Google",
    "google brain": "Google",
    "meta": "Meta",
    "meta ai": "Meta",
    "fair": "Meta",
    "microsoft": "Microsoft",
    "microsoft research": "Microsoft",
    "amazon": "Amazon",
    "aws": "Amazon",
    "oracle": "Oracle",
    "coreweave": "CoreWeave",
    "xai": "xAI",
    "x.ai": "xAI",
    "spacexai": "xAI",
    "spacex ai": "xAI",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}


def canonicalize(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return ""
    return _ALIASES.get(text.lower(), text)


def _split_parties(value: str) -> list[tuple[str, str]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parties = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "#" in chunk:
            name, confidence = chunk.rsplit("#", 1)
            parties.append((canonicalize(name.strip()), confidence.strip().lower()))
        else:
            parties.append((canonicalize(chunk), "confident"))
    return [(name, confidence) for name, confidence in parties if name]


def chip_family(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)) or not str(text).strip():
        return "Unspecified"
    first = str(text).split(",")[0].strip().lower()
    if "tpu" in first:
        return "TPU"
    if "trainium" in first:
        return "Trainium"
    nvidia = ("h100", "h200", "a100", "b200", "b300", "gb200", "gb300", "h20", "nvidia", "v100")
    if any(token in first for token in nvidia):
        return "NVIDIA"
    return "Other"


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _prepare_campuses(timeline: pd.DataFrame) -> pd.DataFrame:
    """Drop a campus whose listed owner is marked speculative."""
    raw = pd.read_csv(DATA_DIR / "data_centers.csv")
    rows = []
    for record in raw.to_dict(orient="records"):
        parties = _split_parties(record.get("Owner"))
        if not parties:
            owner, confidence = "", "confident"
        else:
            owner, confidence = parties[0]
        if confidence == "speculative":
            continue
        users = [
            name
            for name, user_confidence in _split_parties(record.get("Users"))
            if user_confidence != "speculative"
        ]
        rows.append(
            {
                "name": record["Name"],
                "h100e": record["Current H100 equivalents"],
                "snapshot_power_mw": record["Current power (MW)"],
                "cost_b": record["Current total capital cost (2025 USD billions)"],
                "owner": owner,
                "users": users,
                "chip_family": chip_family(record.get("Current chip types")),
                "chip_raw": "" if pd.isna(record.get("Current chip types")) else str(record.get("Current chip types")).strip(),
                "country": record.get("Country") or "",
            }
        )
    campuses = pd.DataFrame(rows)
    campuses["h100e"] = _to_number(campuses["h100e"])
    campuses["snapshot_power_mw"] = _to_number(campuses["snapshot_power_mw"])
    campuses["cost_b"] = _to_number(campuses["cost_b"]).fillna(0)
    campuses["owner_label"] = campuses["owner"].where(campuses["owner"].astype(str).str.len() > 0, "Unknown")
    campuses["users_label"] = campuses["users"].map(lambda users: ", ".join(users) if users else "Unknown")
    campuses["chips_label"] = campuses["chip_raw"].map(_chip_label)

    history = timeline.loc[timeline["date"] <= pd.Timestamp(AS_OF)]
    powered_now = history.loc[history["it_power_mw"] > 0, "campus"]
    campuses["status"] = campuses["name"].isin(set(powered_now)).map({True: "operating", False: "planned"})
    return campuses


METRIC_COLUMNS = (
    "buildings",
    "it_power_mw",
    "power_mw",
    "h100e",
    "performance",
    "cost_b",
    "compute_cost_b",
    "construction_cost_b",
    "operating_cost_b",
)


def _chip_label(text) -> str:
    parts = [part.strip() for part in str(text or "").split(",") if part.strip() and part.strip().lower() != "nan"]
    return ", ".join(parts) if parts else "Unknown"


def _prepare_timelines() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "data_center_timelines.csv")
    timelines = pd.DataFrame(
        {
            "campus": raw["Data center"],
            "date": pd.to_datetime(raw["Date"], errors="coerce"),
            "status": raw["Construction status"].fillna("").astype(str),
            "buildings": _to_number(raw["Buildings operational"]),
            "it_power_mw": _to_number(raw["IT power (MW)"]),
            "power_mw": _to_number(raw["Power (MW)"]),
            "h100e": _to_number(raw["H100 equivalents"]),
            "performance": _to_number(raw["Performance (8-bit OP/s)"]),
            "cost_b": _to_number(raw["Total capital cost (2025 USD billions)"]),
            "compute_cost_b": _to_number(raw["Compute cost (2025 USD billions)"]),
            "construction_cost_b": _to_number(raw["Construction cost (2025 USD billions)"]),
            "operating_cost_b": _to_number(raw["Annual operating cost (2025 USD billions)"]),
        }
    )
    timelines = timelines.dropna(subset=["date"]).sort_values(["campus", "date"])
    return _repair_timeline(timelines)


def _repair_timeline(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill compute holes that are zero-power or unchanged from the previous row.

    A missing building count is left empty. The 2028 Stargate Milam row says the
    site is finished but never gives a building count.
    """
    frame = frame.copy()
    for indexes in frame.groupby("campus", sort=False).groups.values():
        previous_power = None
        previous_h100 = None
        previous_performance = None
        for index in indexes:
            power = frame.at[index, "it_power_mw"]
            if pd.isna(frame.at[index, "h100e"]):
                if pd.notna(power) and power == 0:
                    frame.at[index, "h100e"] = 0.0
                    frame.at[index, "performance"] = 0.0
                elif (
                    previous_h100 is not None
                    and pd.notna(power)
                    and power == previous_power
                ):
                    frame.at[index, "h100e"] = previous_h100
                    frame.at[index, "performance"] = previous_performance
            if pd.notna(frame.at[index, "h100e"]):
                previous_h100 = float(frame.at[index, "h100e"])
                performance = frame.at[index, "performance"]
                previous_performance = None if pd.isna(performance) else float(performance)
            previous_power = None if pd.isna(power) else float(power)
    return frame


def _org_list(value: str) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    seen = []
    for chunk in str(value).split(","):
        name = canonicalize(chunk.strip())
        if name and name not in seen:
            seen.append(name)
    return seen


SCORE_FIELDS = ("compute", "params", "data")
SCORE_LABELS = {"compute": "compute", "params": "parameters", "data": "data"}
MODEL_TIERS = ("all", "notable", "frontier")
RAW_METRICS = SCORE_FIELDS
STATS = ("max", "median")
# Oracle and CoreWeave own campuses, and neither has models in this file.
MODEL_FOCUS = [name for name in FOCUS_OWNERS + LABS if name not in {"Oracle", "CoreWeave"}]


def _prepare_models() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "all_ai_models.csv")
    criteria = raw["Notability criteria"]
    criteria_text = criteria.fillna("").astype(str).str.strip()
    frontier_flag = raw["Frontier model"]
    if frontier_flag.dtype == bool:
        is_frontier = frontier_flag.fillna(False)
    else:
        is_frontier = frontier_flag.astype(str).str.strip().str.lower().eq("true")
    models = pd.DataFrame(
        {
            "model": raw["Model"].fillna("").astype(str),
            "orgs": raw["Organization"].map(_org_list),
            "date": pd.to_datetime(raw["Publication date"], errors="coerce"),
            "compute": _to_number(raw["Training compute (FLOP)"]),
            "params": _to_number(raw["Parameters"]),
            "data": _to_number(raw["Training dataset size (total)"]),
            "notable": criteria_text.ne("") & criteria_text.str.lower().ne("nan"),
            "frontier": is_frontier.fillna(False).astype(bool),
        }
    )
    models = models.dropna(subset=["date"]).reset_index(drop=True)
    models["point_org"] = models["orgs"].map(_point_org)
    return _attach_scores(models)


def _point_org(orgs: list[str]) -> str:
    for name in orgs:
        if name in MODEL_FOCUS:
            return name
    return orgs[0] if orgs else "Other"


def _attach_scores(models: pd.DataFrame) -> pd.DataFrame:
    """Percentile each metric inside a tier, across all years, then average the ones present."""
    tiers = {
        "all": pd.Series(True, index=models.index),
        "notable": models["notable"],
        "frontier": models["frontier"],
    }
    for tier, mask in tiers.items():
        percentiles = pd.DataFrame(index=models.index)
        for field in SCORE_FIELDS:
            values = models[field].where(mask & (models[field] > 0))
            percentiles[field] = values.rank(method="average", pct=True)
        present = percentiles.notna()
        score = percentiles.mean(axis=1, skipna=True).where(present.any(axis=1))
        models[f"score_{tier}"] = score
        models[f"used_{tier}"] = present.apply(
            lambda row: ", ".join(SCORE_LABELS[field] for field in SCORE_FIELDS if row[field]),
            axis=1,
        )
    return models


def _prepare_staff() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "ai_companies_staff_reports.csv")
    excluded = raw["Exclude from graph view"].fillna("").astype(str).str.strip().str.lower()
    kept = raw.loc[~excluded.isin(["true", "1", "yes"])]
    staff = pd.DataFrame(
        {
            "company": kept["Company"].map(canonicalize),
            "date": pd.to_datetime(kept["Date"], errors="coerce"),
            "staff": _to_number(kept["Staff count"]),
        }
    )
    staff = staff.dropna(subset=["date", "staff"])
    staff = staff.loc[staff["company"].isin(REVENUE_COMPANIES)]
    return staff.sort_values("date")


@lru_cache(maxsize=1)
def load():
    timelines = _prepare_timelines()
    campuses = _prepare_campuses(timelines)
    keep = set(campuses["name"])
    timelines = timelines.loc[timelines["campus"].isin(keep)]
    models = _prepare_models()
    return {
        "campuses": campuses,
        "timelines": timelines,
        "monthly": {column: _monthly_matrix(timelines, column) for column in METRIC_COLUMNS},
        "models": models,
        "staff": _prepare_staff(),
    }


def _monthly_matrix(timelines: pd.DataFrame, column: str) -> pd.DataFrame:
    """One metric per campus (rows) on the first day of each month, carried forward."""
    rows = {}
    for campus, group in timelines.groupby("campus"):
        series = group.sort_values("date").set_index("date")[column]
        series = series[~series.index.duplicated(keep="last")]
        filled = series.reindex(series.index.union(MONTHS)).ffill().reindex(MONTHS)
        rows[campus] = filled.fillna(0)
    return pd.DataFrame(rows).T


def company_order() -> list[str]:
    campuses = load()["campuses"]
    totals = []
    for name in FOCUS_OWNERS:
        total = campuses.loc[campuses["owner"] == name, "h100e"].sum()
        totals.append((total, name))
    totals.sort(reverse=True)
    return [name for _, name in totals] + LABS


def campuses_for_companies(campuses: pd.DataFrame, companies: list[str], role: str = "owner") -> pd.DataFrame:
    if not companies:
        return campuses
    wanted = set(companies)
    if role == "user":
        return campuses.loc[campuses["users"].map(lambda users: bool(wanted & set(users)))]
    return campuses.loc[campuses["owner"].isin(wanted)]


def attribution(campuses: pd.DataFrame, role: str) -> pd.DataFrame:
    """One row per campus and company; shares of a campus sum to 1."""
    rows = []
    for record in campuses[["name", "owner", "users"]].to_dict(orient="records"):
        if role == "user":
            users = record["users"]
            if not users:
                rows.append((record["name"], "Unknown", 1.0))
                continue
            for user in users:
                rows.append((record["name"], user, 1.0 / len(users)))
        else:
            rows.append((record["name"], record["owner"] or "Unknown", 1.0))
    frame = pd.DataFrame(rows, columns=["campus", "company", "share"])
    return frame.groupby(["campus", "company"], as_index=False)["share"].sum()


def compute_series(monthly: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Monthly metric by company (columns), indexed by month."""
    if shares.empty:
        return pd.DataFrame(index=MONTHS)
    values = monthly.reindex(shares["campus"]).fillna(0).to_numpy() * shares["share"].to_numpy()[:, None]
    frame = pd.DataFrame(values, columns=range(len(MONTHS)))
    frame["company"] = shares["company"].to_numpy()
    series = frame.groupby("company").sum().T
    series.index = MONTHS
    return series


def _orgs_overlap(orgs: list[str], wanted: set[str]) -> bool:
    return bool(wanted.intersection(orgs))


def _companies_named(names: list[str] | None) -> list[str]:
    if not names:
        return []
    chosen = set(names)
    return [name for name in company_order() if name in chosen]


def _filter_models(models: pd.DataFrame, names: list[str] | None, start, end) -> pd.DataFrame:
    scoped = models
    if names is not None:
        wanted = set(names)
        scoped = scoped.loc[scoped["orgs"].map(lambda orgs: _orgs_overlap(orgs, wanted))]
    if start and end:
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    else:
        lo, hi = MONTHS[0], MONTHS[-1]
    return scoped.loc[(scoped["date"] >= lo) & (scoped["date"] <= hi)].copy()


def _yearly_stats(models: pd.DataFrame, companies: list[str] | None, column: str, stat: str) -> dict:
    """One row per company, one column per publication year. Missing years stay empty."""
    frame = models.loc[models[column].notna(), ["orgs", "date", column]].copy()
    if column != "score":
        frame = frame.loc[frame[column] > 0]
    if frame.empty:
        return {"years": [], "rows": [{"company": name} for name in (companies or [])]}
    frame = frame.explode("orgs").rename(columns={"orgs": "company", column: "value"})
    frame = frame.dropna(subset=["company"])
    if companies is not None:
        frame = frame.loc[frame["company"].isin(companies)]
    if frame.empty:
        return {"years": [], "rows": [{"company": name} for name in companies]}
    frame["year"] = frame["date"].dt.year.astype(int)
    grouped = frame.groupby(["company", "year"])["value"].agg(stat)
    years = [str(year) for year in sorted(grouped.index.get_level_values("year").unique())]
    if companies is None:
        present = set(grouped.index.get_level_values("company"))
        focus = [name for name in company_order() if name in present and name in MODEL_FOCUS]
        rest = sorted((name for name in present if name not in focus), key=str.lower)
        names = focus + rest
    else:
        names = list(companies)
    rows = []
    for name in names:
        row = {"company": name}
        for year in years:
            key = (name, int(year))
            if key in grouped.index:
                value = float(grouped.loc[key])
                row[year] = None if pd.isna(value) else value
            else:
                row[year] = None
        rows.append(row)
    return {"years": years, "rows": rows}


def _weighted_snapshot(campuses: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    metrics = campuses.set_index("name")[["h100e", "snapshot_power_mw", "cost_b"]]
    frame = shares.merge(metrics, left_on="campus", right_index=True, how="left")
    for column in ("h100e", "snapshot_power_mw", "cost_b"):
        frame[column] = frame[column].fillna(0) * frame["share"]
    return frame


def year_end_month(year: int) -> pd.Timestamp:
    """Last month of a year in the timeline. 2030 only has January."""
    months = MONTHS[MONTHS.year == int(year)]
    return months[-1] if len(months) else MONTHS[-1]


def _levels(monthly: pd.DataFrame, shares: pd.DataFrame, stamp: pd.Timestamp) -> pd.Series:
    if shares.empty or stamp not in monthly.columns:
        return pd.Series(0.0, index=shares.index)
    values = monthly[stamp].reindex(shares["campus"]).fillna(0).to_numpy() * shares["share"].to_numpy()
    return pd.Series(values, index=shares.index)


def share_segments(monthly: pd.DataFrame, shares: pd.DataFrame, stamp: pd.Timestamp, kind: str) -> list[dict]:
    levels = _levels(monthly, shares, stamp)
    if levels.empty:
        return []
    frame = shares[["campus", "company"]].copy()
    frame["level"] = levels.reindex(frame.index).fillna(0).to_numpy()
    if kind == "campus":
        grouped = frame.groupby(["company", "campus"], as_index=False)["level"].sum()
    else:
        grouped = frame.groupby("company", as_index=False)["level"].sum()
    grouped = grouped.loc[grouped["level"] > 0].sort_values("level", ascending=False)
    return [
        {
            "label": str(row.campus if kind == "campus" else row.company),
            "company": str(row.company),
            "value": float(row.level),
        }
        for row in grouped.itertuples(index=False)
    ]


def _year_company_rows(data: dict, shares: pd.DataFrame, stamp: pd.Timestamp, names: list[str]) -> list[dict]:
    h100 = _levels(data["monthly"]["h100e"], shares, stamp)
    power = _levels(data["monthly"]["power_mw"], shares, stamp)
    cost = _levels(data["monthly"]["cost_b"], shares, stamp)
    counts = shares.groupby("company")["campus"].nunique()
    rows = []
    for name in names:
        mask = shares["company"].to_numpy() == name
        rows.append({
            "company": name,
            "campuses": int(counts.at[name]) if name in counts.index else 0,
            "h100e": float(h100[mask].sum()) if mask.any() else 0.0,
            "power_mw": float(power[mask].sum()) if mask.any() else 0.0,
            "cost_b": float(cost[mask].sum()) if mask.any() else 0.0,
        })
    return rows


def view_state(selection: dict | None) -> dict:
    """Turn the shared selection into the frames each view draws.

    No companies selected means the same as every focus company: the current
    scope is shown in full. All nine, every organization: campuses and models
    stay unfiltered. A smaller set: the bar and the model tables keep those
    companies. On the focus time series the other companies stay in the frame
    and are dimmed through ``highlight``.
    """
    data = load()
    selection = selection or {}
    focus_names = company_order()
    scope = selection.get("model_scope") or "focus"
    if scope not in ("focus", "all"):
        scope = "focus"
    picks = [name for name in (selection.get("companies") or []) if name in focus_names]
    role = selection.get("role") or "owner"
    chips = [family for family in (selection.get("chips") or []) if family in CHIP_PATTERNS]
    metric = selection.get("capacity_metric") or "h100e"
    if metric not in METRIC_COLUMNS:
        metric = "h100e"

    base = data["campuses"]
    if chips:
        base = base.loc[base["chip_family"].isin(chips)]
    scope_names = list(focus_names) if scope == "focus" else []
    scope_campuses = campuses_for_companies(base, scope_names, role).copy()
    scope_shares = attribution(scope_campuses, role)
    if scope == "focus":
        scope_shares = scope_shares.loc[scope_shares["company"].isin(focus_names)]
    picked_all = not picks or set(picks) == set(focus_names)
    if scope == "all" and picked_all:
        bar_shares = scope_shares
        series_shares = scope_shares
        highlight = []
        drilled = False
        org_names = None
        table_names = None
    elif scope == "focus" and picked_all:
        bar_shares = scope_shares
        series_shares = scope_shares
        highlight = []
        drilled = False
        org_names = list(focus_names)
        table_names = _companies_named(org_names)
    else:
        bar_shares = scope_shares.loc[scope_shares["company"].isin(picks)]
        if scope == "focus":
            series_shares = scope_shares
            highlight = picks
        else:
            series_shares = bar_shares
            highlight = []
        drilled = True
        org_names = list(picks)
        table_names = _companies_named(org_names)
    series = compute_series(data["monthly"][metric], series_shares)
    weighted = _weighted_snapshot(scope_campuses, bar_shares)
    visible = scope_campuses.loc[scope_campuses["name"].isin(set(bar_shares["campus"]))]

    chip_summary = []
    for family in chips:
        own_names = set(scope_campuses.loc[scope_campuses["chip_family"] == family, "name"])
        part = weighted.loc[weighted["campus"].isin(own_names)]
        chip_summary.append(
            {
                "chip": family,
                "campuses": int(part["campus"].nunique()),
                "h100e": float(part["h100e"].sum()),
                "cost_b": float(part["cost_b"].sum()),
            }
        )

    tier = selection.get("model_tier") or "all"
    if tier not in MODEL_TIERS:
        tier = "all"
    raw_metric = selection.get("raw_metric") or "compute"
    if raw_metric not in RAW_METRICS:
        raw_metric = "compute"
    model_stat = selection.get("model_stat") or selection.get("score_stat") or "max"
    if model_stat not in STATS:
        model_stat = "max"

    catalog = data["models"]
    if tier == "notable":
        pool = catalog.loc[catalog["notable"]].copy()
    elif tier == "frontier":
        pool = catalog.loc[catalog["frontier"]].copy()
    else:
        pool = catalog
    scored = pool.dropna(subset=[f"score_{tier}"]).copy()
    scored["score"] = scored[f"score_{tier}"]
    scored["used"] = scored[f"used_{tier}"]
    ranked = _filter_models(scored, org_names, None, None)
    rank_summary = _yearly_stats(ranked, table_names, "score", model_stat)

    frontier_pool = catalog.loc[catalog["frontier"]].copy()
    measured = _filter_models(frontier_pool, org_names, None, None)
    raw = measured.loc[measured[raw_metric] > 0].copy()
    raw["value"] = raw[raw_metric]
    raw_summary = _yearly_stats(measured, table_names, raw_metric, model_stat)

    model_years = [] if ranked.empty else sorted(int(year) for year in ranked["date"].dt.year.unique())
    requested = selection.get("model_year")
    try:
        requested = int(requested) if requested is not None else None
    except (TypeError, ValueError):
        requested = None
    model_year = requested if requested is not None and 2018 <= requested <= 2030 else (
        model_years[-1] if model_years else 2026
    )
    if model_year not in model_years:
        ranked_year = ranked.iloc[0:0]
        raw_year = raw.iloc[0:0]
    else:
        ranked_year = ranked.loc[ranked["date"].dt.year == model_year]
        raw_year = raw.loc[raw["date"].dt.year == model_year]

    stamp = year_end_month(model_year)
    kind = "campus" if drilled else "company"
    summary = _year_company_rows(data, bar_shares, stamp, picks) if drilled and len(picks) <= 3 else []
    return {
        "companies": picks,
        "scope": scope,
        "summary": summary,
        "segments": share_segments(data["monthly"][metric], bar_shares, stamp, kind),
        "segment_kind": kind,
        "highlight": highlight,
        "chip_summary": chip_summary,
        "role": role,
        "chips": chips,
        "campuses": visible,
        "shares": scope_shares,
        "series": series,
        "capacity_metric": metric,
        "h100e_now": float(weighted["h100e"].sum()) if not weighted.empty else 0.0,
        "model_tier": tier,
        "model_scope": scope,
        "raw_metric": raw_metric,
        "model_stat": model_stat,
        "model_year": model_year,
        "model_years": model_years,
        "org_names": org_names,
        "ranked": ranked,
        "ranked_year": ranked_year,
        "rank_summary": rank_summary,
        "raw": raw,
        "raw_year": raw_year,
        "raw_summary": raw_summary,
        "staff": data["staff"],
    }


def format_h100(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"
