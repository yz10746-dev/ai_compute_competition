"""Load and clean Epoch AI tables for the dashboard.

Raw CSVs stay untouched. Speculative campuses are dropped, organization
names are aligned, and chip types are grouped into a few families.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "epoch"
AS_OF = date(2026, 11, 4)

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
    "Unattributed": "#DCDFE3",
}

CHIP_PATTERNS = {
    "NVIDIA": "",
    "TPU": "/",
    "Trainium": "x",
    "Other": ".",
    "Unspecified": "-",
}
CHIPS = list(CHIP_PATTERNS)
ROLES = ["owner", "user"]
STATUSES = ["all", "operating", "planned"]

MONTHS = pd.date_range("2018-01-01", "2030-01-01", freq="MS")
QUARTERS = pd.date_range("2018-01-01", "2030-01-01", freq="QS")

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
    campuses["color"] = campuses["owner"].where(campuses["owner"].isin(COLORS), "Other")
    campuses["color"] = campuses["color"].where(campuses["owner"].isin(FOCUS_OWNERS), "Other")

    as_of = pd.Timestamp(AS_OF)
    history = timeline.loc[timeline["date"] <= as_of].sort_values("date")
    latest = history.groupby("campus", as_index=False).tail(1)
    power = latest.set_index("campus")["it_power_mw"]
    campuses["it_power_mw"] = campuses["name"].map(power)
    campuses["it_power_mw"] = campuses["it_power_mw"].where(
        campuses["it_power_mw"] > 0, campuses["snapshot_power_mw"]
    )
    powered_now = history.loc[history["it_power_mw"] > 0, "campus"]
    campuses["status"] = campuses["name"].isin(set(powered_now)).map({True: "operating", False: "planned"})
    return campuses


def _prepare_timelines() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "data_center_timelines.csv")
    timelines = pd.DataFrame(
        {
            "campus": raw["Data center"],
            "date": pd.to_datetime(raw["Date"], errors="coerce"),
            "it_power_mw": _to_number(raw["IT power (MW)"]),
            "h100e": _to_number(raw["H100 equivalents"]).fillna(0),
            "status": raw["Construction status"].fillna("").astype(str),
        }
    )
    return timelines.dropna(subset=["date"]).sort_values(["campus", "date"])


def _org_list(value: str) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    seen = []
    for chunk in str(value).split(","):
        name = canonicalize(chunk.strip())
        if name and name not in seen:
            seen.append(name)
    return seen


def _prepare_models() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "notable_ai_models.csv")
    models = pd.DataFrame(
        {
            "model": raw["Model"],
            "orgs": raw["Organization"].map(_org_list),
            "date": pd.to_datetime(raw["Publication date"], errors="coerce"),
            "compute": _to_number(raw["Training compute (FLOP)"]),
            "params": _to_number(raw["Parameters"]),
            "domain": raw["Domain"].fillna("").astype(str),
        }
    )
    models = models.dropna(subset=["date"])
    models["has_compute"] = models["compute"] > 0
    models["primary"] = models["orgs"].map(_primary_org)
    return models


def _primary_org(orgs: list[str]) -> str:
    for name in orgs:
        if name in FOCUS_OWNERS or name in LABS:
            return name
    return "Other"


def _prepare_revenue() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "ai_companies_revenue_reports.csv")
    revenue = pd.DataFrame(
        {
            "company": raw["Company"].map(canonicalize),
            "date": pd.to_datetime(raw["Date"], errors="coerce"),
            "revenue_usd": _to_number(raw["Annualized revenue (USD)"]),
            "revenue_type": raw["Annualized revenue type"].fillna("").astype(str),
        }
    )
    revenue = revenue.dropna(subset=["date", "revenue_usd"])
    revenue = revenue.loc[revenue["company"].isin(REVENUE_COMPANIES)]
    revenue = revenue.sort_values("date").drop_duplicates(["company", "date"], keep="last")
    return revenue


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
    timelines = timelines.loc[timelines["campus"].isin(keep)].copy()
    owner = campuses.set_index("name")["owner"]
    timelines["owner"] = timelines["campus"].map(owner)
    timelines["color"] = timelines["owner"].where(timelines["owner"].isin(FOCUS_OWNERS), "Other")
    models = _prepare_models()
    return {
        "campuses": campuses,
        "timelines": timelines,
        "monthly": _monthly_matrix(timelines),
        "models": models,
        "revenue": _prepare_revenue(),
        "staff": _prepare_staff(),
    }


def _monthly_matrix(timelines: pd.DataFrame) -> pd.DataFrame:
    """H100e per campus (rows) on the first day of each month (columns), carried forward."""
    rows = {}
    for campus, group in timelines.groupby("campus"):
        series = group.sort_values("date").set_index("date")["h100e"]
        series = series[~series.index.duplicated(keep="last")]
        rows[campus] = series.reindex(series.index.union(MONTHS)).ffill().fillna(0).reindex(MONTHS)
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
                rows.append((record["name"], "Unattributed", 1.0))
                continue
            for user in users:
                rows.append((record["name"], user if user in COLORS else "Other", 1.0 / len(users)))
        else:
            owner = record["owner"]
            rows.append((record["name"], owner if owner in FOCUS_OWNERS else "Other", 1.0))
    frame = pd.DataFrame(rows, columns=["campus", "company", "share"])
    return frame.groupby(["campus", "company"], as_index=False)["share"].sum()


def compute_series(monthly: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Monthly H100e by company (columns), indexed by month."""
    if shares.empty:
        return pd.DataFrame(index=MONTHS)
    values = monthly.reindex(shares["campus"]).fillna(0).to_numpy() * shares["share"].to_numpy()[:, None]
    frame = pd.DataFrame(values, columns=range(len(MONTHS)))
    frame["company"] = shares["company"].to_numpy()
    series = frame.groupby("company").sum().T
    series.index = MONTHS
    return series


def campus_h100e_at(monthly: pd.DataFrame, names, end) -> pd.Series:
    month = _month_on_or_before(end)
    return monthly.reindex(list(names)).fillna(0)[month]


def _month_on_or_before(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value) if value else MONTHS[-1]
    position = MONTHS.searchsorted(stamp, side="right") - 1
    return MONTHS[max(0, min(position, len(MONTHS) - 1))]


def frontier(models: pd.DataFrame, companies: list[str]) -> pd.DataFrame:
    """Models per company with the running max of training compute and a record flag."""
    pieces = []
    scoped = models.loc[models["has_compute"]]
    for company in companies:
        group = scoped.loc[scoped["orgs"].map(lambda orgs: company in orgs)].sort_values("date").copy()
        if group.empty:
            continue
        group["company"] = company
        group["best"] = group["compute"].cummax()
        group["record"] = group["compute"] > group["best"].shift(1).fillna(0)
        pieces.append(group)
    if not pieces:
        return pd.DataFrame(columns=list(scoped.columns) + ["company", "best", "record"])
    return pd.concat(pieces, ignore_index=True)


def view_state(selection: dict | None) -> dict:
    """Apply company, campus, role, chip, status, and date filters shared by every view."""
    data = load()
    selection = selection or {}
    companies = [name for name in (selection.get("companies") or []) if name in COLORS]
    campus = selection.get("campus") or None
    start = selection.get("date_start") or None
    end = selection.get("date_end") or None
    role = selection.get("role") or "owner"
    chips = [family for family in (selection.get("chips") or []) if family in CHIP_PATTERNS]
    status = selection.get("status") or "all"

    base = data["campuses"]
    if chips:
        base = base.loc[base["chip_family"].isin(chips)]
    if status != "all":
        base = base.loc[base["status"] == status]
    campuses = campuses_for_companies(base, companies, role).copy()

    shares = attribution(campuses, role)
    if companies:
        shares = shares.loc[shares["company"].isin(companies)]
    series = compute_series(data["monthly"], shares)

    at_end = campus_h100e_at(data["monthly"], campuses["name"], end)
    campuses["h100e_end"] = campuses["name"].map(at_end).fillna(0)
    share_at_end = shares.assign(value=shares["campus"].map(at_end).fillna(0) * shares["share"])

    summary = []
    for name in companies:
        own = campuses.loc[campuses_for_companies(campuses, [name], role).index]
        summary.append(
            {
                "company": name,
                "campuses": int((own["h100e_end"] > 0).sum()),
                "h100e": float(share_at_end.loc[share_at_end["company"] == name, "value"].sum()),
                "cost_b": float(own["cost_b"].sum()),
            }
        )

    chip_summary = []
    for family in chips:
        own = campuses.loc[campuses["chip_family"] == family]
        chip_summary.append(
            {
                "chip": family,
                "campuses": int((own["h100e_end"] > 0).sum()),
                "h100e": float(share_at_end.loc[share_at_end["campus"].isin(own["name"]), "value"].sum()),
                "cost_b": float(own["cost_b"].sum()),
            }
        )

    campus_series = None
    if campus and campus in data["monthly"].index:
        campus_series = data["monthly"].loc[campus]

    model_companies = selection.get("model_companies") or None
    if model_companies:
        frontier_companies = list(model_companies)
    elif companies:
        frontier_companies = companies
    else:
        frontier_companies = FOCUS_OWNERS + LABS
    models = frontier(data["models"], frontier_companies)
    in_window = models
    if start and end:
        in_window = models.loc[(models["date"] >= pd.Timestamp(start)) & (models["date"] <= pd.Timestamp(end))]

    return {
        "companies": companies,
        "summary": summary,
        "chip_summary": chip_summary,
        "campus": campus,
        "role": role,
        "chips": chips,
        "status": status,
        "date_start": start,
        "date_end": end,
        "end_month": _month_on_or_before(end),
        "campuses": campuses,
        "shares": shares,
        "series": series,
        "campus_series": campus_series,
        "h100e_end": float(share_at_end["value"].sum()),
        "models": models,
        "models_in_window": in_window,
        "revenue": data["revenue"],
        "staff": data["staff"],
    }


def quarter_date(index: int) -> str:
    return QUARTERS[max(0, min(int(index), len(QUARTERS) - 1))].date().isoformat()


def quarter_index(value, round_up: bool = False) -> int:
    stamp = pd.Timestamp(value)
    side = "left" if round_up else "right"
    position = QUARTERS.searchsorted(stamp, side=side) - (0 if round_up else 1)
    return max(0, min(int(position), len(QUARTERS) - 1))


def format_h100(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def parse_bound(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(value / 1000).date()
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    head = text[:10]
    try:
        return datetime.strptime(head, "%Y-%m-%d").date()
    except ValueError:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
