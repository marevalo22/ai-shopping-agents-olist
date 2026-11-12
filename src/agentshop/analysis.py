"""Analysis: conditional logit, price headroom, stated-vs-revealed, cold-start curves, charts.

Everything reads the processed parquet files (no API calls), so it runs unchanged in Colab.

    from agentshop import analysis as A
    long = A.load_long("choice")                    # one row per scenario x model x listing
    fits = A.fit_all(long, n_boot=200)              # conditional logit per model (+ bootstrap CIs)
    A.headroom_table(fits)                          # ACES-style price headroom
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

from .schemas import ROOT

PROC = ROOT / "data/processed"
FIG = ROOT / "reports/figures"

# ----------------------------------------------------------------------------- palette (dataviz reference)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {  # categorical slots 1-3: validated all-pairs (CVD-safe), fixed order
    "claude-haiku-5-5": "#2a78d6",
    "gemini-3.5-flash-lite": "#eb6834",
    "gemini-3.1-flash-lite": "#1baf7a",
}
HUMAN_COLOR = "#8a8984"  # neutral: the human benchmark is not a model series
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
MODEL_LABEL = {
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "font.size": 10,
    })


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")


# ----------------------------------------------------------------------------- data
def load_choices(run: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{run}_choices.parquet")


def load_long(run: str) -> pd.DataFrame:
    """Long table with model features. Built by `agentshop.cli summary <run>`."""
    df = pd.read_parquet(PROC / f"{run}_long.parquet")
    df["ln_price"] = np.log(df.price)
    df["ln_reviews"] = np.log(df.reviews)
    df["free_ship"] = (df.freight == 0).astype(int)
    df["scarce"] = df.scarcity.notna().astype(int)
    df["sponsored"] = df.sponsored.astype(int)
    df["overall_pick"] = df.overall_pick.astype(int)
    for k in range(1, 8):  # position 8 is the omitted category (ACES headless spec)
        df[f"pos{k}"] = (df.position == k).astype(int)
    return df


def human_shares() -> pd.DataFrame:
    hs = pd.read_csv(PROC / "human_shares.csv")
    hs["product_id"] = hs.category + "_" + hs.label.str[-1].str.lower()
    return hs


# ----------------------------------------------------------------------------- conditional logit
ATTRS = ["ln_price", "rating", "ln_reviews", "promised_days", "freight", "free_ship",
         "sponsored", "overall_pick", "scarce"]
POS = [f"pos{k}" for k in range(1, 8)]


@dataclass
class Fit:
    model: str
    params: pd.Series
    bse: pd.Series
    boot: pd.DataFrame | None
    n_sets: int
    dropped_products: list[str]
    pseudo_r2: float


def _design(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attributes + position dummies + product FE (one baseline product per category).
    Products never chosen by this model are dropped, as in ACES (their FE would be -inf)."""
    chosen = d.groupby("product_id").chosen.sum()
    never = sorted(chosen[chosen == 0].index)
    d = d[~d.product_id.isin(never)]
    fe = []
    for _, g in d.groupby("category"):
        prods = sorted(g.product_id.unique())
        fe += [f"fe_{p}" for p in prods[1:]]
    X = d[ATTRS + POS].astype(float).copy()
    for c in fe:
        X[c] = (d.product_id == c[3:]).astype(float)
    return d, X, never


def _fit_once(d: pd.DataFrame):
    d, X, never = _design(d)
    X = X.loc[:, X.std() > 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ConditionalLogit(d.chosen.values, X, groups=d.scenario_id.values).fit(disp=False, method="bfgs", maxiter=500)
    return res, never, d


def fit_model(long: pd.DataFrame, model: str, n_boot: int = 0, seed: int = 0) -> Fit:
    d = long[long.model == model]
    res, never, dd = _fit_once(d)
    # McFadden pseudo-R2 against the equal-probability null
    null_ll = -dd.groupby("scenario_id").size().map(np.log).sum()
    boot = None
    if n_boot:
        rng = np.random.default_rng(seed)
        sets = d.scenario_id.unique()
        by_set = {s: g for s, g in d.groupby("scenario_id")}
        draws = []
        for b in range(n_boot):
            pick = rng.choice(sets, size=len(sets), replace=True)
            parts = [by_set[s].assign(scenario_id=f"{s}#{i}") for i, s in enumerate(pick)]
            try:
                r, _, _ = _fit_once(pd.concat(parts))
                draws.append(r.params)
            except Exception:
                continue
        boot = pd.DataFrame(draws)
    return Fit(model, res.params, res.bse, boot, dd.scenario_id.nunique(), never, 1 - res.llf / null_ll)


def fit_all(long: pd.DataFrame, n_boot: int = 0, category: str | None = None) -> dict[str, Fit]:
    """One fit per model. With `category`, only that category's choice sets (recommended for price-based
    quantities: agents are far less price-sensitive for watches than for cartridges). Without it, both
    categories are pooled with per-category product FE (fine for position and tag effects)."""
    if category:
        long = long[long.category == category]
    return {m: fit_model(long, m, n_boot=n_boot) for m in MODEL_COLORS if m in set(long.model)}


def coef_table(fits: dict[str, Fit]) -> pd.DataFrame:
    rows = {}
    for m, f in fits.items():
        rows[(m, "coef")] = f.params
        rows[(m, "se")] = f.bse
    t = pd.DataFrame(rows)
    return t.loc[[i for i in ATTRS + POS if i in t.index]]


# ----------------------------------------------------------------------------- headroom (ACES)
LEVERS = {  # name: (variable, change)
    "Overall Pick badge": ("overall_pick", 1),
    "+0.1 rating": ("rating", 0.1),
    "Double the reviews": ("ln_reviews", np.log(2)),
    "Free shipping": ("free_ship", 1),
    "Delivery 1 week faster": ("promised_days", -7),
    "Position 1 (vs. 8)": ("pos1", 1),
    "Sponsored tag": ("sponsored", 1),
}


def _headroom(params: pd.Series, var: str, dx: float) -> float:
    """exp(beta_x * dx / |beta_price|) - 1: price change that leaves utility unchanged."""
    return float(np.exp(params[var] * dx / abs(params["ln_price"])) - 1)


def headroom_table(fits: dict[str, Fit]) -> pd.DataFrame:
    rows = []
    for m, f in fits.items():
        for lever, (var, dx) in LEVERS.items():
            if var not in f.params:
                continue
            est = _headroom(f.params, var, dx)
            lo = hi = np.nan
            if f.boot is not None and len(f.boot) > 20:
                draws = f.boot.apply(lambda p: _headroom(p, var, dx), axis=1)
                lo, hi = np.percentile(draws, [2.5, 97.5])
            rows.append({"model": m, "lever": lever, "headroom": est, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _lift(params: pd.Series, var: str, dx: float, n: int = 8) -> float:
    """Chance of being picked for one of n otherwise-identical listings once it gets the lever
    (baseline 1/n): exp(b*dx) / (exp(b*dx) + n - 1). Currency-free, so it also works when the
    price coefficient is too small for a meaningful price headroom (watches)."""
    e = np.exp(params[var] * dx)
    return float(e / (e + n - 1))


def lift_table(fits: dict[str, Fit]) -> pd.DataFrame:
    rows = []
    for m, f in fits.items():
        for lever, (var, dx) in LEVERS.items():
            if var not in f.params:
                continue
            lo = hi = np.nan
            if f.boot is not None and len(f.boot) > 20:
                draws = f.boot.apply(lambda p: _lift(p, var, dx), axis=1)
                lo, hi = np.percentile(draws, [2.5, 97.5])
            rows.append({"model": m, "lever": lever, "p": _lift(f.params, var, dx), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def freight_vs_price(fits: dict[str, Fit], mean_price: float) -> pd.DataFrame:
    """R$ of price that one R$ of freight is 'worth' to the agent at the mean price.
    1.0 = treats freight exactly like price; >1 = freight-averse (as humans tend to be)."""
    return pd.DataFrame([
        {"model": m, "freight_R$_equiv_price_R$": (f.params["freight"] / (f.params["ln_price"] / mean_price))}
        for m, f in fits.items()
    ])


# ----------------------------------------------------------------------------- stated vs. revealed
MENTIONS = {
    "rating": r"\brat(?:ing|ed)\b|\bstars?\b|out of 5|/5\b",
    "price": r"\bprice|\bcheap|\bafford|\binexpensive|\bbudget|\bcost-effective|\bvalue\b|\blow(?:er|est)? cost",
    "reviews": r"\breview",
    "delivery": r"\bdeliver|\bfast(?:er|est)?\b|\bquick|\barriv|\bdays\b|\bshort(?:er|est)? (?:wait|time)",
    "freight": r"\bshipping\b|\bfreight\b",
    "overall pick": r"overall pick",
    "sponsored": r"\bsponsor",
    "position": r"\bposition\b|\blisted first\b|\bfirst (?:option|listing|product)\b|\btop of the list\b",
}
# attribute -> logit variables whose combined effect defines its revealed importance
REVEALED_VARS = {
    "rating": ["rating"], "price": ["ln_price"], "reviews": ["ln_reviews"], "delivery": ["promised_days"],
    "freight": ["freight", "free_ship"], "overall pick": ["overall_pick"], "sponsored": ["sponsored"],
    "position": POS,
}


def code_reasons(choices: pd.DataFrame) -> pd.DataFrame:
    """One 0/1 column per attribute: does the stated reason mention it?"""
    out = choices[["call_id", "model", "category", "reason"]].copy()
    txt = out.reason.fillna("").str.lower()
    for k, pat in MENTIONS.items():
        out[k] = txt.str.contains(pat, regex=True).astype(int)
    return out


def revealed_importance(fit: Fit, long: pd.DataFrame) -> pd.Series:
    """Share of the systematic-utility spread each attribute accounts for:
    sd(beta_x * x) within choice sets, summed over the attribute's variables, normalized to 1."""
    d = long[long.model == fit.model]
    imp = {}
    for k, vars_ in REVEALED_VARS.items():
        vs = [v for v in vars_ if v in fit.params]
        if not vs:
            continue
        u = sum(fit.params[v] * d[v] for v in vs)
        imp[k] = float((u - u.groupby(d.scenario_id).transform("mean")).std())
    s = pd.Series(imp)
    return s / s.sum()


def stated_vs_revealed(fits_by_cat: dict[str, dict[str, Fit]], long: pd.DataFrame,
                       choices: pd.DataFrame) -> pd.DataFrame:
    """fits_by_cat: {category: {model: Fit}}. Stated = mention rate over all reasons; revealed = the
    attribute's share of utility spread, averaged over the per-category fits."""
    coded = code_reasons(choices[choices.chosen_product_id.notna()])
    rows = []
    models = [m for m in MODEL_COLORS if any(m in f for f in fits_by_cat.values())]
    for m in models:
        stated = coded[coded.model == m][list(MENTIONS)].mean()
        revealed = pd.concat([revealed_importance(f[m], long[long.category == c])
                              for c, f in fits_by_cat.items() if m in f], axis=1).mean(axis=1)
        for k in MENTIONS:
            rows.append({"model": m, "attribute": k, "stated_mention_rate": stated.get(k, np.nan),
                         "revealed_share": revealed.get(k, np.nan)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- shares and cold-start
def agent_vs_human_shares(base_choices: pd.DataFrame) -> pd.DataFrame:
    c = base_choices[base_choices.chosen_product_id.notna()]
    agent = (c.groupby(["category", "model"]).chosen_product_id.value_counts(normalize=True)
             .rename("share").reset_index().rename(columns={"chosen_product_id": "product_id"}))
    wide = agent.pivot_table(index=["category", "product_id"], columns="model", values="share", fill_value=0)
    hs = human_shares().set_index(["category", "product_id"])["human_share"]
    wide = wide.join(hs.rename("human"), how="outer").fillna(0).reset_index()
    wide["label"] = wide.product_id.str.split("_").str[1].str.upper()
    return wide


def coldstart_curves(cs_choices: pd.DataFrame) -> pd.DataFrame:
    """P(the focal product is chosen) per arm and model, with a Wilson 95% interval."""
    c = cs_choices[cs_choices.status != "api_error"].copy()
    arm = c.arm.map(json.loads)
    c["arm_name"] = arm.map(lambda a: a["arm"])
    c["lever"] = arm.map(lambda a: a["lever"])
    c["dose"] = arm.map(lambda a: a["dose"])
    c["won"] = (c.chosen_product_id == arm.map(lambda a: a["focal"])).astype(int)
    g = c.groupby(["category", "model", "arm_name", "lever", "dose"], dropna=False).won.agg(["mean", "size"]).reset_index()
    z = 1.96
    p, n = g["mean"], g["size"]
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    g["lo"], g["hi"] = centre - half, centre + half
    return g


# ----------------------------------------------------------------------------- charts
def chart_shares(shares: pd.DataFrame, category: str):
    style()
    d = shares[shares.category == category].sort_values("human", ascending=False)
    series = [("human", "Human buyers (Olist sales)", HUMAN_COLOR)] + [
        (m, MODEL_LABEL[m], MODEL_COLORS[m]) for m in MODEL_COLORS if m in d]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(d))
    w = 0.8 / len(series)
    for i, (col, lab, colr) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, d[col] * 100, w * 0.9, color=colr, label=lab)
    ax.set_xticks(x, list(d.label))
    ax.set_xlabel(f"{category.title()} (sorted by human sales share; agent share from the base-attribute arm)")
    ax.set_ylabel("Share of purchases (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Same {category}es, different winners: human vs. agent share" if category == "watch"
                 else f"Same {category}s, different winners: human vs. agent share")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(series), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=28)
    _save(fig, f"shares_{category}")
    return fig


def chart_headroom(hr: pd.DataFrame, category: str = "cartridge"):
    style()
    levers = [l for l in LEVERS if l in set(hr.lever)]
    models = [m for m in MODEL_COLORS if m in set(hr.model)]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(levers) * len(models) / 2 + 1.5))
    y = np.arange(len(levers))
    h = 0.8 / len(models)
    for i, m in enumerate(models):
        d = hr[hr.model == m].set_index("lever").reindex(levers)
        yy = y + (i - (len(models) - 1) / 2) * h
        ax.barh(yy, d.headroom * 100, h * 0.9, color=MODEL_COLORS[m], label=MODEL_LABEL[m])
        if d.lo.notna().any():
            ax.errorbar(d.headroom * 100, yy, xerr=[(d.headroom - d.lo) * 100, (d.hi - d.headroom) * 100],
                        fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(y, levers)
    ax.invert_yaxis()
    ax.set_xlabel("Price change that keeps the same chance of being picked (%)")
    ax.set_title(f"What each lever is worth in price, per agent model ({category}s)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, f"headroom_{category}")
    return fig


def chart_position(fits: dict[str, Fit]):
    """Predicted selection share of identical products at each position (ACES Fig. 11)."""
    style()
    models = list(fits)
    mat = []
    for m in models:
        b = np.array([fits[m].params.get(p, 0.0) for p in POS] + [0.0])
        mat.append(np.exp(b) / np.exp(b).sum() * 100)
    mat = np.array(mat)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(9, 0.7 * len(models) + 1.2))
    ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=max(25, mat.max()))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}%", ha="center", va="center",
                    color="white" if mat[i, j] > 0.6 * max(25, mat.max()) else INK, fontsize=9)
    ax.set_xticks(range(8), [f"#{k}" for k in range(1, 9)])
    ax.set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models])
    ax.set_xlabel("Position in the product list (identical products)")
    ax.set_title("Where in the list an agent buys (12.5% = no position bias)")
    ax.grid(False)
    _save(fig, "position_heatmap")
    return fig


def chart_lift(lifts: dict[str, pd.DataFrame]):
    """lifts: {category: lift_table}. One panel per category; dotted line = 12.5% baseline."""
    style()
    cats = list(lifts)
    levers = [l for l in LEVERS if l in set(pd.concat(lifts.values()).lever)]
    models = [m for m in MODEL_COLORS if m in set(pd.concat(lifts.values()).model)]
    fig, axes = plt.subplots(1, len(cats), figsize=(5.2 * len(cats), 0.55 * len(levers) * len(models) / 2 + 1.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(levers))
    h = 0.8 / len(models)
    for ax, c in zip(axes, cats):
        lt = lifts[c]
        for i, m in enumerate(models):
            d = lt[lt.model == m].set_index("lever").reindex(levers)
            yy = y + (i - (len(models) - 1) / 2) * h
            ax.barh(yy, d.p * 100, h * 0.85, color=MODEL_COLORS[m], label=MODEL_LABEL[m])
            if d.lo.notna().any():
                ax.errorbar(d.p * 100, yy, xerr=[(d.p - d.lo) * 100, (d.hi - d.p) * 100],
                            fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
        ax.axvline(12.5, color=INK_2, lw=0.9, ls=":")
        ax.set_xlim(0, 100)
        ax.set_title(f"{c.title()}es" if c == "watch" else f"{c.title()}s", fontsize=10)
        ax.set_xlabel("Chance the agent picks the listing (%)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, levers)
    axes[0].invert_yaxis()
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("One listing among 8 identical ones (12.5% = dotted line): what each lever does",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    _save(fig, "lever_lift")
    return fig


def chart_coldstart(curves: pd.DataFrame, model: str = "claude-haiku-5-5"):
    """curves: coldstart_curves() over coldstart + coldstart2, with a `category` column.
    Left: newcomer review count at its normal price. Right: price cut for a 5-review newcomer."""
    style()
    d = curves[curves.model == model]
    marks = {"cartridge": ("o", "-"), "watch": ("s", "--")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colr = MODEL_COLORS[model]
    for cat, (mk, ls) in marks.items():
        g = d[d.category == cat]
        if g.empty:
            continue
        rev = g[g.arm_name == "newcomer"].assign(x=5.0)
        rev = pd.concat([rev, g[g.lever == "reviews"].assign(x=lambda t: t.dose.astype(float))]).sort_values("x")
        a1.plot(rev.x, rev["mean"] * 100, ls=ls, marker=mk, color=colr, lw=2, ms=7, label="Cartridges" if cat == "cartridge" else "Watches")
        a1.fill_between(rev.x, rev.lo * 100, rev.hi * 100, color=colr, alpha=0.08, lw=0)
        pr = pd.concat([g[g.arm_name == "newcomer"].assign(x=0.0),
                        g[g.lever == "price"].assign(x=lambda t: -t.dose.astype(float) * 100)]).sort_values("x")
        a2.plot(pr.x, pr["mean"] * 100, ls=ls, marker=mk, color=colr, lw=2, ms=7, label="Cartridges" if cat == "cartridge" else "Watches")
        a2.fill_between(pr.x, pr.lo * 100, pr.hi * 100, color=colr, alpha=0.08, lw=0)
        est = g[g.arm_name == "established"]["mean"]
        for ax in (a1, a2):
            if len(est):
                ax.axhline(float(est.iloc[0]) * 100, color=INK_2, lw=0.8, ls=ls)
    a1.set_xlabel("Newcomer's review count (normal price)")
    a1.set_xticks([5, 25, 50, 100])
    a2.set_xlabel("Newcomer's price cut, % (5 reviews)")
    a1.set_ylabel("Chance the agent picks the newcomer (%)")
    a1.set_ylim(0, 40)
    a2.text(50, 13.6, "grey: same product with all its reviews", ha="right", va="bottom", color=INK_2, fontsize=8)
    a1.legend(fontsize=9, loc="upper left")
    fig.suptitle(f"Cold start ({MODEL_LABEL[model]}): reviews win an agent, discounts don't",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    _save(fig, "coldstart")
    return fig


def chart_stated_revealed(svr: pd.DataFrame):
    style()
    models = [m for m in MODEL_COLORS if m in set(svr.model)]
    attrs = list(MENTIONS)
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        d = svr[svr.model == m].set_index("attribute").reindex(attrs)
        y = np.arange(len(attrs))
        ax.barh(y - 0.2, d.stated_mention_rate * 100, 0.38, color=MODEL_COLORS[m], label="Stated: % of reasons mentioning it")
        ax.barh(y + 0.2, d.revealed_share * 100, 0.38, color=HUMAN_COLOR, label="Revealed: % of choice driven by it")
        ax.set_yticks(y, attrs)
        ax.invert_yaxis()
        ax.set_title(MODEL_LABEL[m], fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("What agents say drives their choice vs. what actually does", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    _save(fig, "stated_vs_revealed")
    return fig
