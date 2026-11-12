"""Seeded scenario generators.

choice     ACES Table 1 randomization + promised delivery days and freight (our additions).
base       real base attributes, only position randomized (agent share vs. Olist human share).
coldstart  one focal product is a newcomer with few reviews, plus graded advantages.
switching  cartridges only: the user "last bought" an incumbent; a rival gets graded advantages.

Every scenario is fully determined by (config seed, category, index), so regenerating
gives identical scenarios.
"""

from __future__ import annotations

import numpy as np

from .schemas import Catalog, Listing, Product, Scenario

FREE_SHIPPING_P = 0.2


def _rng(seed: int, category: str, i: int) -> np.random.Generator:
    return np.random.default_rng([seed, sum(map(ord, category)), i])


def _nonzero_freight(cat: Catalog, p: Product) -> float:
    """Products with base freight 0 still need a non-free value when the free-shipping draw fails."""
    if p.base_freight > 0:
        return p.base_freight
    return float(np.median([q.base_freight for q in cat.products if q.base_freight > 0]))


def _base_listing(p: Product, position: int) -> Listing:
    return Listing(
        position=position,
        product_id=p.product_id,
        label=p.label,
        price=round(p.base_price, 2),
        rating=round(p.base_rating, 1),
        reviews=p.base_reviews,
        promised_days=p.base_promised_days,
        freight=round(p.base_freight, 2),
    )


def _order(listings: list[Listing], rng: np.random.Generator) -> list[Listing]:
    perm = rng.permutation(len(listings))
    out = [listings[j] for j in perm]
    for k, li in enumerate(out, start=1):
        li.position = k
    return out


# --------------------------------------------------------------------------- choice (ACES Table 1)
def choice_scenario(cat: Catalog, seed: int, i: int, prefix: str) -> Scenario:
    rng = _rng(seed, cat.category, i)
    listings = []
    for p in cat.products:
        alpha = rng.uniform(-0.8, 0.8)
        free = rng.random() < FREE_SHIPPING_P
        listings.append(
            Listing(
                position=0,
                product_id=p.product_id,
                label=p.label,
                price=round(p.base_price * rng.lognormal(0, 0.3), 2),
                rating=round(min(5.0, p.base_rating + alpha * (5 - p.base_rating)), 1),
                reviews=max(1, int(round(p.base_reviews * rng.lognormal(0, 1.0)))),
                promised_days=max(1, int(round(p.base_promised_days * rng.lognormal(0, 0.3)))),
                freight=0.0 if free else round(_nonzero_freight(cat, p) * rng.lognormal(0, 0.3), 2),
            )
        )
    listings = _order(listings, rng)

    # Tags, assigned to random listings (ACES Table 1)
    idx = list(rng.permutation(len(listings)))
    n_sp = int(rng.integers(1, 5))
    for j in idx[:n_sp]:
        listings[j].sponsored = True
    listings[idx[n_sp]].overall_pick = True
    listings[idx[n_sp + 1]].scarcity = int(rng.integers(1, 6))

    return Scenario(
        scenario_id=f"{prefix}-{cat.category}-{i:04d}",
        experiment="choice",
        category=cat.category,
        seed=seed,
        listings=listings,
    )


# --------------------------------------------------------------------------- base (only position)
def base_scenario(cat: Catalog, seed: int, i: int, prefix: str) -> Scenario:
    rng = _rng(seed, cat.category, i)
    listings = _order([_base_listing(p, 0) for p in cat.products], rng)
    return Scenario(
        scenario_id=f"{prefix}-{cat.category}-{i:04d}",
        experiment="base",
        category=cat.category,
        seed=seed,
        listings=listings,
    )


# --------------------------------------------------------------------------- cold-start
COLDSTART_ARMS = [
    {"arm": "established", "lever": None, "dose": 0},
    {"arm": "newcomer", "lever": None, "dose": 0},
    *[{"arm": f"newcomer_price{int(d * 100)}", "lever": "price", "dose": d} for d in (-0.05, -0.10, -0.20, -0.30)],
    *[{"arm": f"newcomer_faster{int(d * 100)}", "lever": "delivery", "dose": d} for d in (-0.30, -0.50)],
    {"arm": "newcomer_overall_pick", "lever": "overall_pick", "dose": 1},
]
# Follow-up run (coldstart2): how many reviews does a newcomer need, and does a deep price cut help?
# Kept as a separate arm list so the original run's arm-to-scenario mapping stays unchanged.
COLDSTART_REVIEW_ARMS = [
    *[{"arm": f"newcomer_reviews{n}", "lever": "reviews", "dose": n} for n in (25, 50, 100)],
    {"arm": "newcomer_price-50", "lever": "price", "dose": -0.50},
]
NEWCOMER_REVIEWS = 5


def coldstart_scenario(cat: Catalog, seed: int, i: int, prefix: str, arms: list[dict] = COLDSTART_ARMS,
                       experiment: str = "coldstart") -> Scenario:
    """Arms cycle with i; the focal product is drawn at random so it isn't confounded with one product."""
    rng = _rng(seed, cat.category, i)
    arm = arms[i % len(arms)]
    focal = cat.products[int(rng.integers(len(cat.products)))].product_id
    listings = [_base_listing(p, 0) for p in cat.products]
    for li in listings:
        if li.product_id != focal:
            continue
        if arm["arm"] != "established":
            li.reviews = NEWCOMER_REVIEWS
        if arm["lever"] == "reviews":
            li.reviews = int(arm["dose"])
        elif arm["lever"] == "price":
            li.price = round(li.price * (1 + arm["dose"]), 2)
        elif arm["lever"] == "delivery":
            li.promised_days = max(1, int(round(li.promised_days * (1 + arm["dose"]))))
        elif arm["lever"] == "overall_pick":
            li.overall_pick = True
    listings = _order(listings, rng)
    return Scenario(
        scenario_id=f"{prefix}-{cat.category}-{i:04d}",
        experiment=experiment,
        category=cat.category,
        seed=seed,
        arm={**arm, "focal": focal, "newcomer_reviews": int(arm["dose"]) if arm["lever"] == "reviews" else NEWCOMER_REVIEWS},
        listings=listings,
    )


def li_reviews(listings: list[Listing], pid: str) -> int:
    return next(li.reviews for li in listings if li.product_id == pid)


def coldstart2_scenario(cat: Catalog, seed: int, i: int, prefix: str) -> Scenario:
    return coldstart_scenario(cat, seed, i, prefix, arms=COLDSTART_REVIEW_ARMS, experiment="coldstart2")


# --------------------------------------------------------------------------- switching (cartridges)
SWITCHING_ARMS = [
    {"arm": "no_history", "lever": None, "dose": 0},
    {"arm": "history_no_change", "lever": None, "dose": 0},
    *[{"arm": f"rival_price{int(d * 100)}", "lever": "price", "dose": d} for d in (-0.05, -0.10, -0.20, -0.30)],
    *[{"arm": f"rival_rating+{d}", "lever": "rating", "dose": d} for d in (0.1, 0.3)],
    {"arm": "rival_faster30", "lever": "delivery", "dose": -0.30},
    {"arm": "rival_overall_pick", "lever": "overall_pick", "dose": 1},
    {"arm": "incumbent_sponsored", "lever": "incumbent_sponsored", "dose": 1},
]


def switching_scenario(cat: Catalog, seed: int, i: int, prefix: str) -> Scenario:
    rng = _rng(seed, cat.category, i)
    arm = SWITCHING_ARMS[i % len(SWITCHING_ARMS)]
    ids = [p.product_id for p in cat.products]
    inc_i, riv_i = rng.choice(len(ids), size=2, replace=False)
    incumbent, rival = ids[int(inc_i)], ids[int(riv_i)]
    listings = [_base_listing(p, 0) for p in cat.products]
    for li in listings:
        if li.product_id == rival:
            if arm["lever"] == "price":
                li.price = round(li.price * (1 + arm["dose"]), 2)
            elif arm["lever"] == "rating":
                li.rating = round(min(5.0, li.rating + arm["dose"]), 1)
            elif arm["lever"] == "delivery":
                li.promised_days = max(1, int(round(li.promised_days * (1 + arm["dose"]))))
            elif arm["lever"] == "overall_pick":
                li.overall_pick = True
        if li.product_id == incumbent and arm["lever"] == "incumbent_sponsored":
            li.sponsored = True
    inc_label = next(p.label for p in cat.products if p.product_id == incumbent)
    history = None if arm["arm"] == "no_history" else (
        f"Purchase history: the user's last order in this category was {inc_label}, and they were happy with it."
    )
    listings = _order(listings, rng)
    return Scenario(
        scenario_id=f"{prefix}-{cat.category}-{i:04d}",
        experiment="switching",
        category=cat.category,
        seed=seed,
        arm={**arm, "incumbent": incumbent, "rival": rival},
        history=history,
        listings=listings,
    )


GENERATORS = {
    "choice": choice_scenario,
    "base": base_scenario,
    "coldstart": coldstart_scenario,
    "coldstart2": coldstart2_scenario,
    "switching": switching_scenario,
}


def generate(experiment: str, cat: Catalog, seed: int, n: int, prefix: str) -> list[Scenario]:
    if experiment == "switching" and cat.category != "cartridge":
        raise ValueError("switching runs on cartridges only")
    return [GENERATORS[experiment](cat, seed, i, prefix) for i in range(n)]
