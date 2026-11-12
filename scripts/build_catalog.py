"""Build the 8-product experiment catalogs from the raw Olist CSVs.

Olist has no product names, and its categories are not substitute sets
(e.g. `health_beauty` mixes razors and scissors). So for each experiment
category we:

  1. keep delivered orders only, joined at item level;
  2. identify the product *type* from the review text: a product qualifies
     if at least MIN_KEYWORD_REVIEWS of its reviews mention the type keyword
     (e.g. "relógio" for watches);
     Products whose reviews match an exclusion pattern (>= 2 reviews) are
     dropped, e.g. wall clocks among watches;
  3. require at least MIN_ORDERS delivered orders per product;
  4. collapse colour/size variants: products from the same seller at the
     same median price count as one listing (we keep the best seller);
  5. take the top 8 by orders and give them neutral labels ("Watch A".."H"),
     assigned in a seeded random order so the letter carries no information.

Outputs
  configs/catalogs/{key}.yaml      attributes shown to the agent + provenance
  data/processed/human_shares.csv  real Olist sales share within each set of 8

Usage
  python scripts/build_catalog.py [--raw data/raw/olist] [--seed 7]
"""

from __future__ import annotations

import argparse
import random
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

MIN_ORDERS = 30
MIN_KEYWORD_REVIEWS = 3
N_PRODUCTS = 8

CATEGORIES = {
    "watch": {
        "olist_category": "watches_gifts",
        "keyword_regex": r"rel[oó]gio",
        # Wall clocks share the keyword; drop products with >= 2 reviews saying "de parede".
        "exclude_regex": r"parede",
        "display_name": "wristwatch",
        "label_prefix": "Watch",
        "prompt_note": None,
    },
    "cartridge": {
        "olist_category": "computers_accessories",
        "keyword_regex": r"cartucho",
        "exclude_regex": None,
        "display_name": "printer ink cartridge",
        "label_prefix": "Cartridge",
        "prompt_note": "All listed cartridges are compatible with the user's printer.",
    },
}


def load(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (item-level delivered sales, product-level review text)."""
    orders = pd.read_csv(
        raw / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    orders = orders[orders.order_status == "delivered"].copy()
    day = 86400.0
    orders["promised_days"] = (
        orders.order_estimated_delivery_date - orders.order_purchase_timestamp
    ).dt.total_seconds() / day
    orders["actual_days"] = (
        orders.order_delivered_customer_date - orders.order_purchase_timestamp
    ).dt.total_seconds() / day
    orders["late"] = orders.order_delivered_customer_date > orders.order_estimated_delivery_date

    # Reviews are per order; a few orders have several reviews -> keep the latest answer.
    reviews = pd.read_csv(raw / "olist_order_reviews_dataset.csv")
    reviews = reviews.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")

    items = pd.read_csv(raw / "olist_order_items_dataset.csv")
    products = pd.read_csv(raw / "olist_products_dataset.csv")
    translation = pd.read_csv(raw / "product_category_name_translation.csv")
    products = products.merge(translation, on="product_category_name", how="left")
    products["product_category_name"] = products.product_category_name_english

    sales = (
        items.merge(orders[["order_id", "promised_days", "actual_days", "late", "order_purchase_timestamp"]], on="order_id")
        .merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
        .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    )

    text = (
        items[["order_id", "product_id"]]
        .drop_duplicates()
        .merge(reviews[["order_id", "review_comment_message"]].dropna(), on="order_id")
        .assign(msg=lambda d: d.review_comment_message.str.lower())
    )
    return sales, text


def product_stats(sales: pd.DataFrame) -> pd.DataFrame:
    # Order-level rows so multi-unit orders don't double-count reviews or delivery.
    per_order = sales.groupby(["product_id", "order_id"], as_index=False).agg(
        review_score=("review_score", "first"),
        promised_days=("promised_days", "first"),
        actual_days=("actual_days", "first"),
        late=("late", "first"),
    )
    o = per_order.groupby("product_id").agg(
        orders=("order_id", "nunique"),
        reviews=("review_score", "count"),
        rating=("review_score", "mean"),
        promised_days=("promised_days", "median"),
        actual_days=("actual_days", "median"),
        late_share=("late", "mean"),
    )
    i = sales.groupby("product_id").agg(
        units=("order_id", "size"),
        revenue=("price", "sum"),
        price=("price", "median"),
        price_cv=("price", lambda s: float(s.std() / s.mean()) if len(s) > 1 else 0.0),
        freight=("freight_value", "median"),
        sellers=("seller_id", "nunique"),
        main_seller=("seller_id", lambda s: s.value_counts().index[0]),
        category=("product_category_name", "first"),
        first_sale=("order_purchase_timestamp", "min"),
        last_sale=("order_purchase_timestamp", "max"),
    )
    return o.join(i).reset_index()


def build_one(key: str, spec: dict, stats: pd.DataFrame, text: pd.DataFrame, seed: int) -> tuple[dict, pd.DataFrame, dict]:
    pat = re.compile(spec["keyword_regex"])
    kw_reviews = (
        text[text.msg.str.contains(pat)].groupby("product_id").size().rename("keyword_reviews")
    )
    pool = stats[stats.category == spec["olist_category"]].merge(
        kw_reviews, left_on="product_id", right_index=True, how="left"
    )
    pool["keyword_reviews"] = pool.keyword_reviews.fillna(0).astype(int)

    funnel = {"in_category": len(pool)}
    pool = pool[pool.keyword_reviews >= MIN_KEYWORD_REVIEWS]
    funnel["type_match"] = len(pool)
    if spec["exclude_regex"]:
        ex = text[text.msg.str.contains(re.compile(spec["exclude_regex"]))].groupby("product_id").size()
        pool = pool[pool.product_id.map(ex).fillna(0) < 2]
        funnel["after_exclusion"] = len(pool)
    pool = pool[pool.orders >= MIN_ORDERS]
    funnel[f"orders>={MIN_ORDERS}"] = len(pool)

    # Collapse same-seller, same-price variants (keep the best seller).
    pool = pool.sort_values("orders", ascending=False)
    pool["variant_key"] = pool.main_seller + "|" + pool.price.round(2).astype(str)
    dropped_variants = pool[pool.duplicated("variant_key")].product_id.tolist()
    pool = pool.drop_duplicates("variant_key")
    funnel["after_variant_collapse"] = len(pool)

    top = pool.head(N_PRODUCTS).copy()
    if len(top) < N_PRODUCTS:
        raise SystemExit(f"{key}: only {len(top)} products qualify, need {N_PRODUCTS}")

    letters = list(string.ascii_uppercase[:N_PRODUCTS])
    random.Random(f"{seed}-{key}").shuffle(letters)
    top["label"] = [f"{spec['label_prefix']} {c}" for c in letters]
    top["human_share"] = top.orders / top.orders.sum()
    top["human_revenue_share"] = top.revenue / top.revenue.sum()
    top = top.sort_values("label")

    products = []
    for r in top.itertuples():
        products.append(
            {
                "product_id": f"{key}_{r.label.split()[-1].lower()}",
                "label": r.label,
                # Shown to the agent (base values, randomized per scenario)
                "base_price": round(float(r.price), 2),
                "base_rating": round(float(r.rating), 2),
                "base_reviews": int(r.reviews),
                "base_promised_days": int(round(r.promised_days)),
                "base_freight": round(float(r.freight), 2),
                # Human benchmark and provenance (never shown to the agent)
                "human": {
                    "orders": int(r.orders),
                    "share": round(float(r.human_share), 4),
                    "revenue_share": round(float(r.human_revenue_share), 4),
                    "actual_days_median": round(float(r.actual_days), 1),
                    "late_share": round(float(r.late_share), 3),
                    "price_cv": round(float(r.price_cv), 3),
                    "sellers": int(r.sellers),
                },
                "olist_product_id": r.product_id,
                "keyword_reviews": int(r.keyword_reviews),
            }
        )

    catalog = {
        "category": key,
        "display_name": spec["display_name"],
        "olist_category": spec["olist_category"],
        "prompt_note": spec["prompt_note"],
        "currency": "BRL",
        "selection": {
            "keyword_regex": spec["keyword_regex"],
            "exclude_regex": spec["exclude_regex"],
            "min_keyword_reviews": MIN_KEYWORD_REVIEWS,
            "min_orders": MIN_ORDERS,
            "funnel": funnel,
            "dropped_variants": dropped_variants,
            "label_seed": seed,
        },
        "products": products,
    }
    shares = top.assign(category=key)[
        ["category", "label", "product_id", "orders", "human_share", "human_revenue_share", "price", "rating", "reviews", "promised_days", "freight"]
    ].rename(columns={"product_id": "olist_product_id"})
    return catalog, shares, funnel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/olist")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sales, text = load(args.raw)
    stats = product_stats(sales)

    out_cat = ROOT / "configs/catalogs"
    out_proc = ROOT / "data/processed"
    out_cat.mkdir(parents=True, exist_ok=True)
    out_proc.mkdir(parents=True, exist_ok=True)

    all_shares = []
    for key, spec in CATEGORIES.items():
        catalog, shares, funnel = build_one(key, spec, stats, text, args.seed)
        with open(out_cat / f"{key}.yaml", "w") as f:
            yaml.safe_dump(catalog, f, sort_keys=False, allow_unicode=True)
        all_shares.append(shares)
        print(f"{key}: funnel {funnel} -> configs/catalogs/{key}.yaml")

    pd.concat(all_shares).round(4).to_csv(out_proc / "human_shares.csv", index=False)
    print("-> data/processed/human_shares.csv")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
