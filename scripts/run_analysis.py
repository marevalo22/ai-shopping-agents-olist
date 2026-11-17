"""Run the full agent-side analysis on the processed run data (no API calls).

    .venv/bin/python scripts/run_analysis.py [--boot 200]

Writes charts to reports/figures/ and tables to reports/tables/.
"""
import argparse
import json

import numpy as np
import pandas as pd

from agentshop import analysis as A

ap = argparse.ArgumentParser()
ap.add_argument("--boot", type=int, default=200, help="bootstrap draws per model and category")
n_boot = ap.parse_args().boot

TAB = A.ROOT / "reports/tables"
TAB.mkdir(parents=True, exist_ok=True)
long, choices = A.load_long("choice"), A.load_choices("choice")

# 1. conditional logit per category and model (bootstrap clustered by choice set)
fits = {c: A.fit_all(long, n_boot=n_boot, category=c) for c in ["cartridge", "watch"]}
for c, f in fits.items():
    A.coef_table(f).to_csv(TAB / f"logit_{c}.csv")
    A.product_fe_table(f).to_csv(TAB / f"product_fe_{c}.csv", index=False)
    print(f"[{c}] sets:", {m: x.n_sets for m, x in f.items()}, "never chosen:",
          {m: x.dropped_products for m, x in f.items()}, "pseudo-R2:", {m: round(x.pseudo_r2, 2) for m, x in f.items()})

# 2. headroom (cartridges, in % of price) and choice-probability lift (both categories)
freight = {c: A.mean_freight(long[long.category == c]) for c in fits}
hr = A.headroom_table(fits["cartridge"], freight["cartridge"])
hr.to_csv(TAB / "headroom_cartridge.csv", index=False)
lifts = {c: A.lift_table(f, freight[c]) for c, f in fits.items()}
pd.concat([d.assign(category=c) for c, d in lifts.items()]).to_csv(TAB / "lever_lift.csv", index=False)
A.chart_headroom(hr, "cartridge")
A.chart_lift(lifts)

# 3. freight vs. price
fvp = pd.concat([A.freight_vs_price(f, float(np.exp(long[long.category == c].ln_price.mean()))).assign(category=c)
                 for c, f in fits.items()])
fvp.to_csv(TAB / "freight_vs_price.csv", index=False)

# 4. position (pooled over categories: position dummies don't depend on price scale)
pooled = A.fit_all(long)
A.chart_position(pooled)

# 5. stated vs. revealed
svr = A.stated_vs_revealed(fits, long, choices)
svr.to_csv(TAB / "stated_vs_revealed.csv", index=False)
A.chart_stated_revealed(svr)

# 6. human vs. agent share (base arm)
shares = A.agent_vs_human_shares(A.load_choices("base"))
shares.to_csv(TAB / "shares_human_vs_agent.csv", index=False)
for c in ["watch", "cartridge"]:
    A.chart_shares(shares, c)

# 7. cold-start (coldstart + coldstart2, Haiku)
cs = pd.concat([A.load_choices("coldstart"), A.load_choices("coldstart2")])
curves = A.coldstart_curves(cs)
curves.to_csv(TAB / "coldstart.csv", index=False)
A.coldstart_pooled_price(curves).to_csv(TAB / "coldstart_pooled_price.csv", index=False)

# 8. does the choice model predict the cold-start result? Logit-implied discount a 5-review newcomer needs
#    to match the established focal product's review count (median shown in the cold-start runs).
cs_long = pd.concat([pd.read_parquet(A.PROC / "coldstart_long.parquet")])
arm = cs.drop_duplicates("scenario_id").set_index("scenario_id").arm.map(json.loads)
est_ids = arm[arm.map(lambda a: a["arm"] == "established")]
focal = est_ids.map(lambda a: a["focal"])
est = cs_long[cs_long.scenario_id.isin(focal.index)]
est = est[est.product_id == est.scenario_id.map(focal)]
rows = []
for c, g in est.groupby("category"):
    old = float(g.reviews.median())
    f = fits[c]["claude-haiku-5-5"]
    rows.append({"category": c, "reviews_new": 5, "reviews_established": old,
                 "implied_price_cut": A.implied_discount(f, 5, old)})
pd.DataFrame(rows).to_csv(TAB / "coldstart_implied_discount.csv", index=False)
print(pd.DataFrame(rows))
A.chart_coldstart(curves)
print("done ->", A.FIG, TAB)
