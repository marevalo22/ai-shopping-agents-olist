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
    print(f"[{c}] sets:", {m: x.n_sets for m, x in f.items()}, "never chosen:",
          {m: x.dropped_products for m, x in f.items()}, "pseudo-R2:", {m: round(x.pseudo_r2, 2) for m, x in f.items()})

# 2. headroom (cartridges, in % of price) and choice-probability lift (both categories)
hr = A.headroom_table(fits["cartridge"])
hr.to_csv(TAB / "headroom_cartridge.csv", index=False)
lifts = {c: A.lift_table(f) for c, f in fits.items()}
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
A.chart_coldstart(curves)
print("done ->", A.FIG, TAB)
