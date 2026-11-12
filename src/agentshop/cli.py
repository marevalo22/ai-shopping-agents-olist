"""Command line entry point.

  python -m agentshop.cli plan    configs/experiments/pilot.yaml   # cost estimate only
  python -m agentshop.cli run     configs/experiments/pilot.yaml --yes
  python -m agentshop.cli parse   configs/experiments/pilot.yaml   # -> data/processed/{run}_choices.parquet
  python -m agentshop.cli summary configs/experiments/pilot.yaml
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .parse import parse
from .runner import ROOT, RunConfig, build_scenarios, log_path, print_estimate, run
from .schemas import RawCall


def parse_run(cfg: RunConfig) -> pd.DataFrame:
    data = build_scenarios(cfg)
    scen = {s.scenario_id: s for _, ss in data.values() for s in ss}
    rows = []
    for m in cfg.models:
        p = log_path(cfg, m)
        if not p.exists():
            continue
        latest: dict[str, RawCall] = {}
        for line in open(p):  # keep the latest attempt per scenario (successes win over errors)
            c = RawCall.model_validate_json(line)
            prev = latest.get(c.scenario_id)
            if prev is None or prev.error or not c.error:
                latest[c.scenario_id] = c
        for c in latest.values():
            s = scen[c.scenario_id]
            ch = parse(c, s)
            rows.append({
                **ch.model_dump(),
                "category": s.category,
                "experiment": s.experiment,
                "arm": json.dumps(s.arm) if s.arm else None,
                "model_version": c.model_version,
                "tokens_in": c.tokens_in,
                "tokens_out": c.tokens_out,
                "tokens_thinking": c.tokens_thinking,
                "cost_usd": c.cost_usd,
                "cost_usd_paid_equivalent": c.cost_usd_paid_equivalent,
                "finish_reason": c.finish_reason,
                "attempts": c.attempts,
                "error": c.error,
            })
    df = pd.DataFrame(rows)
    out = ROOT / "data/processed" / f"{cfg.name}_choices.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def long_table(cfg: RunConfig, choices: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario x model x listing, chosen in {0,1}: the conditional-logit input."""
    data = build_scenarios(cfg)
    scen = {s.scenario_id: s for _, ss in data.values() for s in ss}
    rows = []
    for c in choices[choices.chosen_product_id.notna()].itertuples():
        for li in scen[c.scenario_id].listings:
            rows.append({"scenario_id": c.scenario_id, "model": c.model, "category": c.category,
                         **li.model_dump(), "chosen": int(li.product_id == c.chosen_product_id)})
    return pd.DataFrame(rows)


def summary(cfg: RunConfig) -> None:
    df = parse_run(cfg)
    if df.empty:
        print("No calls logged yet.")
        return
    print("\nStatus by model (share of calls):")
    print(pd.crosstab(df.model, df.status, margins=True))
    print("\nTokens and cost by model:")
    print(df.groupby("model").agg(calls=("call_id", "size"), tok_in=("tokens_in", "mean"),
                                  tok_out=("tokens_out", "mean"), tok_think=("tokens_thinking", "mean"),
                                  cost=("cost_usd", "sum"), paid_equiv=("cost_usd_paid_equivalent", "sum"),
                                  versions=("model_version", lambda s: ",".join(sorted(set(s.dropna())))))
          .round(4))
    lt = long_table(cfg, df)
    lt.to_parquet(ROOT / "data/processed" / f"{cfg.name}_long.parquet", index=False)
    print("\nChoice share by product (%), per category and model:")
    for cat, g in df[df.chosen_product_id.notna()].groupby("category"):
        print(f"\n[{cat}]")
        print((pd.crosstab(g.chosen_product_id, g.model, normalize="columns") * 100).round(0))
    print("\nChosen position distribution (%):")
    print((pd.crosstab(df.chosen_position, df.model, normalize="columns") * 100).round(0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "parse", "summary"])
    ap.add_argument("config")
    ap.add_argument("--yes", action="store_true", help="actually execute (after reviewing the estimate)")
    a = ap.parse_args()
    cfg = RunConfig.load(a.config)
    if a.cmd == "plan":
        print_estimate(cfg)
    elif a.cmd == "run":
        run(cfg, yes=a.yes)
    elif a.cmd == "parse":
        print(parse_run(cfg).status.value_counts())
    else:
        summary(cfg)


if __name__ == "__main__":
    main()
