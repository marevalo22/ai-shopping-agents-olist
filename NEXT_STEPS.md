# Next steps (handoff note for the next Claude session)

Last updated: 2026-11-17. Read this first, then run `git log --oneline -10` to confirm the state.

## Where we are
- All runs are done and committed locally: choice, base, coldstart and coldstart2. Total spend is $0.68 (cap $10). Nothing is pushed, and there is no GitHub repo; keep it that way unless Mo asks.
- The analysis is done. `scripts/run_analysis.py --boot 200` (~20 min) writes the charts to `reports/figures/` and the tables to `reports/tables/`.
- The team write-up is `reports/agent_results.html` (source), published privately at https://claude.ai/artifact/NYCmV8gY4hENMEE1RNrNrq. To update it, edit the file and republish to that URL. `reports/agent_results_summary.html` is a standalone copy with embedded images.

## Done 2026-11-17
- Analysis fixes in `src/agentshop/analysis.py`: free shipping now = flag on + freight to 0 (was flag only, which understated it); headroom rows beyond the tested price band (x0.56 to x1.82) are flagged `in_tested_range=False` and faded in the chart; stated vs. revealed both scaled to 100%; cold start restricted to Haiku, plus pooled price arms and the logit-implied discount; product fixed-effects tables; bootstrap CIs on freight vs. price; position removed from the lift chart (impossible shelf).
- Notebook `notebooks/agent_experiment.ipynb` (Section 5.3, runs without keys, Colab badge in README) and `README.md`.
- `reports/agent_results.html` (team write-up) still shows the pre-fix free-shipping/headroom numbers; update it from the notebook.

## Open decision (ask Mo first)
- **Persona arm:** should Haiku run the base shelf with a few shopper types (budget, fast delivery, best quality, no preference)? It costs about $0.05 and takes about 10 minutes.
- **Why:** it tests whether "agents concentrate demand on one product" still holds when users give different instructions. Right now the claim compares one default prompt with many different humans, so it's partly mechanical.
- **Either result is a finding:** concentration survives, or personalization breaks the winner-takes-all effect.
- Until it's run, phrase the claim as: "if many shoppers delegate with default instructions, demand concentrates".

## To do, in order
1. Merge the agent notebook into the team's full template notebook (sections 1–11).
2. Optional: a base-shelf arm with equal review counts, to separate "agents follow past human sales" from "agents concentrate".
3. Optional: the persona arm (above), but only if Mo says yes.

## Rules for writing up the evidence
- **Human vs. agent comparison: use watches only.** The 8 cartridges are different HP models (60XL, 662XL, 92, 74/75XL), so their human shares reflect printer ownership. Show the cartridge human comparison, but draw no conclusions from it. Cartridges remain valid for agent drivers (the logit and headroom).
- **Strong:** agent drivers; the position flip (Haiku picks position 1 2% of the time, Gemini 3.5 37%, Gemini 3.1 17%).
- **Medium:** cold-start (n = 30 per arm); watch concentration.
- **Weak:** the cartridge human comparison.
- **The listings are reconstructed from Olist transactions** (no titles, brands or photos), and humans saw more than the agents did. Say so.

## Working rules (also in CLAUDE.md)
- Use `.venv/bin/python` only.
- Never print the keys in `.env`.
- Commit locally only.
- A run under $1 may start without asking; above $1, wait for Mo's OK. Use only the cheapest model tier.
