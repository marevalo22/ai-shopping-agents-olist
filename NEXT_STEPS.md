# Next steps (handoff note for the next Claude session)

Last updated: 2026-11-13. Read this first, then run `git log --oneline -10` to confirm the state.

## Where we are
- All runs are done and committed locally: choice, base, coldstart and coldstart2. Total spend is $0.68 (cap $10). Nothing is pushed, and there is no GitHub repo; keep it that way unless Mo asks.
- The analysis is done. `scripts/run_analysis.py --boot 200` (~20 min) writes the charts to `reports/figures/` and the tables to `reports/tables/`.
- The team write-up is `reports/agent_results.html` (source), published privately at https://claude.ai/artifact/NYCmV8gY4hENMEE1RNrNrq. To update it, edit the file and republish to that URL. `reports/agent_results_summary.html` is a standalone copy with embedded images.

## Open decision (ask Mo first)
- **Persona arm:** should Haiku run the base shelf with a few shopper types (budget, fast delivery, best quality, no preference)? It costs about $0.05 and takes about 10 minutes.
- **Why:** it tests whether "agents concentrate demand on one product" still holds when users give different instructions. Right now the claim compares one default prompt with many different humans, so it's partly mechanical.
- **Either result is a finding:** concentration survives, or personalization breaks the winner-takes-all effect.
- Until it's run, phrase the claim as: "if many shoppers delegate with default instructions, demand concentrates".

## To do, in order
1. **Colab notebook:** follow the course template (sections 1–11) for the agent part, 5.3. It reads the committed `reports/tables/*.csv` and `data/processed/*` and needs no API keys. Include the charts and plain-language interpretation.
2. **README:** the research question, the design, how to run it, the results and charts. Cite ACES (arXiv 2508.02630) and link the Kaggle Olist dataset.
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
