#!/usr/bin/env bash
# Run the full experiment sequence and commit locally after each run.
# Safe to re-run: every run resumes (only missing or failed calls are made),
# so after a Gemini daily quota runs out, just run this again the next day.
#
#   bash scripts/run_all.sh            # choice -> base -> coldstart -> coldstart2
#   bash scripts/run_all.sh base       # a subset, in the given order
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNS=("${@:-choice base coldstart}")
[ $# -eq 0 ] && RUNS=(choice base coldstart coldstart2)

for run in "${RUNS[@]}"; do
  cfg="configs/experiments/${run}.yaml"
  echo "=== $(date '+%F %T') starting ${run}"
  $PY -u -m agentshop.cli run "$cfg" --yes || echo "!!! ${run} exited with an error (continuing)"
  $PY -u -m agentshop.cli summary "$cfg" > "data/processed/${run}_summary.txt" 2>&1 || true
  git add data/logs data/processed data/scenarios
  if ! git diff --cached --quiet; then
    stats=$($PY - "$run" <<'EOF'
import sys, pandas as pd
df = pd.read_parquet(f"data/processed/{sys.argv[1]}_choices.parquet")
parts = [f"{m}: {len(g)} calls, {(g.status == 'api_error').sum()} errors, ${g.cost_usd.sum():.4f}"
         for m, g in df.groupby("model")]
print("\n".join(parts))
EOF
)
    git commit -q -F - <<EOF
Run ${run}: log calls and parsed choices

${stats}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
    echo "=== committed ${run}"
  fi
done
echo "=== $(date '+%F %T') all done; total spend: $($PY -c 'from agentshop.runner import spent_usd; print(f"${spent_usd():.4f}")')"
