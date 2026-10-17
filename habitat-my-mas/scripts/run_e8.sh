#!/bin/bash
# E8: scaling. Is the SLM planning, or filling in a template?
#   - objects per task: 2 (official) -> 4 -> 6 -> 8 (mas/datasets/scale_episodes.py)
#   - SLM-only with the four-step template (reach_v2) vs without it (free)
#   - object and place names replaced by pseudo-words (--anonymize)
# 60 episodes (5 per HSSD validation scene, scripts/e8_episodes.txt), all with >= 8 objects
# so the same episodes are used at every N. One repeat.
#
#   PYTHONPATH=. python -m mas.datasets.scale_episodes --n 4 6 8 --episode-ids $(cat scripts/e8_candidates.txt)   # once
#   HABITAT_PY=~/anaconda3/envs/habitat-mas/bin/python bash scripts/run_e8.sh
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-results/e8}
PORT=${PORT:-8766}
HABITAT_PY=${HABITAT_PY:-$(conda run -n habitat-mas which python 2>/dev/null || echo python)}
EPISODES=$(cat scripts/e8_episodes.txt)
MODELS=${MODELS:-"qwen3:4b-instruct qwen3:8b gemma3:27b"}
mkdir -p "$OUT/logs"

start_sim() {  # start_sim <benchmark>
  "$HABITAT_PY" -m mas.sim_host.server --benchmark "$1" --port "$PORT" > "$OUT/logs/sim_host_$1.log" 2>&1 &
  SIM=$!
  until grep -q -E "listening|Traceback" "$OUT/logs/sim_host_$1.log" 2>/dev/null; do sleep 1; done
}
SIM=
trap 'kill $SIM 2>/dev/null' EXIT

run() {  # run <args...>
  local name; name=$(echo "$BENCH $*" | tr ' :/' '___')
  echo "[$(date '+%m-%d %H:%M')] $BENCH $*"
  .venv/bin/python -m mas.eval.run "$@" --episode-ids $EPISODES --repeat 1 --out "$OUT" --max-tokens 8192 \
    --sim-url "http://127.0.0.1:$PORT" --feasibility-cache "$OUT/feasibility" --no-video --skip-existing \
    > "$OUT/logs/$name.log" 2>&1
  grep -E '"success_rate"' "$OUT/logs/$name.log" | sed 's/^/    /'
}

matrix() {  # the full set of configurations on the current benchmark
  run --planner classical_only
  run --planner slm_classical --model qwen3:4b-instruct
  run --planner slm_classical --model qwen3:4b-instruct --anonymize
  for m in $MODELS; do run --planner slm_only --prompt reach_v2 --normalize --repair --model "$m"; done
  for m in $MODELS; do run --planner slm_only --prompt free --normalize --repair --model "$m"; done
  for m in $MODELS; do run --planner slm_only --prompt reach_v2 --normalize --repair --anonymize --model "$m"; done
}

# N=2 (official episodes), then 4 and 8 (the ends of the curve), then 6
for BENCH in hssd_fetch_stretch hssd_n4 hssd_n8 hssd_n6; do
  start_sim $BENCH; matrix; kill $SIM; wait $SIM 2>/dev/null
done

.venv/bin/python -m mas.eval.report "$OUT" > /dev/null
echo "[$(date '+%m-%d %H:%M')] all done -> $OUT/index.html"
