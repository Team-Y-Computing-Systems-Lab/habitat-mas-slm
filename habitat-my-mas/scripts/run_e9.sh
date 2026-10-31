#!/bin/bash
# E9: the SLM improves the planner's model instead of replacing it (docs/e9_design.md).
# Receptacle occupancy is on in the sim host (--occupancy). Planner: P1 (true goal), so the
# conditions differ only in the model additions:
#   A none | B hand (upper bound) | C slm_raw | D slm_verified | E slm_hint
# Episode sets: hssd_n{2,4,6,8} (E8 episodes) and hssd_hub_n{4,6,8} (all goals on 1-2 hubs).
# Stage 0 is the pilot gate (A vs B, 10 hub episodes, N=8); stop there with GATE_ONLY=1.
#
#   PYTHONPATH=. python -m mas.datasets.scale_episodes --n 4 6 8 --hub 2 --episode-ids $(cat scripts/e8_episodes.txt)
#   HABITAT_PY=~/anaconda3/envs/habitat-mas/bin/python bash scripts/run_e9.sh
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-results/e9}
PORT=${PORT:-8767}
HABITAT_PY=${HABITAT_PY:-$(conda run -n habitat-mas which python 2>/dev/null || echo python)}
MODELS=${MODELS:-"qwen3:4b-instruct qwen3:8b gemma3:27b"}
mkdir -p "$OUT/logs"

SIM=
trap 'kill $SIM 2>/dev/null' EXIT
start_sim() {
  "$HABITAT_PY" -m mas.sim_host.server --benchmark "$1" --occupancy --port "$PORT" > "$OUT/logs/sim_host_$1.log" 2>&1 &
  SIM=$!
  until grep -q -E "listening|Traceback" "$OUT/logs/sim_host_$1.log" 2>/dev/null; do sleep 1; done
}
stop_sim() { kill $SIM 2>/dev/null; wait $SIM 2>/dev/null; SIM=; }

run() {  # run <out> <episodes> <args...>
  local out=$1 eps=$2; shift 2
  local name; name=$(echo "$BENCH $*" | tr ' :/' '___')
  echo "[$(date '+%m-%d %H:%M')] $BENCH $*"
  .venv/bin/python -m mas.eval.run --planner classical_only "$@" --episode-ids $eps --repeat 1 --out "$out" \
    --max-tokens 8192 --sim-url "http://127.0.0.1:$PORT" --feasibility-cache "$OUT/feasibility" \
    --e9-cache "$OUT" --no-video --skip-existing > "$OUT/logs/$name.log" 2>&1
  grep -E '"success_rate"|"avg_sim_steps"|"avg_wait_steps"' "$OUT/logs/$name.log" | sed 's/^/    /'
}

matrix() {  # matrix <episodes>
  run "$OUT" "$1" --aid none
  run "$OUT" "$1" --aid hand
  for m in $MODELS; do
    for c in slm_raw slm_verified slm_hint; do run "$OUT" "$1" --aid $c --aid-model "$m"; done
  done
}

# 0. pilot gate: does interference handling written by hand help at all?
BENCH=hssd_hub_n8; start_sim $BENCH
PILOT=$(cut -d' ' -f1-10 scripts/e9_hub_n8_episodes.txt)
run "$OUT/pilot" "$PILOT" --aid none
run "$OUT/pilot" "$PILOT" --aid hand
stop_sim
[ "${GATE_ONLY:-0}" = 1 ] && { echo "[$(date '+%m-%d %H:%M')] pilot done (GATE_ONLY)"; exit 0; }

# 1. contention set first (where mutexes can matter), then the E8 episodes
for n in 8 4 6; do
  BENCH=hssd_hub_n$n; start_sim $BENCH; matrix "$(cat scripts/e9_hub_n${n}_episodes.txt)"; stop_sim
done
for BENCH in hssd_n8 hssd_n4 hssd_n6 hssd_fetch_stretch; do
  start_sim $BENCH; matrix "$(cat scripts/e8_episodes.txt)"; stop_sim
done

.venv/bin/python -m mas.eval.report "$OUT" > /dev/null
echo "[$(date '+%m-%d %H:%M')] all done -> $OUT/index.html"
