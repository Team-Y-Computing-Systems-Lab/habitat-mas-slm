#!/bin/bash
# Experiment matrix: P0 / P1 / P2 planners x models x repeats on one benchmark.
#
#   conda env "habitat-mas" must exist; Ollama must be running with the models pulled.
#   bash scripts/run_experiments.sh                       # defaults below
#   EPISODES=10 REPEATS=1 MODELS="qwen3:4b-instruct" bash scripts/run_experiments.sh
#
# Starts its own sim host, runs every configuration for repeat 1 before repeat 2
# (so a partial run still covers everything), then stops the host.
# Safe to re-run after an interruption: finished runs are skipped (--skip-existing);
# a half-finished run is started again from scratch.
set -u
cd "$(dirname "$0")/.."
BENCHMARK=${BENCHMARK:-hssd_fetch_stretch}
EPISODES=${EPISODES:-30}
REPEATS=${REPEATS:-3}
MAIN_MODEL=${MAIN_MODEL:-qwen3:4b-instruct}
SMALL_MODELS=${SMALL_MODELS:-"qwen3:0.6b llama3.2:3b gemma2:2b smollm2:1.7b"}
OUT=${OUT:-results/matrix}
PORT=${PORT:-8765}
HABITAT_PY=${HABITAT_PY:-$(conda run -n habitat-mas which python 2>/dev/null || echo python)}
mkdir -p "$OUT/logs"

"$HABITAT_PY" -m mas.sim_host.server --benchmark "$BENCHMARK" --port "$PORT" > "$OUT/logs/sim_host.log" 2>&1 &
SIM=$!
trap 'kill $SIM 2>/dev/null' EXIT
until grep -q -E "listening|Traceback" "$OUT/logs/sim_host.log" 2>/dev/null; do sleep 1; done

run() {  # run <repeat> <args...>
  local r=$1; shift
  local video=""; [ "$r" -gt 1 ] && video="--no-video"
  local name; name=$(echo "$* r$r" | tr ' :/' '___')
  echo "[$(date +%H:%M:%S)] r$r $*"
  .venv/bin/python -m mas.eval.run "$@" --episodes "$EPISODES" --repeat "$r" --out "$OUT" \
    --sim-url "http://127.0.0.1:$PORT" --feasibility-cache "$OUT/feasibility" $video --skip-existing \
    > "$OUT/logs/$name.log" 2>&1
  grep -E '"success_rate"' "$OUT/logs/$name.log" | sed 's/^/    /'
}

for r in $(seq 1 "$REPEATS"); do
  run "$r" --planner classical_only                  # P1 (also fills the feasibility cache)
  run "$r" --planner slm_only --prompt reach_v2 --model "$MAIN_MODEL"                        # P0 baseline
  run "$r" --planner slm_only --prompt reach_v2 --normalize --model "$MAIN_MODEL"            # + rewrite
  run "$r" --planner slm_only --prompt reach_v2 --normalize --repair --model "$MAIN_MODEL"   # + check/repair
  run "$r" --planner slm_classical --model "$MAIN_MODEL"                                     # P2
  for m in $SMALL_MODELS; do                         # Jetson-sized models: best P0 and P2
    run "$r" --planner slm_only --prompt reach_v2 --normalize --repair --model "$m"
    run "$r" --planner slm_classical --model "$m"
  done
done
.venv/bin/python -m mas.eval.report "$OUT" > /dev/null
echo "[$(date +%H:%M:%S)] all done -> $OUT/index.html"
