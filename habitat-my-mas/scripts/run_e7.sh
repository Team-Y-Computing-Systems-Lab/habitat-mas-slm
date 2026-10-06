#!/bin/bash
# E7: scale-up. 120 episodes (10 per HSSD validation scene, scripts/e7_episodes.txt),
# larger LLMs, repair, and replanning after failures. One repeat.
#
#   HABITAT_PY=~/anaconda3/envs/habitat-mas/bin/python bash scripts/run_e7.sh
#
# Stages run in order, so the headline results arrive first; finished runs are
# skipped when the script is started again (--skip-existing).
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-results/e7}
PORT=${PORT:-8765}
HABITAT_PY=${HABITAT_PY:-$(conda run -n habitat-mas which python 2>/dev/null || echo python)}
EPISODES=$(cat scripts/e7_episodes.txt)
LARGE=${LARGE:-"qwen3:4b-instruct qwen3:8b llama3.1:8b deepseek-r1:8b gemma3:12b gpt-oss:20b gemma3:27b qwen3.6:35b"}
SMALL=${SMALL:-"gemma2:2b llama3.2:3b qwen3:0.6b smollm2:1.7b qwen3:4b-instruct"}
mkdir -p "$OUT/logs"

"$HABITAT_PY" -m mas.sim_host.server --port "$PORT" > "$OUT/logs/sim_host.log" 2>&1 &
SIM=$!
trap 'kill $SIM 2>/dev/null' EXIT
until grep -q -E "listening|Traceback" "$OUT/logs/sim_host.log" 2>/dev/null; do sleep 1; done

run() {  # run <video:yes|no> <args...>
  local video=$1; shift
  local flag="--no-video"; [ "$video" = yes ] && flag=""
  local name; name=$(echo "$*" | tr ' :/' '___')
  echo "[$(date '+%m-%d %H:%M')] $*"
  .venv/bin/python -m mas.eval.run "$@" --episode-ids $EPISODES --repeat 1 --out "$OUT" --max-tokens 8192 \
    --sim-url "http://127.0.0.1:$PORT" --feasibility-cache "$OUT/feasibility" $flag --skip-existing \
    > "$OUT/logs/$name.log" 2>&1
  grep -E '"success_rate"' "$OUT/logs/$name.log" | sed 's/^/    /'
}

# A. headline: classical with and without replanning; SLM + classical with every model
run yes --planner classical_only
run no  --planner classical_only --replan 2
for m in $LARGE; do run "$( [ "$m" = qwen3:4b-instruct ] && echo yes || echo no )" --planner slm_classical --model "$m"; done
# B. SLM-only with repair, every larger model
for m in $LARGE; do run no --planner slm_only --prompt reach_v2 --normalize --repair --model "$m"; done
# C. replanning after failures
for m in $LARGE; do run no --planner slm_classical --replan 2 --model "$m"; done
for m in $LARGE; do run no --planner slm_only --prompt reach_v2 --normalize --repair --replan 2 --model "$m"; done
# D. small models without repair (open question from E6)
for m in $SMALL; do run no --planner slm_only --prompt reach_v2 --normalize --model "$m"; done

.venv/bin/python -m mas.eval.report "$OUT" > /dev/null
echo "[$(date '+%m-%d %H:%M')] all done -> $OUT/index.html"
