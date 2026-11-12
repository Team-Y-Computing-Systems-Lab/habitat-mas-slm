#run's SLM energy 0.028 J/token (from calibrate, per model)  ×  recorded tokens  =  energy for that run
#.venv/bin/python scripts/slm_energy.py calibrate --model qwen3:4b-instruct
#.venv/bin/python scripts/slm_energy.py apply results

import argparse, json, os, shutil, subprocess, sys, threading, time

## constants
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
CALIB_PATH = os.path.join(ROOT, 'slm_energy_calibration.json')
SLM_URL = os.environ.get('SLM_URL', 'http://localhost:11434/v1') # ollama url
PROMPT = "You are a small robot-planning assistant in a simulated home. " * 8 #calibration prompt; calibration used to assess J/token.  
MAX_TOKENS = 300
IDLE_S = 10
_last_err = ""

def fail(msg):
    print("FAIL  " +  msg)
    sys.exit(1)

# get smi binary
def smi():
    path = shutil.which("nvidia-smi")
    if path:
        return path
    for path in ("/usr/local/bin/nvidia-smi", "/usr/bin/nvidia-smi", "/usr/local/cuda/bin/nvidia-smi"):
        if os.path.exists(path):
            return path
    fail("failed to locate nvidia-smi binary")

# one nvidia smi query
def smi_query(field):
    global _last_err
    res = subprocess.run([smi(), "--query-gpu=index," + field, "--format=csv,noheader"], capture_output=True, text=True)
    if res.returncode != 0: # query failed
        _last_err = (res.stderr or res.stdout).strip().replace("\n", " ")[:300]
        return None
    vals = {}
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            # first whitespace token is the number
            vals[int(parts[0])] = float(parts[1].split()[0])
        except (ValueError, IndexError):
            # rc was 0 but the value is "[Not Supported]" / "N/A" — card lacks the field
            _last_err = "non-numeric %r for field %s" % (parts[1], field)
            return None
    return vals or None

# where joules are actually computed
def _integrate(steps):
    acc = {}
    for prev, cur, t_prev, t in steps:
        for g in prev:
            if g in cur:
                acc[g] = acc.get(g, 0.0) + (prev[g] + cur[g]) / 2.0 * (t - t_prev)
    return sum(acc.values())

# for cards with energy counter
def measure_counter(do_call, max_tokens):
    e0 = smi_query("energy_consumed")
    t0 = time.time()
    time.sleep(IDLE_S)
    e1 = smi_query("energy_consumed")
    t1 = time.time()
    idle_w = sum(e1[g] - e0[g] for g in e0 if g in e1) * 3600.0 / (t1 - t0)
    e2 = e1
    t2 = t1
    pt, ct = do_call(max_tokens)
    e3 = smi_query("energy_consumed")
    t3 = time.time()
    busy_j = sum(e3[g] - e2[g] for g in e2 if g in e3) * 3600.0
    if busy_j < 0:
        fail("energy counter went backwards (a GPU was reset mid-measurement); rerun")
    return pt, ct, idle_w, busy_j, t3 - t2

# for cards without energy counter
def measure_polling(do_call, max_tokens):
    t0 = time.time()
    prev, tries = None, 0
    while prev is None and tries < 25:
        prev = smi_query("power.draw")
        tries += 1
        if prev is None:
            time.sleep(0.2)
    if prev is None:
        fail("no usable power.draw sample in the baseline window; nvidia-smi said: " + _last_err)
    steps = []
    prev_t = t0
    skipped = 0
    while time.time() - t0 < IDLE_S:
        time.sleep(0.2)
        cur = smi_query("power.draw")
        if cur is None:
            skipped += 1
            continue
        t = time.time()
        steps.append((prev, cur, prev_t, t))
        prev, prev_t = cur, t
    if not steps:
        fail("no usable power.draw samples in the baseline window (skipped %d ticks); nvidia-smi said: " % skipped + _last_err)
    idle_w = _integrate(steps) / (steps[-1][3] - steps[0][2])


    done = []

    def worker():
        done.append(do_call(max_tokens))

    th = threading.Thread(target=worker)
    steps = []
    t2 = time.time()
    prev, tries = None, 0
    while prev is None and tries < 25:
        prev = smi_query("power.draw")
        tries += 1
        if prev is None:
            time.sleep(0.2)
    if prev is None:
        fail("no usable power.draw sample before the call window; nvidia-smi said: " + _last_err)
    prev_t = t2
    th.start()
    while th.is_alive():
        time.sleep(0.1)
        cur = smi_query("power.draw")
        if cur is None:
            skipped += 1
            continue
        t = time.time()
        steps.append((prev, cur, prev_t, t))
        prev, prev_t = cur, t
    th.join()
    cur = smi_query("power.draw")
    if cur is None:
        skipped += 1
    else:
        steps.append((prev, cur, prev_t, time.time()))
    dt = time.time() - t2
    if not steps:
        fail("no usable power.draw samples during the call window (skipped %d ticks); nvidia-smi said: " % skipped + _last_err)
    if skipped:
        print("note: %d nvidia-smi sample(s) skipped (transient driver errors)" % skipped)
    pt, ct = done[0]
    return pt, ct, idle_w, _integrate(steps), dt

# run reference SLM call to calc J/token, write to json
def calibrate(model):

    try:
        from openai import OpenAI
    except ImportError:
        fail("the openai package is missing; run calibrate with .venv/bin/python")

    cli = OpenAI(base_url=SLM_URL, api_key="ollama")
    
    def do_call(max_tokens):
        r = cli.chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}],
            max_tokens=max_tokens, temperature=0)
        return r.usage.prompt_tokens, r.usage.completion_tokens

    print("warm-up: one throwaway call so the model is resident in GPU memory")
    do_call(1)

    if smi_query("energy_consumed") is not None:
        measure, method = measure_counter, "cumulative energy counter"
    else:
        measure, method = measure_polling, "polled power draw (this GPU has no energy counter)"

    print("idle baseline: %d s, measured via %s" % (IDLE_S, method))
    pt, ct, idle_w, busy_j, dt = measure(do_call, MAX_TOKENS)
    net_j = busy_j - idle_w * dt
    if net_j <= 0:
        fail("net energy after idle subtraction is not positive; raise MAX_TOKENS and rerun")

    tokens = pt + ct
    eps = net_j / tokens

    entry = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "method": method,
             "url": SLM_URL, "prompt_tokens": pt, "completion_tokens": ct,
             "wall_s": round(dt, 2), "busy_j": round(busy_j, 1),
             "idle_w": round(idle_w, 1), "j_per_token": round(eps, 3)}

    cal = {}
    if os.path.exists(CALIB_PATH):
        cal = json.load(open(CALIB_PATH))
    cal[model] = entry
    json.dump(cal, open(CALIB_PATH, "w"), indent=2)

    print("call: %d prompt + %d completion tokens in %.2f s" % (pt, ct, dt))
    print("%s across all GPUs: %.0f J, minus idle %.0f J = %.0f J for %d tokens"
          % (method, busy_j, idle_w * dt, net_j, tokens))
    print("PASS  calibrated j_per_token = %.3f  (saved to %s)" % (eps, CALIB_PATH))


# read calibration json into dict
def load_cal():
    if os.path.exists(CALIB_PATH):
        return json.load(open(CALIB_PATH))
    return {}

# walks results and sums the token counts to calc estimated SLM energy
def apply(results_dir, override_eps):
    if not os.path.isdir(results_dir):
        fail("no such results dir: " + results_dir)
    cal = load_cal()
    rows = []
    for dirpath, _dirs, filenames in os.walk(results_dir):
        if "config.json" not in filenames or "records.jsonl" not in filenames:
            continue
        cfg = json.load(open(os.path.join(dirpath, "config.json")))
        records = [json.loads(line) for line in open(os.path.join(dirpath, "records.jsonl"))  if line.strip()]
        tin = sum(r.get("metrics", {}).get("prompt_tokens", 0) or 0 for r in records)
        tout = sum(r.get("metrics", {}).get("completion_tokens", 0) or 0 for r in records)
        model = cfg.get("model") or "none"
        n = max(len(records), 1)
        eps = override_eps
        if eps is None and model in cal:
            eps = cal[model]["j_per_token"]
        if tin + tout == 0:
            flag = "0 J (no SLM calls)"
        elif eps is None:
            flag = "unknown (no calibration for %r; run: calibrate --model %s)" % (model, model)
        else:
            est = (tin + tout) * eps
            flag = "%.0f J total, %.0f J/episode (at %.3f J/token)" % (est, est / n, eps)
        rows.append((cfg.get("benchmark", "?"), os.path.basename(dirpath), cfg.get("planner", "?"), model, tin, tout, flag))
    rows.sort(key=lambda r: (r[0], r[1]))
    if not rows:
        fail("no run folders (config.json + records.jsonl) found under " + results_dir)
    print("runs found: %d  (est. SLM energy = recorded tokens x calibrated J/token)" % len(rows))
    for bench, run_id, planner, model, tin, tout, flag in rows:
        print("[%s] %s" % (bench, run_id))
        print("    %-15s model=%-19s tokens %d+%d   est. SLM energy: %s" % (planner, model, tin, tout, flag))


def main():
    p = argparse.ArgumentParser(
        description="estimate SLM energy: calibrate joules-per-token, then apply it to recorded token counts")
    p.add_argument("mode", choices=["calibrate", "apply"])
    p.add_argument("results_dir", nargs="?", default=os.path.join(ROOT, "results"))
    p.add_argument("--model", help="model to calibrate (calibrate mode)")
    p.add_argument("--j-per-token", type=float, help="apply: override the calibrated J/token")
    a = p.parse_args()
    if a.mode == "calibrate":
        if not a.model:
            fail("calibrate needs --model (e.g. qwen3:4b-instruct)")
        calibrate(a.model)
    else:
        apply(a.results_dir, a.j_per_token)

if __name__ == "__main__":
    main()
