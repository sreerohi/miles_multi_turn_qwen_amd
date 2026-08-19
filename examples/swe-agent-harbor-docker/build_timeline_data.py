#!/usr/bin/env python3
"""Build the JSON data files that ``rollout_timeline.html`` reads.

The timeline page is static and loads four files from its own directory:
    timeline_metrics.json   per-step perf/agent/rollout metrics (from train.log)
    per_step_trials.json    each step's real trials, mapped by wall-clock window
    trial_phase_stats.json  run-wide per-phase duration distribution
    timeline_source.json    provenance (log mtime, parse time, step count)

Everything here is parsed from run outputs — nothing is invented:
  * train.log            trainer/rollout/session-server stdout (== dicts sent to W&B)
  * cc_trials/<id>/result.json        Harbor per-trial TimingInfo + verifier reward
  * cc_trials/<id>/agent/trajectory.json   mini-swe-agent per-turn steps (for turns)

Trials are assigned to a rollout step by wall clock: step N's rollout ended at the
timestamp of its ``perf N:`` log line, so its window is
    [end - perf/rollout_time , end]
and a trial belongs to step N when its ``started_at`` falls in that window.
(NB: trajectory step timestamps are written at serialization, so they are NOT real
per-turn times; agent_run is therefore not split here.)

Usage:
    python build_timeline_data.py                      # uses defaults below
    python build_timeline_data.py --work-dir /path/to/runtime/work --out-dir .
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
from datetime import datetime

TS_LOG = "%Y-%m-%d %H:%M:%S"
TS_ISO = "%Y-%m-%dT%H:%M:%S.%fZ"

ROLLOUT_KEYS = {"perf/rollout_time", "perf/tokens_per_gpu_per_sec"}
AGENT_KEYS = {
    "agent/agent_run_time_mean", "agent/eval_time_mean", "agent/turns_mean", "agent/total_time_mean",
    "agent/spawn_time_mean", "agent/agent_setup_time_mean", "agent/teardown_overhead_time_mean",
}
TRAIN_KEYS = {
    "perf/train_time", "perf/train_wait_time", "perf/actor_train_time", "perf/log_probs_time",
    "perf/ref_log_probs_time", "perf/data_preprocess_time", "perf/update_weights_time",
    "perf/update_weights_implementation_time", "perf/finalize_and_resume_engines_time",
    "perf/ref_model_update_time", "perf/step_time", "perf/wait_time_ratio",
}
WANT = ROLLOUT_KEYS | AGENT_KEYS | TRAIN_KEYS | {"rollout/raw_reward"}


def _iso(x):
    try:
        return datetime.strptime(x, TS_ISO).timestamp()
    except Exception:
        return None


def _dur(a, b):
    A, B = _iso(a), _iso(b)
    return round(B - A, 1) if (A is not None and B is not None) else None


def _timing(o, k):
    v = o.get(k)
    if isinstance(v, dict):
        return _dur(v.get("started_at"), v.get("finished_at"))
    return None


def parse_step_metrics(train_log: str) -> dict:
    """Merge the per-step ``perf N:`` / ``rollout N:`` metric dicts from train.log."""
    steps: dict[int, dict] = {}
    rx = re.compile(r"(?:perf|rollout) (\d+): (\{.*\})\s*$")
    with open(train_log, errors="replace") as f:
        for line in f:
            m = rx.search(line.strip())
            if not m:
                continue
            n = int(m.group(1))
            try:
                import ast
                d = ast.literal_eval(m.group(2))
            except Exception:
                continue
            s = steps.setdefault(n, {})
            for k, v in d.items():
                if k in WANT:
                    s[k] = round(v, 3) if isinstance(v, float) else v
    return {n: steps[n] for n in sorted(steps) if "perf/rollout_time" in steps[n]}


def step_windows(train_log: str) -> dict:
    """[start, end] wall-clock window per step, from the rollout-side perf line."""
    rx = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ [^\]]+\] metrics\.py:\d+ - perf (\d+): (.*'perf/rollout_time'.*)$")
    wins = {}
    with open(train_log, errors="replace") as f:
        for line in f:
            m = rx.search(line.strip())
            if not m:
                continue
            end = datetime.strptime(m.group(1), TS_LOG).timestamp()
            n = int(m.group(2))
            mm = re.search(r"'perf/rollout_time': ([0-9.]+)", m.group(3))
            if mm:
                wins[n] = (end - float(mm.group(1)), end)
    return wins


def parse_gen_times(train_log: str) -> dict:
    """Sum the session server's per-call gen_time (GPU generation) by session_id.

    Lines look like: ``[session-server] GEN session_id=<hex> gen_time=12.34s status=200``
    """
    gen: dict[str, float] = {}
    rx = re.compile(r"GEN session_id=([0-9a-fA-F]+) gen_time=([0-9.]+)s")
    with open(train_log, errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                gen[m.group(1)] = gen.get(m.group(1), 0.0) + float(m.group(2))
    return gen


def _turns(traj_path: str):
    # Try canonical trajectory.json format (steps[].source == "agent")
    try:
        steps = json.load(open(traj_path)).get("steps", [])
        return sum(1 for s in steps if isinstance(s, dict) and s.get("source") == "agent")
    except Exception:
        pass
    # Fallback: mini-swe-agent.trajectory.json format (messages[].role == "assistant")
    alt = os.path.join(os.path.dirname(traj_path), "mini-swe-agent.trajectory.json")
    try:
        msgs = json.load(open(alt)).get("messages", [])
        return sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant")
    except Exception:
        return None


def parse_trials(trials_dir: str, gen_by_session: dict | None = None) -> list:
    gen_by_session = gen_by_session or {}
    rows = []
    for fp in glob.glob(os.path.join(trials_dir, "*", "result.json")):
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        st_ = _iso(r.get("started_at", ""))
        if st_ is None:
            continue
        vr = r.get("verifier_result") or {}
        rw = vr.get("rewards") or {}
        reward = rw.get("reward", next(iter(rw.values()), None)) if isinstance(rw, dict) else None
        # Extract session_id from lock.json (always present): Harbor embeds it in
        # the OPENAI_API_BASE URL as .../sessions/<session_id>/v1
        rid, sid = None, None
        lock = os.path.join(os.path.dirname(fp), "lock.json")
        if os.path.exists(lock):
            try:
                lk = json.load(open(lock))
                base_url = (
                    lk.get("agent", {}).get("env", {}).get("OPENAI_API_BASE", "")
                )
                # URL format: http://host:port/sessions/<uuid>/v1
                parts = [p for p in base_url.split("/") if p]
                if "sessions" in parts:
                    sid = parts[parts.index("sessions") + 1]
            except Exception:
                pass
        agent_run = _timing(r, "agent_execution")
        # EXACT GPU vs container-CPU split of agent_run via the session server's gen_time.
        gpu_gen = round(gen_by_session[sid], 1) if sid in gen_by_session else None
        container_cpu = round(agent_run - gpu_gen, 1) if (agent_run is not None and gpu_gen is not None) else None
        rows.append({
            "st": st_,
            "rollout_id": rid,
            "name": os.path.basename(os.path.dirname(fp)),
            "reward": reward,
            "turns": _turns(os.path.join(os.path.dirname(fp), "agent", "trajectory.json")),
            "total": _dur(r.get("started_at"), r.get("finished_at")),
            "spawn": _timing(r, "environment_setup"),
            "install": _timing(r, "agent_setup"),
            "agent_run": agent_run,
            "verifier": _timing(r, "verifier"),
            "gpu_gen": gpu_gen,              # Σ gen_time for this trial's session (GPU)
            "container_cpu": container_cpu,  # agent_run − gpu_gen (container/tool CPU)
        })
    return rows


def build_trial_stats(rows: list) -> dict:
    cols = {"environment_setup": "spawn", "agent_setup": "install", "agent_execution": "agent_run",
            "verifier": "verifier", "total": "total"}
    acc = {k: [] for k in cols}
    acc["teardown_resid"] = []
    for t in rows:
        for phase, key in cols.items():
            v = t[key]
            if v is not None:
                acc[phase].append(v)
        if t["total"] is not None:
            known = sum(x for x in (t["spawn"], t["install"], t["agent_run"], t["verifier"]) if x is not None)
            acc["teardown_resid"].append(max(t["total"] - known, 0))
    out = {}
    for k, v in acc.items():
        if not v:
            continue
        v2 = sorted(v)
        out[k] = {"min": round(v2[0], 2), "p50": round(st.median(v2), 2),
                  "p90": round(v2[min(len(v2) - 1, int(0.9 * len(v2)))], 2),
                  "max": round(v2[-1], 2), "mean": round(st.mean(v2), 2), "n": len(v2)}
    return out


def build_per_step(rows: list, wins: dict) -> dict:
    """Group trials by rollout step.

    Prefer the native ``rollout_id`` sidecar (written by server.py); fall back to the
    wall-clock window [end - rollout_time, end] for trials from runs that predate the
    tagging. Offset (delay since rollout start) uses the step window when available.
    """
    per: dict[int, list] = {}
    items = sorted(wins.items())

    def record(n, t):
        s0 = wins[n][0] if n in wins else t["st"]
        per.setdefault(n, []).append({
            "name": t["name"], "reward": t["reward"], "turns": t["turns"], "total": t["total"],
            "spawn": t["spawn"], "install": t["install"], "agent_run": t["agent_run"],
            "verifier": t["verifier"], "gpu_gen": t.get("gpu_gen"), "container_cpu": t.get("container_cpu"),
            "offset": round(t["st"] - s0, 1),
        })

    for t in sorted(rows, key=lambda x: x["st"]):
        if t.get("rollout_id") is not None:          # native tag
            record(t["rollout_id"], t)
            continue
        for n, (s0, s1) in items:                    # wall-clock fallback
            if s0 <= t["st"] < s1:
                record(n, t)
                break
    return per


def main():
    default_work = os.environ.get(
        "WORK_DIR",
        os.path.join(os.environ.get("RUNTIME", "/mnt/disk_nvme1n1/sreerohi/qwen3-codecontests/runtime"), "work"),
    )
    ap = argparse.ArgumentParser(description="Build rollout_timeline.html data files")
    ap.add_argument("--work-dir", default=default_work, help="runtime/work dir (contains train.log and cc_trials/)")
    ap.add_argument("--train-log", default="", help="path to log file directly (overrides work-dir/train.log)")
    ap.add_argument("--trials-dir", default="", help="path to trials directory directly (overrides work-dir/cc_trials)")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="where to write the 4 JSON files (default: alongside this script / the HTML)")
    args = ap.parse_args()

    train_log = args.train_log or os.path.join(args.work_dir, "train.log")
    trials_dir = args.trials_dir or os.path.join(args.work_dir, "cc_trials")
    if not os.path.isfile(train_log):
        raise SystemExit(f"log file not found: {train_log}")

    print(f"parsing {train_log} ...")
    metrics = parse_step_metrics(train_log)
    wins = step_windows(train_log)
    print(f"  {len(metrics)} steps with rollout_time; {len(wins)} step windows")

    gen = parse_gen_times(train_log)
    print(f"  {len(gen)} sessions with gen_time (GPU generation)")
    print(f"parsing trials under {trials_dir} ...")
    rows = parse_trials(trials_dir, gen)
    tagged = sum(1 for r in rows if r.get("gpu_gen") is not None)
    print(f"  {len(rows)} trials ({tagged} with exact gpu_gen via session_id)")

    stats = build_trial_stats(rows)
    per = build_per_step(rows, wins)
    counts = {n: len(v) for n, v in sorted(per.items())}
    print(f"  per-step trial counts (first 8): {dict(list(counts.items())[:8])}")

    src = {
        "source": "runtime/work/train.log (identical to W&B)",
        "log_mtime": datetime.utcfromtimestamp(os.path.getmtime(train_log)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parsed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "steps": len(metrics),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    outputs = {
        "timeline_metrics.json": metrics,
        "per_step_trials.json": per,
        "trial_phase_stats.json": stats,
        "timeline_source.json": src,
    }
    for fn, obj in outputs.items():
        path = os.path.join(args.out_dir, fn)
        json.dump(obj, open(path, "w"))
        print(f"wrote {path}  ({os.path.getsize(path)} bytes)")

    # Export trajectory JSONs so the HTML viewer can load them on click.
    # Create symlinks (not copies) so the HTML viewer can fetch trajectories
    # without duplicating data on disk.
    traj_dir = os.path.join(args.out_dir, "traj")
    os.makedirs(traj_dir, exist_ok=True)
    exported = 0
    for row in rows:
        name = row.get("name", "")
        if not name:
            continue
        src_traj = os.path.join(trials_dir, name, "agent", "mini-swe-agent.trajectory.json")
        if not os.path.exists(src_traj):
            src_traj = os.path.join(trials_dir, name, "agent", "trajectory.json")
        if os.path.exists(src_traj):
            dst = os.path.join(traj_dir, f"{name}.json")
            if os.path.islink(dst):
                os.unlink(dst)  # refresh stale symlink
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src_traj), dst)
            exported += 1
    print(f"linked {exported} trajectory files in {traj_dir}/ (symlinks, no copy)")
    print("done — refresh rollout_timeline.html to see the update.")


if __name__ == "__main__":
    main()
