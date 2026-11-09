"""End-to-end grid runner.

For every named configuration it launches ``agent.py`` as a subprocess
with ``--max_workers=100``, so all cases within a config run truly in
parallel. The runner prints a side-by-side comparison table at the end
and writes a combined grid report.

Configuration spec format
-------------------------
Each config is a comma-separated list of ``key=value`` pairs; multiple
configs are separated by ``;``.

Supported keys (the first form is a short alias, the second is the
full agent CLI flag):

  h      max_history_steps
  i      lookback_image_count
  ref    reference_traj      (values: forward_only / all_actions / none)
  refimg reference_traj_images (values: none / all / interval / turn_only / smart)
  sl     reference_traj_smart_layers (comma-separated: start,stage,end,turn)
  sc     reference_traj_smart_sample_count (int, max sampled images)

Default configs (compared when ``--configs`` is not given)::

  h=0,i=0
  h=5,i=5
  h=5,i=5,ref=FWD
  h=5,i=5,ref=FWD,refimg=smart

Usage
-----
  python3 code/test_run.py                                 # default 4 configs × 100 cases
  python3 code/test_run.py --case_count 10 --max_workers 10 # smoke run
  python3 code/test_run.py --no_server                     # reuse a running server
  python3 code/test_run.py --configs "h=5,i=5;h=10,i=10,ref=FWD,refimg=smart"
  python3 code/test_run.py --configs "h=3,i=0,ref=FWD,refimg=smart,sl=start+end,sc=5"
"""
import os
import sys
import json
import time
import argparse
import subprocess
import requests

# Windows consoles default to the ANSI code page (GBK on this machine),
# which makes the emoji in the progress prints raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SERVER = os.path.join(PROJECT_ROOT, "code", "server.py")
DEFAULT_AGENT = os.path.join(PROJECT_ROOT, "code", "agent.py")
DEFAULT_PORT = 5005
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "result")

# Default paths for the memory-recall flags. Used by the bare `sum` /
# `sem` config tokens (e.g. `--configs "i=6,h=6,sum,sem"`); also serve as
# defaults if the user explicitly passes "" or omits the flag (feature
# off). Forwarded to agent.py, which resolves relative paths against
# its own PROJECT_ROOT (the repo root). To opt in, either pass a path
# via CLI / config value, or use the bare `sum` / `sem` tokens.
DEFAULT_SEMANTIC_PROMPT_PATH = "dataset_v1-v3_memory/v0/semantic_memory/memory.md"
DEFAULT_SUMMARY_PROMPT_PATH = "dataset_v1-v3_memory/v0/summary_memory/summary.md"

# (config spec, label). The label is just for display; the spec is parsed.
DEFAULT_CONFIGS = (
    'h=2,i=2',
)


SHORT_TO_LONG = {
    "h": "max_history_steps",
    "i": "lookback_image_count",
    "ref": "reference_traj",
    "refimg": "reference_traj_images",
    "sl": "reference_traj_smart_layers",
    "sc": "reference_traj_smart_sample_count",
    "peer": "peer_reference",
    "em": "episodic_memory",
    "em_h": "episodic_memory_horizon",
    "em_m": "episodic_memory_matcher",
    "em_c": "episodic_memory_content",
    "sum": "summary_prompt_path",
    "sem": "semantic_prompt_path",
    "max_history_steps": "max_history_steps",
    "lookback_image_count": "lookback_image_count",
    "reference_traj": "reference_traj",
    "reference_traj_images": "reference_traj_images",
    "reference_traj_smart_layers": "reference_traj_smart_layers",
    "reference_traj_smart_sample_count": "reference_traj_smart_sample_count",
    "peer_reference": "peer_reference",
    "episodic_memory": "episodic_memory",
    "episodic_memory_horizon": "episodic_memory_horizon",
    "episodic_memory_matcher": "episodic_memory_matcher",
    "episodic_memory_content": "episodic_memory_content",
    "summary_prompt_path": "summary_prompt_path",
    "semantic_prompt_path": "semantic_prompt_path",
}

# Allow abbreviated values for the choice-typed keys.
REF_TRAJ_ALIASES = {"FWD": "forward_only", "AA": "all_actions"}
EM_MATCHER_ALIASES = {"p": "pose", "v": "vit"}
EM_CONTENT_ALIASES = {"a": "all", "act": "actions", "img": "images"}


# ==========================================================================
# EXPERIMENT MODE — the v0-v3 -> v4 memory-transfer matrix
# ==========================================================================


def start_experiment_server(args, port, server_log_dir, log_name, config_path):
    """Spawn a server bound to the experiment's v4 test set."""
    os.makedirs(server_log_dir, exist_ok=True)
    log_path = os.path.join(server_log_dir, log_name)
    log_fp = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, args.server_script, "--port", str(port),
         "--log_dir", server_log_dir, "--experiment_config", config_path],
        stdout=log_fp, stderr=subprocess.STDOUT,
    )
    return proc, log_fp


def run_experiment_group(args, group_id, port, results_dir, config_path,
                         want_fingerprint=None):
    """Bring up a dedicated server, run one group against it, tear it down.

    ``want_fingerprint`` is the config fingerprint this group must talk to.
    Ports outlive runs: a leftover server from an earlier experiment holds
    the port, the new one fails to bind, and without this check the agent
    would hand its cases to the wrong configuration.
    """
    server_log_dir = os.path.join(results_dir, "server_logs", group_id)
    client_logs_dir = os.path.join(results_dir, "client_logs")
    os.makedirs(client_logs_dir, exist_ok=True)
    run_name = f"{args.experiment_id_hint}_{group_id}"

    proc = log_fp = None
    url = None
    for attempt in range(args.server_start_retries):
        p = port + attempt * 100
        proc, log_fp = start_experiment_server(
            args, p, server_log_dir, f"server_{group_id}.log", config_path)
        url = f"http://127.0.0.1:{p}"
        info = wait_server(url, timeout=90)
        if info is not None:
            exp = info.get("experiment") or {}
            if not exp:
                stop_server(proc, log_fp)
                return group_id, {"error": "server started without experiment mode"}, 0.0
            got = exp.get("config_fingerprint", "")
            if want_fingerprint and got != want_fingerprint:
                # Someone else already owns this port. Our own server could
                # not bind and the agent would silently talk to theirs.
                print(f"   [{group_id}] port {p} is held by a different "
                      f"experiment ({exp.get('experiment_id')} "
                      f"{got[:12]}); trying the next port")
                stop_server(proc, log_fp)
                proc = log_fp = url = None
                continue
            print(f"   [{group_id}] server up on {p} "
                  f"(fingerprint {got[:12]})")
            break
        stop_server(proc, log_fp)
        proc = log_fp = url = None
    if url is None:
        return group_id, {"error": "server failed to start"}, 0.0

    cmd = [
        sys.executable, args.agent_script,
        "--server_url", url,
        "--experiment_config", config_path,
        "--run_name", run_name,
        "--results_dir", results_dir,
        "--round_id", str(args.round_id),
    ]
    if args.case_ids:
        cmd += ["--case_ids", args.case_ids]
    if args.api_excel:
        cmd += ["--api_excel", args.api_excel]
    if args.max_workers:
        cmd += ["--max_workers", str(args.max_workers)]
    if args.save_prompts:
        cmd += ["--save_prompts"]
    if args.revalidate_keys:
        cmd += ["--revalidate_keys"]
    if args.resume_dir:
        cmd += ["--resume"]
    if getattr(args, "resume_steps", False):
        cmd += ["--resume_steps"]

    client_log_path = os.path.join(client_logs_dir, f"client_{run_name}.log")
    t0 = time.time()
    with open(client_log_path, "w", encoding="utf-8") as client_log_fp:
        try:
            subprocess.run(cmd, stdout=client_log_fp, stderr=subprocess.STDOUT,
                           check=False)
        finally:
            stop_server(proc, log_fp)
    elapsed = time.time() - t0

    summary_path = os.path.join(results_dir, f"client_{run_name}", "SUMMARY.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {"error": "no SUMMARY.json produced",
                   "client_log": client_log_path}
    return group_id, summary, elapsed


def run_experiment(args):
    """Run one config: preflight, then its group against a dedicated server."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
    import experiment as ex

    config_path = ex.resolve_config_path(args.experiment_config)
    cfg = ex.load_config(config_path)
    args.experiment_id_hint = cfg.experiment_id
    gid = cfg.group_id

    if args.resume_dir:
        results_dir = os.path.abspath(args.resume_dir)
        if not os.path.isdir(results_dir):
            raise SystemExit("--resume_dir does not exist: %s" % results_dir)
        print("RESUMING into %s (settled cases will be skipped)" % results_dir)
    else:
        from result_paths import experiment_result_dir
        results_dir = experiment_result_dir(
            cfg, config_path, round_id=args.round_id, root=args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    _acquire_run_lock(results_dir)
    _write_run_info(cfg, config_path, results_dir, args)

    report = preflight(cfg, config_path, results_dir)
    if args.preflight_only:
        print("\nPreflight only — no model calls were made.")
        return report

    print()
    print("=" * 78)
    print("GROUP %s: %s" % (gid, cfg.group.slug))
    print("=" * 78)
    try:
        _, summary, elapsed = run_experiment_group(
            args, gid, args.port, results_dir, config_path, cfg.fingerprint())
    except Exception as e:
        summary, elapsed = {"error": str(e)}, 0.0
    results = {gid: {"summary": summary, "elapsed": elapsed}}
    _print_group_line(gid, summary, elapsed)

    grid = {
        "experiment_id": cfg.experiment_id,
        "config_fingerprint": cfg.fingerprint(),
        "config_path": config_path,
        "description": cfg.raw.get("description", ""),
        "model": dict(cfg.model),
        "test_version": cfg.test_version,
        "case_ids": args.case_ids or "all 100",
        "round_id": args.round_id,
        "results_dir": results_dir,
        "preflight": report,
        "groups": {gid: results[gid]["summary"] for gid in results},
        "elapsed_by_group": {gid: results[gid]["elapsed"] for gid in results},
    }
    grid_path = os.path.join(results_dir, f"grid_{cfg.experiment_id}_r{args.round_id}.json")
    with open(grid_path, "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2, ensure_ascii=False)

    _print_matrix(results)
    print(f"\nGrid summary: {grid_path}")
    return grid


def _acquire_run_lock(results_dir):
    """Refuse to start when another runner already owns this directory.

    Two runners sharing one results directory interleave their writes:
    both append to the same steps_*.jsonl and both write the same
    case_*.json, so a case they both touch ends up holding a mix of two
    independent executions that cannot be separated afterwards. Cheap to
    prevent, impossible to repair.

    Released via atexit; a lock whose owning process is gone is taken over.
    """
    import atexit
    import errno

    path = os.path.join(results_dir, "RUN.lock")
    payload = json.dumps({"pid": os.getpid(),
                          "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "command": " ".join(sys.argv)}, ensure_ascii=False)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        try:
            with open(path, encoding="utf-8") as f:
                holder = json.load(f)
        except Exception:
            holder = {}
        if _pid_alive(holder.get("pid")):
            msg = """REFUSING TO START: another runner already owns this results directory.
  pid     : {pid}
  started : {started}
  command : {cmd}

Two runners over one directory corrupt every case they both touch: their
step logs interleave and their case results overwrite each other.
Stop that process first, or start a fresh run without --resume_dir.
If you are certain it is dead, delete {path}"""
            raise SystemExit(msg.format(pid=holder.get("pid"),
                                        started=holder.get("started"),
                                        cmd=holder.get("command", "?"),
                                        path=path))
        print("[lock] taking over a stale lock from dead pid %s" % holder.get("pid"))
        os.unlink(path)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)
    atexit.register(_release_run_lock, path)
    return path


def _pid_alive(pid):
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %s" % pid],
                             capture_output=True, text=True, timeout=15).stdout
        return str(pid) in out
    except Exception:
        return True          # cannot tell -> assume alive, fail safe


def _release_run_lock(path):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception:
        pass


def _write_run_info(cfg, config_path, results_dir, args):
    """Identify the run before anything is spent on it.

    Written first so that even an aborted run leaves a record of which
    experiment, model and code revision produced whatever is in the
    directory.
    """
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git"] + list(a), cwd=PROJECT_ROOT,
                                  capture_output=True, text=True,
                                  timeout=15).stdout.strip()
        except Exception:
            return ""

    info = {
        "experiment_id": cfg.experiment_id,
        "config_fingerprint": cfg.fingerprint(),
        "config_path": config_path,
        "description": cfg.raw.get("description", ""),
        "test_version": cfg.test_version,
        "round_id": args.round_id,
        "group_id": cfg.group_id,
        "case_ids": args.case_ids or "all 100",
        "max_steps": cfg.max_steps,
        "working_memory": {"history_steps": cfg.history_steps,
                           "history_images": cfg.history_images},
        "model": dict(cfg.model),
        "save_prompts": bool(cfg.execution.get("save_prompts")) or bool(args.save_prompts),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_dir": results_dir,
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    path = os.path.join(results_dir, "RUN_INFO.json")
    if os.path.exists(path):
        # Resuming into a directory that already holds finished groups:
        # keep the previous manifest instead of overwriting it, or the
        # record of how those groups were run is lost.
        try:
            with open(path, encoding="utf-8") as f:
                prev = json.load(f)
            stamp = (prev.get("started_at") or "").replace(":", "").replace(" ", "_")                 or time.strftime("%Y%m%d_%H%M%S")
            archived = os.path.join(results_dir, "RUN_INFO_%s.json" % stamp)
            if not os.path.exists(archived):
                os.rename(path, archived)
                print("previous manifest kept as %s" % os.path.basename(archived))
            info["previous_runs"] = [os.path.basename(archived)] +                 (prev.get("previous_runs") or [])
        except Exception as e:
            print("could not archive the previous RUN_INFO.json: %s" % e)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("=" * 78)
    print("EXPERIMENT : %s" % cfg.experiment_id)
    print("FINGERPRINT: %s" % cfg.fingerprint())
    print("MODEL      : %s @ %s (provider=%s, max_tokens=%s)"
          % (cfg.model.get("name"), cfg.model.get("api_url"),
             cfg.model.get("provider", "internai"), cfg.model.get("max_tokens")))
    print("GROUP      : %s   CASES: %s   ROUND: %s"
          % (cfg.group_id, info["case_ids"], args.round_id))
    print("OUTPUT     : %s" % results_dir)
    print("=" * 78)
    return info


def _print_group_line(group_id, summary, elapsed):
    if "error" in summary:
        print(f"   [{group_id}] ERROR: {summary['error']}  ({elapsed:.1f}s)")
        return
    print(f"   [{group_id}] resolved {summary.get('cases_resolved')}/"
          f"{summary.get('cases_expected')} | success {summary.get('success')} | "
          f"rate {summary.get('success_rate')} | {elapsed:.1f}s")


def _print_matrix(results):
    print()
    print("=" * 100)
    print(f"{'GROUP':<6} {'METHOD':<26} {'RESOLVED':>9} {'SUCCESS':>8} {'RATE':>12} "
          f"{'IN-TOK':>12} {'STEPS(OK)':>10} {'TIME(s)':>9}")
    print("-" * 100)
    for gid in sorted(results):
        s = results[gid]["summary"]
        if "error" in s:
            print(f"{gid:<6} {'-':<26} {'ERROR: ' + str(s['error'])[:60]}")
            continue
        grp = s.get("group") or {}
        tok = (s.get("token_usage_settled") or {}).get("prompt_tokens", 0)
        steps = s.get("steps_median_success")
        print(f"{gid:<6} {str(grp.get('slug') or grp.get('label'))[:26]:<26} "
              f"{s.get('cases_resolved')}/{s.get('cases_expected'):<6} "
              f"{s.get('success'):>8} {str(s.get('success_rate')):>12} "
              f"{tok:>12} {('-' if steps is None else '%.1f' % steps):>10} "
              f"{results[gid]['elapsed']:>9.1f}")
    print("=" * 100)
    incomplete = [g for g in sorted(results)
                  if not (results[g]["summary"].get("complete", False))]
    if incomplete:
        print("Groups with unresolved cases (complete-set success rate NOT "
              "published): %s" % ", ".join(incomplete))


def preflight(cfg, config_path, results_dir):
    """Offline checks. Loads every input and builds every group's request
    shape without contacting the model."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
    import experiment as ex

    print("=" * 78)
    print(f"PREFLIGHT  {cfg.experiment_id}  fingerprint={cfg.fingerprint()[:16]}")
    print("=" * 78)
    report = {"experiment_id": cfg.experiment_id,
              "config_fingerprint": cfg.fingerprint(),
              "config_path": config_path,
              "group_id": cfg.group_id,
              "errors": [], "warnings": []}

    tasks, task_report = ex.load_test_tasks(cfg)
    report["test_tasks"] = {"count": len(tasks),
                            "case_map_sha256": cfg.case_map.source_sha256}
    print(f"[ok] {cfg.test_version} test tasks: {len(tasks)} cases")

    legacy = ex.resolve_path(cfg.raw["scene"].get("legacy_tasks", ""))
    if legacy and os.path.exists(legacy):
        if cfg.test_version == "v0":
            # Testing on v0 is testing on the legacy task set itself, so the
            # invariant is full identity -- origin included.
            inv = ex.assert_v0_matches_legacy_tasks(cfg, tasks, legacy)
            report["v0_vs_legacy_invariant"] = inv
            print(f"[ok] v0 test tasks == legacy tasks.json "
                  f"({inv['cases']} cases: task, goals and origin identical)")
        else:
            inv = ex.assert_v4_matches_v0_tasks(cfg, tasks, legacy)
            report["v4_vs_v0_invariant"] = inv
            print(f"[ok] {cfg.test_version} == v0 tasks except origin "
                  f"({inv['goals_identical']} identical goals, "
                  f"{len(inv['return_to_origin_coord_updated'])} return-to-origin coords updated)")

    names = ex.scene_base64_names(cfg)
    report["scene"] = {"base64_files": len(names)}
    print(f"[ok] scene base64 frames: {len(names)}")

    # Per-case scenes: one for a v4 run, up to 38 for v5. Each needs its own
    # base64 cache and ViT table, so check them before the run rather than
    # discovering a hole mid-case.
    scenes = {}
    for t in tasks:
        scenes.setdefault(t.scene.scene_id, t.scene)
    scene_report, scene_problems = {}, []
    for sid, scene in sorted(scenes.items()):
        n_b64 = len(ex.scene_base64_names(cfg, scene))
        scene_report[sid] = {"base64": n_b64, "embeddings": scene.has_embeddings()}
        scene_problems += ["%s: %s" % (sid, p)
                           for p in ex.scene_index.validate(scene)]
    report["scenes"] = scene_report
    if scene_problems:
        raise SystemExit("[FAIL] scene layout problems:\n  "
                         + "\n  ".join(scene_problems))
    n_emb = sum(1 for v in scene_report.values() if v["embeddings"])
    print(f"[ok] scenes: {len(scene_report)} "
          f"({n_emb} with a ViT table, "
          f"{sum(v['base64'] for v in scene_report.values())} base64 frames total)")

    versions_needed = sorted(cfg.group.versions)
    traj_stats, anomalies = {}, []
    if versions_needed:
        for key in cfg.case_map.all_keys():
            for v in versions_needed:
                t = ex.load_history_trajectory(cfg, key, v, base64_names=names)
                st = traj_stats.setdefault(v, {"count": 0, "actions": 0})
                st["count"] += 1
                st["actions"] += t.length
                if t.validation_notes:
                    anomalies.append({"case": key.label(), "version": v,
                                      "notes": t.validation_notes})
        report["history"] = traj_stats
        report["known_anomalies"] = anomalies
        for v in sorted(traj_stats):
            print(f"[ok] {v}: {traj_stats[v]['count']} trajectories, "
                  f"{traj_stats[v]['actions']} forward_only actions")
        for a in anomalies:
            print(f"[note] known anomaly {a['case']} {a['version']}: {a['notes'][0][:70]}...")

    report["memory_profile"] = ex.memory_profile(cfg)
    print("[ok] memory profile:")
    for row in report["memory_profile"]:
        print(f"       {row['kind']:<9} {row['version']}  {row['bytes']:>7} bytes  "
              f"{row['chars']:>7} chars  max Case {row['max_case_mentioned']}")

    report["image_budget"] = _preflight_image_budget(cfg, names)
    for gid, row in sorted(report["image_budget"].items()):
        print(f"[ok] {gid} reference images/case: min={row['min']} "
              f"median={row['median']} max={row['max']} "
              f"(cap binds on {row['cap_binds']}/{row['cases']} cases)")

    report["request_shapes"] = _preflight_request_shapes(cfg, tasks)
    for gid, row in sorted(report["request_shapes"].items()):
        if "error" in row:
            report["errors"].append(f"{gid}: {row['error']}")
            print(f"[FAIL] {gid}: {row['error']}")
        else:
            print(f"[ok] {gid} first-decision request: {row['images']} images, "
                  f"{row['text_chars']} text chars, "
                  f"non-memory sha {row['non_memory_sha'][:12]}")

    path = os.path.join(results_dir, "preflight.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nPreflight report: {path}")
    if report["errors"]:
        raise SystemExit("preflight failed with %d error(s)" % len(report["errors"]))
    return report


def _preflight_image_budget(cfg, names):
    """How many reference images a Full Context group really sends."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
    import experiment as ex
    from prompt.reference_traj_images import _select_smart_indices

    g = cfg.group
    if g.memory != "full_context":
        return {}
    per_case, cap_binds = [], 0
    for key in cfg.case_map.all_keys():
        total, bound = 0, False
        remaining = min(g.max_reference_images, cfg.max_prompt_images - 1 - cfg.history_images)
        for v in g.versions:
            t = ex.load_history_trajectory(cfg, key, v, base64_names=names)
            budget = max(0, min(g.max_smart_images_per_traj, remaining))
            picked = _select_smart_indices(t.actions, t.stage_events, budget)
            if len(picked) == budget and (t.length + 1) > budget:
                bound = True
            total += len(picked)
            remaining -= len(picked)
        per_case.append(total)
        cap_binds += 1 if bound else 0
    per_case.sort()
    return {g.group_id: {"cases": len(per_case), "min": per_case[0],
                         "median": per_case[len(per_case) // 2], "max": per_case[-1],
                         "cap_binds": cap_binds,
                         "max_allowed": g.max_reference_images}}


def _preflight_request_shapes(cfg, tasks):
    """Build the group's first-decision request offline and record its shape:
    image count, text size and a digest of everything outside memory (equal
    digests mean two configs send the same non-memory request)."""
    import hashlib
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
    import agent as ag
    import experiment as ex

    names = ex.scene_base64_names(cfg)
    key = cfg.case_map.by_legacy(0)
    task = tasks[0]
    origin = {"x": int(task.origin["coord"][0]), "y": int(task.origin["coord"][1]),
              "heading": task.origin["heading"]}
    img = task.scene.image_path(origin["x"], origin["y"], origin["heading"])

    g = cfg.group
    ag.EXPERIMENT, ag.EXPERIMENT_GROUP = cfg, g
    ag.MAX_HISTORY_STEPS = cfg.history_steps
    ag.LOOKBACK_IMAGE_COUNT = cfg.history_images
    ag.MAX_PROMPT_IMAGES = cfg.max_prompt_images
    ag.ENABLE_HISTORY = True
    ag.EXPERIMENT_MEMORY_TEXT = ""
    try:
        trajs = []
        if g.uses_text_memory:
            ag.prepare_experiment_text_memory(cfg, g)
        if g.uses_history_trajectories:
            loaded = ex.load_history_set(cfg, key, g.versions, base64_names=names)
            trajs = [t.as_prompt_traj() for t in loaded]
        ctx = ag.DecisionContext(
            case_label=(key.task_id, key.case_no),
            initial_origin=dict(origin), current_pose=dict(origin),
            decision_index=1, stages_completed=0, steps_in_current_stage=0)
        telemetry = {}
        parts = ag.build_experiment_user_content(
            task.description, origin, origin, [], img, [], trajs, ctx, telemetry)
    except Exception as e:
        return {g.group_id: {"error": "%s: %s" % (type(e).__name__, e)}}
    finally:
        ag.EXPERIMENT = None
        ag.EXPERIMENT_GROUP = None
        ag.EXPERIMENT_MEMORY_TEXT = ""

    images = sum(1 for b in parts if b.get("type") == "image_url")
    text_chars = sum(len(b.get("text", "")) for b in parts if b.get("type") == "text")
    # Strip every memory section, then hash what is left. Two configs that
    # differ anywhere outside their memory sections will not produce the
    # same digest.
    digest = hashlib.sha256(
        json.dumps(_strip_memory_sections(parts), sort_keys=True,
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {g.group_id: {"images": images, "text_chars": text_chars,
                         "non_memory_sha": digest, "telemetry": telemetry}}


def _strip_memory_sections(parts):
    """Keep every block that is NOT part of a long-term-memory section.

    ``build_experiment_user_content`` tags memory blocks with
    ``_section="memory"``, so this is an exact split rather than a guess
    at where the memory text starts and ends. What remains - mission,
    current view, history, closing reminder - must be byte-identical
    across all nine groups for the same case and step.
    """
    keep = []
    for b in parts:
        if b.get("_section") == "memory":
            continue
        if b.get("type") == "image_url":
            keep.append({"image": b.get("_image_path")})
        else:
            keep.append({"text": b.get("text", "")})
    return keep


def wait_server(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/get_server_time", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(0.5)
    return None


def start_server(args, port, server_log_dir, log_name):
    """Spawn a fresh server.py subprocess on the given port.

    Returns (proc, log_file_handle). Caller is responsible for stopping
    the server and closing the log handle.

    The server's per-case log dir is redirected into ``server_log_dir``
    so the entire test run lives under one test_run_<ts>/ subtree.
    """
    os.makedirs(server_log_dir, exist_ok=True)
    log_path = os.path.join(server_log_dir, log_name)
    log_fp = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, args.server_script, "--port", str(port),
         "--log_dir", server_log_dir],
        stdout=log_fp, stderr=subprocess.STDOUT,
    )
    return proc, log_fp


def stop_server(proc, log_fp):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    if log_fp and not log_fp.closed:
        log_fp.close()


def parse_config(config_str):
    """Parse 'h=5,i=5,ref=FWD,refimg=smart,peer' into (label, args_list, errors)."""
    label_parts = []
    args = []
    errors = []
    for raw in config_str.split(","):
        kv = raw.strip()
        if not kv:
            continue
        # Handle boolean flags (peer_reference, episodic_memory) that can be
        # passed as bare keys or as key=true/false.
        if "=" not in kv:
            long_key = SHORT_TO_LONG.get(kv)
            if long_key in ("peer_reference", "episodic_memory"):
                args.append(f"--{long_key}")
                label_parts.append(kv)
                continue
            # Bare `sum` / `sem`: enable the corresponding memory-recall
            # flag with its conventional default path. Requires a non-empty
            # DEFAULT_* at module load.
            if long_key in ("summary_prompt_path", "semantic_prompt_path"):
                default_path = (
                    DEFAULT_SUMMARY_PROMPT_PATH
                    if long_key == "summary_prompt_path"
                    else DEFAULT_SEMANTIC_PROMPT_PATH
                )
                if not default_path:
                    errors.append(
                        f"bare '{kv}' requires a default path to be configured "
                        f"(set DEFAULT_{long_key.upper()} in test_run.py)"
                    )
                    continue
                args += [f"--{long_key}", default_path]
                label_parts.append(kv)
                continue
            errors.append(f"invalid token '{kv}' (expected key=value)")
            continue
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        long_key = SHORT_TO_LONG.get(k)
        if long_key is None:
            errors.append(f"unknown key '{k}'")
            continue
        # Translate short-form values for choice-typed keys
        if long_key == "reference_traj" and v in REF_TRAJ_ALIASES:
            v = REF_TRAJ_ALIASES[v]
        if long_key == "episodic_memory_matcher":
            key = v.lower()
            v = EM_MATCHER_ALIASES.get(key, v)
            if v not in ("pose", "vit"):
                errors.append(f"--episodic_memory_matcher must be pose|vit (got '{v}')")
                continue
        if long_key == "episodic_memory_content":
            key = v.lower()
            v = EM_CONTENT_ALIASES.get(key, v)
            if v not in ("all", "actions", "images"):
                errors.append(f"--episodic_memory_content must be all|actions|images (got '{v}')")
                continue
        # For smart_layers, allow '+' as separator since ',' is the config delimiter
        if long_key == "reference_traj_smart_layers":
            v = v.replace("+", ",")
        # Handle boolean flags with explicit value
        if long_key in ("peer_reference", "episodic_memory"):
            if v.lower() in ("true", "1", "yes", "on"):
                args.append(f"--{long_key}")
                label_parts.append(k)
            # If false/0/no/off, don't add the flag
            continue
        args += [f"--{long_key}", v]
        label_parts.append(f"{k}={v}")
    return ", ".join(label_parts), args, errors


def run_one_config(spec, args, server_url, case_count, max_workers, results_dir,
                   start_idx):
    label, cfg_args, errors = parse_config(spec)
    if errors:
        return label, {"error": "; ".join(errors)}, 0.0, ""

    run_name = f"cfg{start_idx:02d}_{time.time_ns()}"
    cmd = [
        sys.executable, args.agent_script,
        "--server_url", server_url,
        "--case_count", str(case_count),
        "--max_workers", str(max_workers),
        "--run_name", run_name,
        "--results_dir", results_dir,
    ] + cfg_args
    if args.save_prompts:
        cmd += ["--save_prompts"]
    # Only add global memory paths if not already specified in config
    if args.summary_prompt_path and "--summary_prompt_path" not in cfg_args:
        cmd += ["--summary_prompt_path", args.summary_prompt_path]
    if args.semantic_prompt_path and "--semantic_prompt_path" not in cfg_args:
        cmd += ["--semantic_prompt_path", args.semantic_prompt_path]
    cmd += ["--max_api_retries", str(args.max_api_retries),
            "--api_timeout", str(args.api_timeout)]

    print()
    print("━" * 78)
    print(f"🏃 Config #{start_idx}: {label}")
    print(f"   cmd: {' '.join(cmd[2:])[:140]}{'...' if len(' '.join(cmd[2:])) > 140 else ''}")
    print(f"   results_dir arg passed to agent: {results_dir}")
    t0 = time.time()
    # Persist the agent subprocess's stdout/stderr alongside the server
    # logs so the full test trace lives inside the test-run folder.
    client_logs_dir = os.path.join(results_dir, "client_logs")
    os.makedirs(client_logs_dir, exist_ok=True)
    client_log_path = os.path.join(client_logs_dir, f"client_{run_name}.log")
    client_log_fp = open(client_log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.run(cmd, stdout=client_log_fp, stderr=subprocess.STDOUT, text=True)
    finally:
        client_log_fp.close()
    elapsed = time.time() - t0

    summary_path = os.path.join(results_dir, f"client_{run_name}", "SUMMARY.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        ok = summary.get("success", 0)
        run = summary.get("cases_run", 0)
        rate = summary.get("success_rate", "?")
        tokens = summary.get("token_usage", {})
        tokens_success = summary.get("token_usage_success_only", {})
        total_tok = tokens.get("total_tokens", 0)
        total_tok_success = tokens_success.get("total_tokens", 0)
        print(f"   ⏱️  {elapsed:6.1f}s  ✅ {ok}/{run} ({rate})  🔤 {total_tok:,} tokens (success: {total_tok_success:,})")
    else:
        # When stdout/stderr is redirected to a file, proc.stdout / proc.stderr
        # are None — fall back to empty strings and rely on the per-config
        # client_logs/<run>.log for the full output.
        stdout_tail = (proc.stdout or "")[-400:]
        stderr_tail = (proc.stderr or "")[-400:]
        summary = {"error": "no SUMMARY.json produced",
                   "stdout_tail": stdout_tail,
                   "stderr_tail": stderr_tail}
        print(f"   ⏱️  {elapsed:6.1f}s  ❌ {summary['error']}")
        if stdout_tail.strip():
            print(f"   --- stdout (last 400) ---\n{stdout_tail}")
        if stderr_tail.strip():
            print(f"   --- stderr (last 400) ---\n{stderr_tail}")
    return label, summary, elapsed, run_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_script", default=DEFAULT_SERVER)
    parser.add_argument("--agent_script", default=DEFAULT_AGENT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--case_count", type=int, default=100,
                        help="number of cases to run per config (default: 100)")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="parallel cases inside agent.py. Legacy mode "
                             "defaults to 100; experiment mode defaults to the "
                             "config's execution.max_workers (8).")
    parser.add_argument("--save_prompts", action="store_true",
                        help="experiment mode: force-enable saving the first-decision "
                             "request (the config's execution.save_prompts decides "
                             "otherwise). Legacy mode: persist every step's prompt + "
                             "response under result/<run>/prompts/")
    parser.add_argument("--max_api_retries", type=int, default=100,
                        help="forward --max_api_retries to agent.py (default 100)")
    parser.add_argument("--api_timeout", type=int, default=300,
                        help="forward --api_timeout to agent.py (default 300s, 5 min)")
    parser.add_argument("--summary_prompt_path",
                        default=DEFAULT_SUMMARY_PROMPT_PATH,
                        help="forward --summary_prompt_path to agent.py. Default: "
                             "the conventional dataset path (feature ON if the .md "
                             "exists). Pass an explicit path (absolute or relative "
                             "to agent.py's PROJECT_ROOT) to override, or '' to "
                             "disable. Independent of --semantic_prompt_path.")
    parser.add_argument("--semantic_prompt_path",
                        default=DEFAULT_SEMANTIC_PROMPT_PATH,
                        help="forward --semantic_prompt_path to agent.py. Default: "
                             "the conventional dataset path (feature ON if the .md "
                             "exists). Pass an explicit path (absolute or relative "
                             "to agent.py's PROJECT_ROOT) to override, or '' to "
                             "disable. Independent of --summary_prompt_path.")
    parser.add_argument("--configs", type=str, default=None,
                        help="semicolon-separated config specs, e.g. "
                             "'h=5,i=5;h=5,i=5,ref=FWD,refimg=smart'")
    parser.add_argument("--parallel", action="store_true",
                        help="run all configs concurrently via ThreadPoolExecutor "
                             "(each config still gets its own agent.py subprocess; "
                             "they share the same server)")
    parser.add_argument("--parallel_max", type=int, default=4,
                        help="max parallel configs when --parallel is set (default 4)")
    parser.add_argument("--per_config_server", action="store_true",
                        help="start a separate server per config on a different port "
                             "(avoids one server bottleneck when 4 configs × 100 cases = "
                             "400 parallel agents)")
    parser.add_argument("--base_port", type=int, default=5005,
                        help="first port for per-config servers; config N uses "
                             "--base_port + (N-1) (default 5000)")
    parser.add_argument("--server_start_retries", type=int, default=10,
                        help="max attempts to start a per-config server before "
                             "giving up; each retry adds +100 to the port "
                             "(default 10)")
    parser.add_argument("--no_server", action="store_true",
                        help="don't start the server, assume it's already running")
    parser.add_argument("--results_dir", default=None,
                        help="output root override; experiment default: "
                             "result_<model>/<config stem>/r<round>_<timestamp>/; "
                             "legacy default: result/test_run_<start_ts>/")
    parser.add_argument("--server_log_dir", default=None,
                        help="override the server_logs/ subdir location (defaults "
                             "inside the test-run folder)")

    # ---- experiment mode (v0-v3 -> v4 study) -------------------------
    parser.add_argument("--experiment_config", default=None,
                        help="path or name of a config/**/*.yaml. Switches to experiment "
                             "mode: one dedicated v4 server + agent per group.")
    parser.add_argument("--case_ids", default=None,
                        help="comma-separated legacy case ids (default: all 100)")
    parser.add_argument("--round_id", type=int, default=0,
                        help="repeat index recorded in every result row")
    parser.add_argument("--preflight_only", action="store_true",
                        help="run the offline data/prompt checks and stop; makes "
                             "no model calls")
    parser.add_argument("--api_excel", default=None,
                        help="API key spreadsheet forwarded to agent.py "
                             "(default: <repo>/api.xlsx)")
    parser.add_argument("--revalidate_keys", action="store_true",
                        help="re-probe the API key pool instead of reusing the "
                             "cached validated subset")
    parser.add_argument("--resume_dir", default=None,
                        help="continue a previous run: write into that "
                             "directory and skip cases already settled there")
    parser.add_argument("--resume_steps", action="store_true",
                        help="restore interrupted case actions without repeating model calls")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment_config:
        # Compare against each flag's default, not truthiness: the two
        # memory-path flags default to non-empty legacy paths.
        conflicting = [name for name, val, default in
                       (("--configs", args.configs, None),
                        ("--summary_prompt_path", args.summary_prompt_path,
                         DEFAULT_SUMMARY_PROMPT_PATH),
                        ("--semantic_prompt_path", args.semantic_prompt_path,
                         DEFAULT_SEMANTIC_PROMPT_PATH))
                       if val != default]
        if conflicting:
            raise SystemExit(
                "these flags conflict with --experiment_config: %s\n"
                "In experiment mode the group matrix is the only configuration "
                "source." % ", ".join(conflicting))
        run_experiment(args)
        return
    if args.preflight_only:
        raise SystemExit("--preflight_only requires --experiment_config")

    if args.max_workers is None:
        args.max_workers = 100
    configs_str = args.configs or ";".join(DEFAULT_CONFIGS)
    configs = [c.strip() for c in configs_str.split(";") if c.strip()]

    # Bundle all outputs (client_<run>/, server_logs/, grid_<ts>.json) under
    # a single timestamped folder so a test run is self-contained.
    test_start_ts = time.strftime("%Y%m%d_%H%M%S")
    if args.results_dir is None:
        args.results_dir = os.path.join(DEFAULT_RESULTS_DIR, f"test_run_{test_start_ts}")
    if args.server_log_dir is None:
        args.server_log_dir = os.path.join(args.results_dir, "server_logs")
    print(f"📦 All outputs → {args.results_dir}")
    os.makedirs(args.results_dir, exist_ok=True)

    server_url = f"http://127.0.0.1:{args.port}"
    server_proc = None
    server_log_fp = None

    if not args.no_server and not args.per_config_server:
        os.makedirs(args.server_log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        server_log = os.path.join(args.server_log_dir, f"server_{ts}.log")
        print(f"📡 Starting server: {args.server_script}")
        print(f"   server log: {server_log}")
        # Server's per-case log dir is also routed into the test-run folder
        # (instead of result/evaluation_logs_<ts>/) via --log_dir.
        server_eval_dir = os.path.join(args.server_log_dir, "evaluation_logs")
        server_log_fp = open(server_log, "w", encoding="utf-8")
        server_proc = subprocess.Popen(
            [sys.executable, args.server_script,
             "--port", str(args.port),
             "--log_dir", server_eval_dir],
            stdout=server_log_fp, stderr=subprocess.STDOUT,
        )

    try:
        case_count = args.case_count
        if args.per_config_server:
            # Each config gets its own server on a different port so 4
            # configs × 100 cases don't bottleneck on one Flask process.
            print(f"   (Outputs → {args.results_dir})")
            print(f"📋 {len(configs)} config(s) will each get its own server "
                  f"(ports {args.base_port}..{args.base_port + len(configs) - 1})")

            def _run_one_with_server(i, spec):
                base = args.base_port + i - 1
                max_attempts = args.server_start_retries
                last_error = None
                for attempt in range(max_attempts):
                    port = base + attempt * 100
                    url = f"http://127.0.0.1:{port}"
                    log_name = f"server_port{port}_{time.strftime('%Y%m%d_%H%M%S')}.log"
                    proc, log_fp = start_server(args, port, args.server_log_dir, log_name)
                    try:
                        print(f"\n⏳ Waiting for server on port {port} "
                              f"(attempt {attempt + 1}/{max_attempts}) ...")
                        cfg = wait_server(url, timeout=60)
                        if cfg is None:
                            print(f"❌ Server on port {port} did not respond "
                                  f"(attempt {attempt + 1}/{max_attempts}).")
                            last_error = {"index": i, "config": spec,
                                          "error": f"server on :{port} not ready",
                                          "port": port, "elapsed_seconds": 0.0}
                            continue
                        print(f"✅ Server on port {port} ready "
                              f"(total_tasks={cfg.get('total_tasks', 0)})")
                        label, summary, elapsed, run_name = run_one_config(
                            spec, args, url, min(args.case_count, cfg.get("total_tasks", 0)),
                            args.max_workers, args.results_dir, i,
                        )
                        return {
                            "index": i,
                            "config": label,
                            "spec": spec,
                            "run_name": run_name,
                            "port": port,
                            "elapsed_seconds": round(elapsed, 2),
                            **{k: v for k, v in summary.items() if k != "hyperparams"},
                        }
                    finally:
                        print(f"🛑 Stopping server on port {port}")
                        stop_server(proc, log_fp)
                print(f"❌ All {max_attempts} attempts failed for config {i}. Skipping.")
                return last_error

            all_results = []
            if args.parallel:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                n_workers = min(args.parallel_max, len(configs))
                print(f"🧵 Running {len(configs)} per-config servers + configs in parallel "
                      f"(max {n_workers} workers)")
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futs = {pool.submit(_run_one_with_server, i, spec): (i, spec)
                            for i, spec in enumerate(configs, 1)}
                    for fut in as_completed(futs):
                        all_results.append(fut.result())
            else:
                for i, spec in enumerate(configs, 1):
                    all_results.append(_run_one_with_server(i, spec))
        else:
            print("⏳ Waiting for server...")
            cfg = wait_server(server_url, timeout=60)
            if cfg is None:
                print("❌ Server did not respond in time. Aborting.")
                return 1
            total = cfg.get("total_tasks", 0)
            case_count = min(args.case_count, total)
            print(f"✅ Server ready. total_tasks={total}, case_count={case_count}, "
                  f"max_workers={args.max_workers}")
            print(f"📋 {len(configs)} config(s) queued:")
            for i, c in enumerate(configs, 1):
                print(f"   {i}. {c}")

            all_results = []
            if args.parallel:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                n_workers = min(args.parallel_max, len(configs))
                print(f"🧵 Running {len(configs)} configs in parallel "
                      f"(max {n_workers} workers)")
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futs = {
                        pool.submit(
                            run_one_config, spec, args, server_url, case_count,
                            args.max_workers, args.results_dir, i,
                        ): (i, spec)
                        for i, spec in enumerate(configs, 1)
                    }
                    for fut in as_completed(futs):
                        i, spec = futs[fut]
                        label, summary, elapsed, run_name = fut.result()
                        all_results.append({
                            "index": i,
                            "config": label,
                            "spec": spec,
                            "run_name": run_name,
                            "elapsed_seconds": round(elapsed, 2),
                            **{k: v for k, v in summary.items() if k != "hyperparams"},
                        })
            else:
                for i, spec in enumerate(configs, 1):
                    label, summary, elapsed, run_name = run_one_config(
                        spec, args, server_url, case_count, args.max_workers,
                        args.results_dir, i,
                    )
                    all_results.append({
                        "index": i,
                        "config": label,
                        "spec": spec,
                        "run_name": run_name,
                        "elapsed_seconds": round(elapsed, 2),
                        **{k: v for k, v in summary.items() if k != "hyperparams"},
                    })

        # === Final comparison table ===
        all_results.sort(key=lambda r: r["index"])
        col_w = max(40, max(len(r["config"]) for r in all_results) + 2)
        print()
        print("=" * 78)
        print(f"📊 FINAL COMPARISON  (case_count={case_count}, max_workers={args.max_workers})")
        print("=" * 78)
        print(f"  {'Config':<{col_w}}  {'Success':>14}  {'Tokens':>12}  {'Tokens(success)':>16}  {'Time':>8}")
        print(f"  {'-' * col_w}  {'-' * 14}  {'-' * 12}  {'-' * 16}  {'-' * 8}")
        for r in all_results:
            if "error" in r and "success" not in r:
                succ = "—"
                tok_str = "—"
                tok_success_str = "—"
            else:
                run = r.get("cases_run", 0)
                ok = r.get("success", 0)
                rate = r.get("success_rate", "?")
                succ = f"{ok}/{run} ({rate})"
                tokens = r.get("token_usage", {})
                total_tok = tokens.get("total_tokens", 0)
                tok_str = f"{total_tok:,}"
                tokens_success = r.get("token_usage_success_only", {})
                total_tok_success = tokens_success.get("total_tokens", 0)
                tok_success_str = f"{total_tok_success:,}"
            print(f"  {r['config']:<{col_w}}  {succ:>14}  {tok_str:>12}  {tok_success_str:>16}  {r['elapsed_seconds']:>7.1f}s")
        print("=" * 94)

        # Save combined report
        grid_path = os.path.join(
            args.results_dir,
            f"grid_{test_start_ts}.json",
        )
        with open(grid_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_start_ts": test_start_ts,
                "case_count": case_count,
                "max_workers": args.max_workers,
                "parallel_configs": bool(args.parallel),
                "per_config_server": bool(args.per_config_server),
                "configs": all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"📁 Grid report: {grid_path}")
        return 0

    finally:
        if server_proc and server_proc.poll() is None:
            print("🛑 Stopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        if server_log_fp and not server_log_fp.closed:
            server_log_fp.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
