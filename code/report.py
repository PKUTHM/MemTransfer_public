"""Live report for an experiment run.

Reads the on-disk artefacts of a run directory and prints what is known so
far. Works mid-run: per-case results are written the moment a case
settles, so this does not wait for a group's SUMMARY.json.

Usage:
    python code/report.py                          # newest run
    python code/report.py --run_dir result/experiment_...
    python code/report.py --cases                  # per-case table
    python code/report.py --steps G01:3            # decision trace for one case
    python code/report.py --json out.json          # machine-readable dump
"""
import argparse
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_NAMES = {1: "find specific object", 2: "find all specified", 3: "find any of category",
              4: "find all of category", 5: "two objects, any order",
              6: "two objects, in order", 7: "find then return", 8: "collect then return",
              9: "A->B->A", 10: "A->B->origin"}


def optimal_steps(run_info):
    """Expert all-actions count for each case in the version under test.

    Read here, in the offline scoring tool, and nowhere else: this is the
    test set's answer key. ``experiment.TestTask`` strips ``planner`` and
    ``checks`` before the server can echo them, so the number cannot reach
    a prompt through the runtime path.
    """
    cfg_path = (run_info or {}).get("config_path")
    dataset_root = os.path.join(PROJECT_ROOT, "dataset", "v0-v5")
    if cfg_path and os.path.isfile(cfg_path):
        cfg = _read(cfg_path) or {}
        root = cfg.get("dataset_root")
        if root:
            dataset_root = root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
    # The answer key belongs to the version under test: v4 by default.
    version = (run_info or {}).get("test_version") or "v4"
    import metrics
    out, relaxed, _ = metrics.load_optimal(dataset_root, version)
    return out, relaxed


def path_metrics(settled_cases, optima, expected_n=None):
    """Success Rate, Step Ratio and SPL.

    SR         = successes / N
    Step Ratio = mean over SUCCESSFUL cases of actual / optimal      (lower better, 1 = optimal)
    SPL        = mean over ALL cases of S * optimal / max(actual, optimal)  (higher better, max 1)

    ``expected_n`` is the intended denominator (100 for a full group).
    When it is larger than the number of settled cases, both SR and SPL
    are reported over what has settled AND flagged as partial, because
    quietly shrinking the denominator is how a partial run comes to look
    like a finished one.
    """
    scored, missing_opt, faster_than_expert = [], [], []
    for c in settled_cases:
        cid = c["legacy_case_id"]
        opt = optima.get(cid)
        act = c.get("actions_executed")
        if not isinstance(opt, int) or not isinstance(act, int):
            missing_opt.append(cid)
            continue
        succ = 1 if c.get("success") else 0
        if succ and act < opt:
            faster_than_expert.append((cid, act, opt))
        scored.append({"case": cid, "success": succ, "optimal": opt, "actual": act,
                       "ratio": (act / opt) if succ else None,
                       "spl": (succ * opt / max(act, opt)) if max(act, opt) else 0.0})
    n_settled = len(scored)
    denom = expected_n if expected_n else n_settled
    succ_list = [x for x in scored if x["success"]]
    ratios = [x["ratio"] for x in succ_list if x["ratio"] is not None]
    spls = [x["spl"] for x in scored]
    return {
        "n_settled": n_settled,
        "n_denominator": denom,
        "partial": bool(expected_n and n_settled < expected_n),
        "successes": len(succ_list),
        "SR": (len(succ_list) / denom) if denom else None,
        "SR_over_settled": (len(succ_list) / n_settled) if n_settled else None,
        "step_ratio_mean": (sum(ratios) / len(ratios)) if ratios else None,
        "step_ratio_median": _median(ratios),
        "SPL": (sum(spls) / denom) if denom else None,
        "SPL_over_settled": (sum(spls) / n_settled) if n_settled else None,
        "cases_missing_optimal": missing_opt,
        "faster_than_expert": faster_than_expert,
        "per_case": scored,
    }


def _read(path):
    try:
        with io.open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _median(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def newest_run(results_root):
    runs = sorted(glob.glob(os.path.join(results_root, "experiment_*")),
                  key=os.path.getmtime, reverse=True)
    if not runs:
        raise SystemExit("no experiment run directories under %s" % results_root)
    return runs[0]


def collect(run_dir):
    """Everything the run has produced so far, grouped by group id."""
    info = _read(os.path.join(run_dir, "RUN_INFO.json")) or {}
    groups = {}
    for client in sorted(glob.glob(os.path.join(run_dir, "client_*"))):
        if not os.path.isdir(client) or os.path.basename(client) == "client_logs":
            continue
        cases, in_flight = [], []
        # case_<group id>_<NNN>_a<k>.json, for any group id (G01.., A01.., B01..)
        for f in sorted(glob.glob(os.path.join(client, "case_*_[0-9][0-9][0-9]_a*.json"))):
            d = _read(f)
            if d:
                cases.append(d)
        settled_ids = {c["legacy_case_id"] for c in cases if c.get("settled")}
        for f in sorted(glob.glob(os.path.join(client, "steps_*_case*_a*.jsonl"))):
            base = os.path.basename(f)
            try:
                cid = int(base.split("case")[1].split("_")[0])
            except Exception:
                continue
            if cid in settled_ids:
                continue
            n = 0
            header = None
            try:
                with io.open(f, encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i == 0:
                            header = json.loads(line)
                        else:
                            n += 1
            except Exception:
                pass
            in_flight.append({"legacy_case_id": cid, "decisions": n,
                              "group_id": (header or {}).get("group_id"),
                              "task_id": (header or {}).get("task_id"),
                              "case_no": (header or {}).get("case_no"),
                              "mtime": os.path.getmtime(f), "path": f})
        summary = _read(os.path.join(client, "SUMMARY.json"))
        gid = None
        if cases:
            gid = cases[0]["group_id"]
        elif in_flight and in_flight[0]["group_id"]:
            gid = in_flight[0]["group_id"]
        elif summary:
            gid = summary.get("group_id")
        if gid:
            groups[gid] = {"cases": cases, "in_flight": in_flight,
                           "summary": summary, "dir": client}
    return info, groups


def group_metrics(entry):
    """Aggregate the per-case files — available long before SUMMARY.json."""
    cases = entry["cases"]
    settled = [c for c in cases if c.get("settled")]
    latest = {}
    for c in settled:                       # keep the accepted attempt per case
        cid = c["legacy_case_id"]
        if cid not in latest or c["attempt_id"] < latest[cid]["attempt_id"]:
            latest[cid] = c
    settled = list(latest.values())
    ok = [c for c in settled if c.get("success")]
    steps_all = [c["actions_executed"] for c in settled
                 if isinstance(c.get("actions_executed"), int)]
    steps_ok = [c["actions_executed"] for c in ok
                if isinstance(c.get("actions_executed"), int)]
    tok = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tok_ok = dict(tok)
    for c in settled:
        for k in tok:
            tok[k] += (c.get("token_usage") or {}).get(k, 0) or 0
    for c in ok:
        for k in tok_ok:
            tok_ok[k] += (c.get("token_usage") or {}).get(k, 0) or 0
    by_type = {}
    for c in settled:
        b = by_type.setdefault(c.get("task_type"), {"n": 0, "ok": 0})
        b["n"] += 1
        b["ok"] += 1 if c.get("success") else 0
    status = {}
    for c in settled:
        status[c["final_status"]] = status.get(c["final_status"], 0) + 1
    return {
        "settled": len(settled),
        "success": len(ok),
        "success_rate_so_far": (len(ok) / len(settled) * 100) if settled else None,
        "unsettled_attempts": len([c for c in cases if not c.get("settled")]),
        "in_flight": len(entry["in_flight"]),
        "attempts_written": len(cases),
        "retried_cases": len({c["legacy_case_id"] for c in cases
                              if (c.get("attempt_id") or 0) > 0}),
        "technical_failures": sum(c.get("technical_failures", 0) or 0 for c in cases),
        "model_output_failures": sum(c.get("model_output_failures", 0) or 0 for c in cases),
        "decisions": sum(c.get("decisions", 0) or 0 for c in cases),
        "actions": sum(c.get("actions_executed", 0) or 0 for c in cases),
        "steps_mean_all": (sum(steps_all) / len(steps_all)) if steps_all else None,
        "steps_median_all": _median(steps_all),
        "steps_mean_success": (sum(steps_ok) / len(steps_ok)) if steps_ok else None,
        "steps_median_success": _median(steps_ok),
        "elapsed_sum": sum(c.get("elapsed", 0) or 0 for c in cases),
        "token_usage_settled": tok,
        "token_usage_success_only": tok_ok,
        "unknown_usage_steps": sum(c.get("steps_with_unknown_usage", 0) or 0 for c in cases),
        "by_task_type": by_type,
        "status_counts": status,
        "settled_cases": settled,
    }


def print_overview(info, groups):
    print("=" * 96)
    print("EXPERIMENT : %s" % info.get("experiment_id", "?"))
    print("FINGERPRINT: %s" % info.get("config_fingerprint", "?"))
    m = info.get("model") or {}
    print("MODEL      : %s @ %s (provider=%s, max_tokens=%s, temp=%s)"
          % (m.get("name"), m.get("api_url"), m.get("provider"),
             m.get("max_tokens"), m.get("temperature")))
    print("SCOPE      : groups=%s  cases=%s  round=%s  max_steps=%s  h/i=%s/%s"
          % (info.get("group_id") or ",".join(info.get("groups") or []),
             info.get("case_ids"),
             info.get("round_id"), info.get("max_steps"),
             (info.get("working_memory") or {}).get("history_steps"),
             (info.get("working_memory") or {}).get("history_images")))
    print("STARTED    : %s   git=%s%s"
          % (info.get("started_at"), (info.get("git_commit") or "")[:8],
             " (dirty)" if info.get("git_dirty") else ""))
    print("=" * 96)

    hdr = ("%-6s %-9s %8s %8s %10s %9s %9s %8s %8s %12s"
           % ("GROUP", "STATE", "SETTLED", "SUCCESS", "RATE*",
              "STEPS~ok", "IN-FLIGHT", "TECH", "MODFAIL", "IN-TOKENS"))
    print(hdr)
    print("-" * 96)
    for gid in sorted(groups):
        e = groups[gid]
        mm = group_metrics(e)
        state = "done" if e["summary"] else ("running" if mm["in_flight"] else "pending")
        rate = ("%.1f%%" % mm["success_rate_so_far"]) if mm["settled"] else "-"
        smed = ("%.0f" % mm["steps_median_success"]) if mm["steps_median_success"] else "-"
        print("%-6s %-9s %8d %8d %10s %9s %9d %8d %8d %12d"
              % (gid, state, mm["settled"], mm["success"], rate, smed,
                 mm["in_flight"], mm["technical_failures"],
                 mm["model_output_failures"],
                 mm["token_usage_settled"]["prompt_tokens"]))
    print("-" * 96)
    print("* RATE is over cases settled SO FAR, not the final 100-case rate.")
    print("  A group is only publishable when its SUMMARY.json says complete=true.")

    optima, relaxed = optimal_steps(info)
    expected = 100 if str(info.get("case_ids")) == "all 100" else None
    print()
    print("%-6s %10s %10s %12s %10s %-24s"
          % ("GROUP", "SR", "SPL", "STEP RATIO", "SETTLED", "BASIS"))
    print("-" * 96)
    for gid in sorted(groups):
        mm = group_metrics(groups[gid])
        pm = path_metrics(mm["settled_cases"], optima, expected_n=expected)
        basis = ("PARTIAL %d/%d settled" % (pm["n_settled"], pm["n_denominator"])
                 if pm["partial"] else "complete over %d" % pm["n_denominator"])
        print("%-6s %10s %10s %12s %10d %-24s"
              % (gid, _pct(pm["SR"]), _num(pm["SPL"], 3),
                 _num(pm["step_ratio_mean"], 2), pm["n_settled"], basis))
        if pm["partial"]:
            print("       over settled only: SR=%s SPL=%s"
                  % (_pct(pm["SR_over_settled"]), _num(pm["SPL_over_settled"], 3)))
        if pm["cases_missing_optimal"]:
            print("       WARNING no optimal length for cases %s"
                  % pm["cases_missing_optimal"][:8])
        if pm["faster_than_expert"]:
            print("       beat the expert (case, actual, optimal): %s"
                  % pm["faster_than_expert"][:5])
    print("-" * 96)
    print("SR         = successes / N                                   higher better")
    print("Step Ratio = mean over SUCCESSES of actual/optimal            lower better, 1 = optimal")
    print("SPL        = mean over ALL of success x optimal/max(act,opt)  higher better, max 1")
    print("Optimal = the %s expert trajectory, all_actions.length "
          "(v4: planner=optimal; v0: planner=original)."
          % (info.get("test_version") or "v4"))
    if relaxed:
        print("Note: %d case(s) were planned under relaxed constraints (relax_level>0): %s"
              % (len(relaxed), relaxed))

    for gid in sorted(groups):
        e = groups[gid]
        mm = group_metrics(e)
        if not mm["settled"]:
            continue
        print()
        print("[%s] status: %s" % (gid, mm["status_counts"]))
        if mm["by_task_type"]:
            parts = []
            for t in sorted(x for x in mm["by_task_type"] if x is not None):
                b = mm["by_task_type"][t]
                parts.append("t%d %d/%d" % (t, b["ok"], b["n"]))
            print("[%s] by task type: %s" % (gid, "  ".join(parts)))
        print("[%s] steps: all mean=%s median=%s | success mean=%s median=%s"
              % (gid,
                 _fmt(mm["steps_mean_all"]), _fmt(mm["steps_median_all"]),
                 _fmt(mm["steps_mean_success"]), _fmt(mm["steps_median_success"])))
        t, to = mm["token_usage_settled"], mm["token_usage_success_only"]
        print("[%s] tokens: settled in=%d out=%d | success-only in=%d | unknown-usage steps=%d"
              % (gid, t["prompt_tokens"], t["completion_tokens"],
                 to["prompt_tokens"], mm["unknown_usage_steps"]))
        if mm["decisions"]:
            print("[%s] decisions=%d actions=%d (gap = technical retries) | elapsed sum=%.0fs"
                  % (gid, mm["decisions"], mm["actions"], mm["elapsed_sum"]))
        if e["summary"] and e["summary"].get("circuit_breaker_tripped"):
            print("[%s] CIRCUIT BREAKER: %s" % (gid, e["summary"]["circuit_breaker_tripped"]))


def _fmt(v):
    return "-" if v is None else ("%.1f" % v)


def _pct(v):
    return "-" if v is None else ("%.1f%%" % (v * 100))


def _num(v, nd):
    return "-" if v is None else ("%.*f" % (nd, v))


def print_cases(groups, info=None):
    optima, _ = optimal_steps(info)
    print()
    print("%-6s %-5s %-12s %-4s %-18s %-4s %5s %5s %5s %5s %6s %6s %9s %8s"
          % ("GROUP", "CASE", "TASK", "TYPE", "STATUS", "OK",
             "DEC", "ACT", "OPT", "RATIO", "SPL", "STG", "IN-TOK", "TIME(s)"))
    print("-" * 128)
    for gid in sorted(groups):
        mm = group_metrics(groups[gid])
        for c in sorted(mm["settled_cases"], key=lambda x: x["legacy_case_id"]):
            opt = optima.get(c["legacy_case_id"])
            act = c.get("actions_executed")
            succ = bool(c.get("success"))
            ratio = ("%.2f" % (act / opt)) if (succ and opt and isinstance(act, int)) else "-"
            spl = ("%.3f" % ((1 if succ else 0) * opt / max(act, opt)))                 if (opt and isinstance(act, int) and max(act, opt)) else "-"
            print("%-6s %-5d %-12s %-4s %-18s %-4s %5s %5s %5s %5s %6s %6s %9d %8.0f"
                  % (gid, c["legacy_case_id"],
                     "%s/case%02d" % (c.get("task_id"), c.get("case_no", 0)),
                     c.get("task_type"), c.get("final_status"),
                     "yes" if succ else "no",
                     c.get("decisions"), act, opt if opt else "-", ratio, spl,
                     "%s/%s" % (c.get("stages_completed"), c.get("stages_total")),
                     (c.get("token_usage") or {}).get("prompt_tokens", 0),
                     c.get("elapsed", 0)))
        for f in sorted(groups[gid]["in_flight"], key=lambda x: x["legacy_case_id"]):
            print("%-6s %-5d %-12s %-4s %-18s %-6s %5d %5s %5s %5s %9s %8s"
                  % (gid, f["legacy_case_id"],
                     "%s/case%s" % (f.get("task_id"), f.get("case_no")),
                     "-", "RUNNING", "-", f["decisions"], "-", "-", "-", "-", "-"))


def print_steps(run_dir, groups, spec):
    gid, _, cid = spec.partition(":")
    gid = gid.upper()
    if gid not in groups:
        raise SystemExit("group %s not in this run (have %s)" % (gid, sorted(groups)))
    matches = glob.glob(os.path.join(groups[gid]["dir"],
                                     "steps_%s_case%03d_a*.jsonl" % (gid, int(cid))))
    if not matches:
        raise SystemExit("no step log for %s case %s" % (gid, cid))
    path = sorted(matches)[0]
    print("\n=== %s ===" % os.path.basename(path))
    with io.open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if d.get("record_type") == "run_header":
                print("  experiment=%s group=%s case=%s/%s model=%s max_tokens=%s"
                      % (d["experiment_id"], d["group_id"], d["task_id"],
                         d["case_no"], d["model"], d["max_tokens"]))
                continue
            tel = (d.get("request") or {}).get("telemetry") or {}
            imgs = tel.get("images") or {}
            att = tel.get("api_attempts") or []
            extra = ""
            if tel.get("episodic_match"):
                m = tel["episodic_match"]
                extra = " | epis v=%s idx=%s kind=%s" % (m.get("version"),
                                                        m.get("matched_index"),
                                                        m.get("match_kind"))
            elif tel.get("full_context"):
                extra = " | refimgs " + ",".join(
                    "%s:%d" % (s["version"], s["images_sent"]) for s in tel["full_context"])
            print("  d%-3s seq=%-4s %-18s %-16s imgs=%s(%s/%s/%s) api=%s %s%s"
                  % (d.get("decision_index"), d.get("action_seq", "-"),
                     str(d.get("action"))[:18], str(d.get("server_status"))[:16],
                     imgs.get("total"), imgs.get("current"), imgs.get("history"),
                     imgs.get("memory"),
                     ",".join("%s/%.1fs" % (a.get("status"), a.get("elapsed", 0))
                              for a in att),
                     str(d.get("pose_after") or ""), extra))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=None)
    p.add_argument("--results_root", default=os.path.join(PROJECT_ROOT, "result"))
    p.add_argument("--cases", action="store_true", help="per-case table")
    p.add_argument("--steps", default=None, metavar="GROUP:CASE",
                   help="print the decision trace, e.g. --steps G01:3")
    p.add_argument("--json", default=None, help="write the aggregate to a JSON file")
    args = p.parse_args()

    run_dir = args.run_dir or newest_run(args.results_root)
    info, groups = collect(run_dir)
    print("RUN DIR    : %s" % run_dir)
    print_overview(info, groups)
    if args.cases:
        print_cases(groups, info)
    if args.steps:
        print_steps(run_dir, groups, args.steps)
    if args.json:
        optima, relaxed = optimal_steps(info)
        expected = 100 if str(info.get("case_ids")) == "all 100" else None
        out = {"run_dir": run_dir, "run_info": info,
               "relaxed_plan_cases": relaxed,
               "groups": {}}
        for g in groups:
            mm = group_metrics(groups[g])
            out["groups"][g] = {k: v for k, v in mm.items() if k != "settled_cases"}
            out["groups"][g]["path_metrics"] = path_metrics(
                mm["settled_cases"], optima, expected_n=expected)
        with io.open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
