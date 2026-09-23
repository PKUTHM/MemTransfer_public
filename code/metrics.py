"""Compute SR, Step Ratio and SPL for every group of a run.

Standalone: reads the result directory directly and needs nothing but the
standard library. Hand it a run directory and it prints the three headline
metrics per group, per task type, and optionally per case, and can export
CSV or JSON.

Definitions
-----------
For case *i* let

    S_i  = 1 if the case ended in ``task_completed``, else 0
    p_i  = actions the agent actually executed (server-confirmed)
    l_i  = actions in the test version's ``all_actions`` trajectory

    SR         = (1/N) * sum_i  S_i                       higher better
    Step Ratio = mean over SUCCESSES of  p_i / l_i        lower better, 1 = optimal
    SPL        = (1/N) * sum_i  S_i * l_i / max(p_i, l_i) higher better, max 1

``N`` is the group's intended case count (100), not the number that
happened to finish. A group that has not finished is reported over what
settled AND marked PARTIAL, because shrinking the denominator silently is
how an unfinished run comes to look like a finished one.

Two things worth knowing when reading the numbers:

* ``p_i`` counts every action the server accepted, including ``blocked``
  and ``out_of_range`` ones that did not move the agent. That is the
  agent's real cost, and it is the same convention on both sides of any
  comparison.
* Step Ratio is averaged over *successes only*, so two groups with very
  different success rates are not averaging over the same cases. A group
  that only ever solves the easy short cases can show a better Step Ratio
  than one that also solves hard long ones. SPL does not have this
  problem — its denominator is all N cases — so prefer SPL when comparing
  groups whose SR differs a lot.

Usage
-----
    python code/metrics.py                            # newest run
    python code/metrics.py --run_dir result/experiment_...
    python code/metrics.py --by_task_type
    python code/metrics.py --cases G03
    python code/metrics.py --csv metrics.csv --json metrics.json
    python code/metrics.py --verify                   # cross-check vs report.py
"""
import argparse
import csv
import glob
import io
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TASK_NAMES = {
    1: "find specific object", 2: "find all specified", 3: "find any of category",
    4: "find all of category", 5: "two objects, any order", 6: "two objects, in order",
    7: "find then return", 8: "collect then return", 9: "A->B->A", 10: "A->B->origin",
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _read_json(path):
    try:
        with io.open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def newest_run(results_root):
    runs = sorted(glob.glob(os.path.join(results_root, "experiment_*")),
                  key=os.path.getmtime, reverse=True)
    if not runs:
        raise SystemExit("no experiment run directories under %s" % results_root)
    return runs[0]


def load_optimal(dataset_root, test_version="v4"):
    """Expert all-actions count per case for the version under test.

    This is the test set's answer key. It is read here, in offline
    scoring, and never by the runtime: ``experiment.TestTask`` drops
    ``planner`` and ``checks`` before the server can echo them. The agent
    may move in any allowed direction, so ``forward_only`` is not the
    appropriate reference for Step Ratio or SPL.
    """
    optimal, relaxed, planners = {}, [], set()
    for task_json in glob.glob(os.path.join(dataset_root, "T*", "case*",
                                            test_version, "task.json")):
        d = _read_json(task_json)
        if not d:
            raise SystemExit("cannot read task: %s" % task_json)
        trajectory_json = os.path.join(os.path.dirname(task_json), "trajectory.json")
        trajectory = _read_json(trajectory_json)
        if not trajectory:
            raise SystemExit("cannot read trajectory: %s" % trajectory_json)
        all_actions = trajectory.get("all_actions") or {}
        n = all_actions.get("length")
        if type(n) is not int or n <= 0:
            raise SystemExit("invalid all_actions.length: %s" % trajectory_json)
        checks = d.get("checks") or {}
        checked_length = checks.get("len_all_actions")
        if checked_length is not None and checked_length != n:
            raise SystemExit("all-actions length mismatch: %s" % task_json)
        cid = d["legacy_case_id"]
        if cid in optimal:
            raise SystemExit("duplicate legacy case ID %s for %s" % (cid, test_version))
        optimal[cid] = n
        planners.add(d.get("planner"))
        if checks.get("relax_level"):
            relaxed.append(cid)
    if not optimal:
        raise SystemExit("no %s all-actions trajectories found under %s"
                         % (test_version, dataset_root))
    return optimal, sorted(relaxed), sorted(x for x in planners if x)


def load_run(run_dir):
    """Return (run_info, {group_id: {'cases': {...}, 'summary': {...}}}).

    Only the accepted attempt of each case is kept: retries exist for
    technical failures, and counting them all would double-count cases.
    """
    info = _read_json(os.path.join(run_dir, "RUN_INFO.json")) or {}
    groups = {}
    for client in sorted(glob.glob(os.path.join(run_dir, "client_*"))):
        if not os.path.isdir(client) or os.path.basename(client) == "client_logs":
            continue
        summary = _read_json(os.path.join(client, "SUMMARY.json"))
        best = {}
        # case_<group id>_<NNN>_a<k>.json, for any group id (G01.., A01.., B01..)
        for f in sorted(glob.glob(os.path.join(client, "case_*_[0-9][0-9][0-9]_a*.json"))):
            c = _read_json(f)
            if not c or not c.get("settled"):
                continue
            cid = c["legacy_case_id"]
            # earliest settled attempt is the accepted one
            if cid not in best or c["attempt_id"] < best[cid]["attempt_id"]:
                best[cid] = c
        if not best and not summary:
            continue
        gid = (summary or {}).get("group_id") or next(iter(best.values()))["group_id"]
        groups[gid] = {"cases": best, "summary": summary, "dir": client}
    return info, groups


# --------------------------------------------------------------------------
# the three metrics
# --------------------------------------------------------------------------
def compute(cases, optimal, expected_n):
    """SR, Step Ratio and SPL for one group. ``cases`` maps id -> record."""
    rows, missing, faster = [], [], []
    for cid, c in sorted(cases.items()):
        l_i = optimal.get(cid)
        p_i = c.get("actions_executed")
        if not isinstance(l_i, int) or not isinstance(p_i, int):
            missing.append(cid)
            continue
        s_i = 1 if c.get("success") else 0
        if s_i and p_i < l_i:
            faster.append({"case": cid, "actual": p_i, "optimal": l_i})
        rows.append({
            "case": cid,
            "task_id": c.get("task_id"), "case_no": c.get("case_no"),
            "task_type": c.get("task_type"),
            "success": s_i,
            "optimal": l_i,
            "actual": p_i,
            "step_ratio": (p_i / l_i) if s_i else None,
            "spl": (s_i * l_i / max(p_i, l_i)) if max(p_i, l_i) else 0.0,
            "final_status": c.get("final_status"),
        })
    n_settled = len(rows)
    denom = expected_n or n_settled
    succ = [r for r in rows if r["success"]]
    ratios = [r["step_ratio"] for r in succ]
    return {
        "n_settled": n_settled,
        "n_denominator": denom,
        "partial": bool(expected_n and n_settled < expected_n),
        "successes": len(succ),
        "SR": (len(succ) / denom) if denom else None,
        "SR_over_settled": (len(succ) / n_settled) if n_settled else None,
        "SPL_over_settled": ((sum(r["spl"] for r in rows) / n_settled)
                             if n_settled else None),
        "step_ratio_mean": (sum(ratios) / len(ratios)) if ratios else None,
        "step_ratio_median": _median(ratios),
        "SPL": (sum(r["spl"] for r in rows) / denom) if denom else None,
        "cases_missing_optimal": missing,
        "faster_than_expert": faster,
        "rows": rows,
    }


def by_task_type(result):
    out = {}
    for r in result["rows"]:
        t = r["task_type"]
        b = out.setdefault(t, {"n": 0, "successes": 0, "ratios": [], "spl": 0.0})
        b["n"] += 1
        b["successes"] += r["success"]
        b["spl"] += r["spl"]
        if r["step_ratio"] is not None:
            b["ratios"].append(r["step_ratio"])
    for t, b in out.items():
        b["SR"] = b["successes"] / b["n"] if b["n"] else None
        b["SPL"] = b["spl"] / b["n"] if b["n"] else None
        b["step_ratio_mean"] = (sum(b["ratios"]) / len(b["ratios"])) if b["ratios"] else None
        del b["spl"], b["ratios"]
    return out


def _median(v):
    if not v:
        return None
    s = sorted(v)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _pct(v):
    return "-" if v is None else "%.1f%%" % (v * 100)


def _num(v, nd=3):
    return "-" if v is None else "%.*f" % (nd, v)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def print_table(info, results, groups, relaxed, planners):
    m = info.get("model") or {}
    print("=" * 92)
    print("EXPERIMENT : %s" % info.get("experiment_id", "?"))
    print("FINGERPRINT: %s" % (info.get("config_fingerprint") or "?"))
    print("MODEL      : %s @ %s" % (m.get("name"), m.get("api_url")))
    print("OPTIMAL    : %s expert trajectory, planner=%s"
          % (info.get("test_version") or "v4", ",".join(planners)))
    print("=" * 92)
    print()
    print("%-6s %-30s %8s %10s %12s %9s %8s"
          % ("GROUP", "METHOD", "SR", "SPL", "STEP RATIO", "SETTLED", "BASIS"))
    print("-" * 92)
    for gid in sorted(results):
        r = results[gid]
        grp = (groups[gid].get("summary") or {}).get("group") or {}
        label = grp.get("label") or grp.get("slug") or ""     # older runs have a label
        print("%-6s %-30s %8s %10s %12s %9d %8s"
              % (gid, label[:30], _pct(r["SR"]), _num(r["SPL"], 4),
                 _num(r["step_ratio_mean"], 2), r["n_settled"],
                 "PARTIAL" if r["partial"] else "complete"))
    print("-" * 92)
    print("SR         = successes / N                                    higher better")
    print("Step Ratio = mean over SUCCESSES of actual/optimal            lower better, 1 = optimal")
    print("SPL        = mean over ALL N of success x optimal/max(a,o)    higher better, max 1")
    partial = [g for g in results if results[g]["partial"]]
    if partial:
        print()
        print("PARTIAL groups: the SR/SPL above use N=100 with unfinished cases")
        print("counted as failures. Over what has actually settled so far:")
        for gid in sorted(partial):
            r = results[gid]
            print("  %-6s settled %d/%d   SR %s   SPL %s"
                  % (gid, r["n_settled"], r["n_denominator"],
                     _pct(r["SR_over_settled"]), _num(r["SPL_over_settled"], 4)))
        print("Neither figure is publishable until the group finishes; the first")
        print("understates and the second overstates.")
    if relaxed:
        print()
        print("Note: %d case(s) were planned under relaxed constraints "
              "(relax_level>0): %s" % (len(relaxed), relaxed))
    for gid in sorted(results):
        r = results[gid]
        if r["cases_missing_optimal"]:
            print("WARNING %s: no expert length for cases %s"
                  % (gid, r["cases_missing_optimal"][:10]))
        if r["faster_than_expert"]:
            print("WARNING %s: beat the 'optimal' expert on %s — check the baseline"
                  % (gid, r["faster_than_expert"][:5]))


def print_by_task_type(results):
    print()
    print("PER TASK TYPE")
    header = "%-4s %-24s" % ("TYPE", "TASK")
    for gid in sorted(results):
        header += " %-18s" % gid
    print(header)
    print("-" * len(header))
    types = sorted({t for r in results.values() for t in by_task_type(r)})
    cache = {g: by_task_type(results[g]) for g in results}
    for t in types:
        line = "%-4s %-24s" % ("t%d" % t, TASK_NAMES.get(t, "")[:24])
        for gid in sorted(results):
            b = cache[gid].get(t)
            line += (" %-18s" % ("%d/%-3d SPL %s" % (b["successes"], b["n"], _num(b["SPL"], 2)))
                     if b else " %-18s" % "-")
        print(line)


def print_cases(result, gid):
    print()
    print("PER CASE  %s" % gid)
    print("%-5s %-12s %-4s %-18s %-4s %6s %6s %8s %7s"
          % ("CASE", "TASK", "TYPE", "STATUS", "OK", "ACTUAL", "OPT", "RATIO", "SPL"))
    print("-" * 82)
    for r in result["rows"]:
        print("%-5d %-12s %-4s %-18s %-4s %6d %6d %8s %7s"
              % (r["case"], "%s/case%02d" % (r["task_id"], r["case_no"] or 0),
                 r["task_type"], r["final_status"], "yes" if r["success"] else "no",
                 r["actual"], r["optimal"],
                 _num(r["step_ratio"], 2), _num(r["spl"], 3)))


def write_csv(path, results):
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "case", "task_id", "case_no", "task_type",
                    "final_status", "success", "actual_steps", "optimal_steps",
                    "step_ratio", "spl"])
        for gid in sorted(results):
            for r in results[gid]["rows"]:
                w.writerow([gid, r["case"], r["task_id"], r["case_no"], r["task_type"],
                            r["final_status"], r["success"], r["actual"], r["optimal"],
                            "" if r["step_ratio"] is None else "%.6f" % r["step_ratio"],
                            "%.6f" % r["spl"]])
    print("\nwrote %s" % path)


def verify_against_report(run_dir, results):
    """Cross-check against report.py, which computes the same three metrics
    on the way to its live view. Two implementations that disagree mean one
    of them is wrong; catching that here is cheaper than publishing it."""
    try:
        import report as rp
    except Exception as e:
        print("\ncould not import report.py for cross-check: %s" % e)
        return True
    info, groups = rp.collect(run_dir)
    optima, _ = rp.optimal_steps(info)
    ok = True
    print()
    print("CROSS-CHECK vs report.py")
    for gid in sorted(results):
        if gid not in groups:
            continue
        mm = rp.group_metrics(groups[gid])
        # Same N on both sides: report.py falls back to "cases settled" when
        # a group has no SUMMARY yet, so comparing without pinning the
        # denominator would flag a convention difference as a bug.
        pm = rp.path_metrics(mm["settled_cases"], optima,
                             expected_n=results[gid]["n_denominator"])
        mine = results[gid]
        diffs = []
        for key in ("SR", "SPL", "step_ratio_mean"):
            a, b = mine.get(key), pm.get(key)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > 1e-9:
                diffs.append("%s %s != %s" % (key, a, b))
        print("  %-6s %s" % (gid, "match" if not diffs else "MISMATCH: " + "; ".join(diffs)))
        ok = ok and not diffs
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", default=None, action="append",
                    help="may be given more than once; groups from several "
                         "run directories are merged (each group must appear "
                         "in only one of them)")
    ap.add_argument("--results_root", default=os.path.join(PROJECT_ROOT, "result"))
    ap.add_argument("--dataset_root", default=None,
                    help="where the v4 task.json files live "
                         "(default: taken from the run's config)")
    ap.add_argument("--groups", default=None, help="comma-separated subset")
    ap.add_argument("--by_task_type", action="store_true")
    ap.add_argument("--cases", default=None, metavar="GROUP",
                    help="print the per-case table for one group")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the numbers against report.py")
    args = ap.parse_args()

    run_dirs = args.run_dir or [newest_run(args.results_root)]
    info, groups, sources = {}, {}, {}
    for d in run_dirs:
        i, g = load_run(d)
        info = info or i
        for gid, entry in g.items():
            if gid in groups:
                raise SystemExit(
                    "group %s appears in two run directories (%s and %s); "
                    "results cannot be merged without knowing which is "
                    "authoritative" % (gid, sources[gid], d))
            groups[gid] = entry
            sources[gid] = d
        fp = i.get("config_fingerprint")
        if fp and info.get("config_fingerprint") and fp != info["config_fingerprint"]:
            raise SystemExit(
                "run directories have different config fingerprints (%s vs %s); "
                "they are not the same experiment"
                % (fp[:16], info["config_fingerprint"][:16]))
    run_dir = run_dirs[0]
    if not groups:
        raise SystemExit("no group results in %s" % ", ".join(run_dirs))

    dataset_root = args.dataset_root
    if not dataset_root:
        cfg = _read_json(info.get("config_path") or "") or {}
        root = cfg.get("dataset_root") or os.path.join("dataset", "v0-v5")
        dataset_root = root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
    optimal, relaxed, planners = load_optimal(dataset_root,
                                              info.get("test_version") or "v4")

    wanted = ([g.strip().upper() for g in args.groups.split(",")]
              if args.groups else sorted(groups))
    results = {}
    for gid in wanted:
        if gid not in groups:
            print("skipping %s: not in this run" % gid)
            continue
        expected = (groups[gid].get("summary") or {}).get("cases_expected") or 100
        results[gid] = compute(groups[gid]["cases"], optimal, expected)
    if not results:
        raise SystemExit("no matching groups")

    for d in run_dirs:
        print("RUN DIR    : %s" % d)
    print_table(info, results, groups, relaxed, planners)
    if args.by_task_type:
        print_by_task_type(results)
    if args.cases:
        gid = args.cases.upper()
        if gid in results:
            print_cases(results[gid], gid)
        else:
            print("no such group in this run: %s" % gid)

    rc = 0
    if args.verify:
        for d in run_dirs:
            here = {g: results[g] for g in results if sources[g] == d}
            if here and not verify_against_report(d, here):
                rc = 1

    if args.csv:
        write_csv(args.csv, results)
    if args.json:
        out = {
            "run_dirs": run_dirs,
            "group_source": sources,
            "experiment_id": info.get("experiment_id"),
            "config_fingerprint": info.get("config_fingerprint"),
            "model": info.get("model"),
            "optimal_source": {"dataset_root": dataset_root,
                               "planner": planners,
                               "relaxed_plan_cases": relaxed},
            "definitions": {
                "SR": "successes / N",
                "step_ratio": "mean over successes of actual/optimal",
                "SPL": "mean over all N of success * optimal / max(actual, optimal)",
            },
            "groups": {g: {k: v for k, v in results[g].items() if k != "rows"}
                       for g in results},
            "per_case": {g: results[g]["rows"] for g in results},
        }
        with io.open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("wrote %s" % args.json)
    return rc


if __name__ == "__main__":
    sys.exit(main())
