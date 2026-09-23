#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemTransfer RQ3 — distractor bank build script
===============================================

Automatically select easy / hard distractor trajectories for each target case,
and mark "trap points".

    python rq3_build_bank.py --selftest              # self-test first, no dataset needed
    python rq3_build_bank.py <traj_dir> <output_dir>  # then run on real data

★ No hyperparameters to choose. ★
----------------------------------
The operator does not need to make any numerical judgments. There are only two kinds of constants in the script:

  (1) Fixed by experimental design, not tuned:
      BANK_SIZE = 4      The bank holds 4 trajectories, to align one-by-one with the
                         already-run D-R (B03/B05, 4 trajectories × 10 frames = 40 frames).
                         Changing this makes results incomparable.

  (2) Determined by the data itself, automatically selected and printed by the script:
      length matching window   Starts from the tightest ±10%, progressively relaxes until
                               every case can fill its bank, then prints the chosen window
                               for the paper appendix. The operator does not choose; the
                               data chooses.

The three roles are selected by pure ranking — no thresholds, no "relax if unsatisfied" —
so there is always a solution:

      H1..H4 = misleading coverage c·m      top 4
      E1..E3 = view overlap ratio c         bottom 3
      L1..L3 = benign coverage c·(1−m)      top 3 (optional arm, only used in Tier 6)

Where do distractor trajectories come from?
--------------------------------------------
Candidate pool = **V0, V1, V2, V3 from other cases** (100 cases → 99×4 = 396 trajectories).
The current case's own V1–V3 are not distractors — they share the same task, goal, and map
as V4, just different start positions. By definition they are "related experience"; the
condition where they fill three slots is called D-R, which has already been run (B03/B05).
The distractor condition does exactly this: replace those three **own** variants with
**others'** trajectories.

★ Each source case contributes at most one trajectory. V0/V1/V2 of the same case follow
  roughly the same route and have nearly identical c·m. Without deduplication, H1/H2/H3
  would very likely be three variants of the same case — "3 distractors" becomes
  "1 distractor repeated 3 times".

What it needs (only one thing)
------------------------------
Step-by-step (x, y, heading, action) for each oracle trajectory, plus which steps on V4
are milestones.

No map files needed (walkable cells are inferred from the union of all trajectories).
No planner needed (progress classification uses "BFS distance to next milestone").
No manual judgment of "does this look similar".

Three numbers
-------------
For each target case t and each candidate trajectory D:

    c    = |Φ(D) ∩ Φ(τ*₄)| / |Φ(τ*₄)|     view overlap ratio: whether it would be retrieved
    m    = proportion of overlapping poses where D's action is wrong   whether it misleads
    c·m  = misleading coverage: proportion of decision points that would be retrieved AND mislead

Three-way progress classification
----------------------------------
MF/MB/ML/MR all translate directly, so turning usually does not change distance to goal —
neither advancing nor retreating. A binary scheme would misclassify every turn as "wrong".
Therefore:

    PRESERVING  BFS distance to next milestone decreases
    REGRESSING  increases
    NEUTRAL     unchanged (most turns, and movement along equidistant edges)

m is only computed on PRESERVING / REGRESSING; NEUTRAL is excluded entirely.

The only thing to change
-------------------------
load_trajectory(). Modify it to read your actual trajectory format; do not change anything else.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Pose = Tuple[int, int, str]          # (x, y, heading)  heading in {N, E, S, W}
Cell = Tuple[int, int]

PRESERVING, NEUTRAL, REGRESSING = "preserving", "neutral", "regressing"

HEADINGS = ("N", "E", "S", "W")
DELTA = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
MOVE_DIR = {"Move_Forward": 0, "Move_Backward": 2, "Move_Left": 3, "Move_Right": 1}
TURNS = {"Turn_Left": -1, "Turn_Right": 1, "Turn_Back": 2}

# ---- Constants fixed by experimental design, not hyperparameters ----
BANK_SIZE = 4          # Aligned with already-run D-R (4 trajectories × 10 frames)
N_HARD, N_EASY, N_LOOK = 4, 3, 3      # H has one extra for D-N

# Distractor trajectory candidate pool: all memory-eligible variants from other cases, not just V0.
# The 09-04 execution version restricted to "only V0 from other cases" (99 candidates);
# expanding to V0–V3 yields 396 candidates, and the selected H is clearly more adversarial.
# The cost is offset by "at most one trajectory per source case" below.
MEMORY_ELIGIBLE = ("V0", "V1", "V2", "V3")

# ---- Candidate ladder for length matching window; script auto-selects the tightest feasible tier, operator does not choose ----
LENGTH_WINDOWS: Sequence[Optional[float]] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, None)


# ============================================================================
# 1. Data structures + the single integration point
# ============================================================================


@dataclass
class Step:
    pose: Pose
    action: str
    milestone: bool = False          # Whether this step completes a required target


@dataclass
class Trajectory:
    episode_id: str                  # MB-WH-T03-C04-V0
    task_id: str                     # T03
    case_id: str                     # T03-C04
    variant: str                     # V0 ... V5
    required: Tuple[str, ...] = ()
    categories: Tuple[str, ...] = ()
    steps: List[Step] = field(default_factory=list)

    def view_set(self) -> Set[Pose]:
        """Phi(tau). Static scene + deterministic rendering => same pose always yields same observation."""
        return {s.pose for s in self.steps}

    def first_action_at(self) -> Dict[Pose, str]:
        """When the same pose appears multiple times, take the first action; consistent with FC frame sampling / episodic indexing."""
        out: Dict[Pose, str] = {}
        for s in self.steps:
            out.setdefault(s.pose, s.action)
        return out

    def length(self) -> int:
        return len(self.steps)


def load_trajectory(path: Path) -> Trajectory:
    """
    ★ The only place in the entire script that needs modification. Expected JSON:

    {
      "episode_id": "MB-WH-T03-C04-V0",
      "task_id": "T03", "case_id": "T03-C04", "variant": "V0",
      "required":   ["can_01", "can_02"],
      "categories": ["tomato_soup_can"],
      "steps": [
        {"pose": [-5, 16, "N"], "action": "Move_Forward", "milestone": false},
        ...
      ]
    }

    If your milestones are stored as a list of step indices, expand them to booleans here.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = [
        Step(pose=(int(s["pose"][0]), int(s["pose"][1]), str(s["pose"][2])),
             action=str(s["action"]),
             milestone=bool(s.get("milestone", False)))
        for s in raw["steps"]
    ]
    return Trajectory(
        episode_id=raw["episode_id"], task_id=raw["task_id"],
        case_id=raw["case_id"], variant=raw["variant"],
        required=tuple(raw.get("required", ())),
        categories=tuple(raw.get("categories", ())),
        steps=steps,
    )


def validate(trajs: Dict[str, Trajectory]) -> None:
    """Catch format issues before running, so we don't discover the bank is empty after the fact."""
    if not trajs:
        raise SystemExit("No trajectories loaded. Check the directory path and load_trajectory.")

    bad_h = {s.pose[2] for t in trajs.values() for s in t.steps} - set(HEADINGS)
    if bad_h:
        raise SystemExit(f"Heading field contains {sorted(bad_h)}, expected N/E/S/W. "
                         f"Map it in load_trajectory.")

    known = set(MOVE_DIR) | set(TURNS)
    bad_a = {s.action for t in trajs.values() for s in t.steps} - known
    if bad_a:
        raise SystemExit(f"Action names {sorted(bad_a)[:5]} not in {sorted(known)}. "
                         f"Map them in load_trajectory.")

    v4 = [t for t in trajs.values() if t.variant == "V4"]
    if not v4:
        raise SystemExit("No V4 trajectories found. Is the variant field named differently?")
    if not any(s.milestone for t in v4 for s in t.steps):
        print("⚠ All V4 milestones are false — will fall back to 'trajectory endpoint as sole milestone'. "
              "Fine for single-goal tasks; exhaustive/ordered tasks will have artificially low m. Please add milestones.")
    pool_variants = {t.variant for t in trajs.values()} & set(MEMORY_ELIGIBLE)
    if pool_variants == {"V0"}:
        print("⚠ Only V0 loaded, no V1–V3 — candidate pool shrinks from 396 to 99, "
              "selected H will be less adversarial. Make sure V1–V3 directories are included.")
    if not any(t.required or t.categories for t in trajs.values()):
        print("⚠ Both required and categories are empty — cannot check 'whether a distractor accidentally satisfies the goal'. "
              "At minimum, fill in categories; otherwise a second correct demonstration may be included as a distractor.")


# ============================================================================
# 2. Walkable cells + BFS (no map files needed)
# ============================================================================


def walkable_from_trajectories(trajs: Iterable[Trajectory]) -> Set[Cell]:
    """
    Union of all cells occupied by oracle trajectories approximates free space. Conservative
    but fully sufficient for BFS along corridors, and no map files needed. If you happen to
    have shelf coordinates, passing real walkable cells is better.
    """
    return {(s.pose[0], s.pose[1]) for t in trajs for s in t.steps}


class Grid:
    """Shortest translation distance on the grid, cached by source."""

    def __init__(self, walkable: Set[Cell]):
        self.walkable = walkable
        self._cache: Dict[Cell, Dict[Cell, int]] = {}

    def _bfs(self, src: Cell) -> Dict[Cell, int]:
        if src not in self._cache:
            dist, q = {src: 0}, deque([src])
            while q:
                c = q.popleft()
                for dx, dy in DELTA.values():
                    n = (c[0] + dx, c[1] + dy)
                    if n in self.walkable and n not in dist:
                        dist[n] = dist[c] + 1
                        q.append(n)
            self._cache[src] = dist
        return self._cache[src]

    def dist(self, a: Cell, b: Cell) -> int:
        return self._bfs(a).get(b, 10 ** 6)


def transition(pose: Pose, action: str, walkable: Set[Cell]) -> Pose:
    """Deterministic transition; hitting a wall = stay in place (consistent with simulator blocked feedback)."""
    x, y, h = pose
    if action in TURNS:
        return (x, y, HEADINGS[(HEADINGS.index(h) + TURNS[action]) % 4])
    if action in MOVE_DIR:
        d = HEADINGS[(HEADINGS.index(h) + MOVE_DIR[action]) % 4]
        nx, ny = x + DELTA[d][0], y + DELTA[d][1]
        return (nx, ny, h) if (nx, ny) in walkable else pose
    return pose


# ============================================================================
# 3. Progress classification
# ============================================================================


def progress_reference(v4: Trajectory) -> Dict[Pose, Cell]:
    """
    For each pose on the V4 oracle, give the cell of its "next milestone".
    Exhaustive and ordered tasks are handled automatically: the next milestone
    is simply the oracle's own next stop.
    """
    idx = [i for i, s in enumerate(v4.steps) if s.milestone] or [len(v4.steps) - 1]
    if idx[-1] != len(v4.steps) - 1:
        idx.append(len(v4.steps) - 1)          # last stop for return-to-origin tasks
    out: Dict[Pose, Cell] = {}
    for i, s in enumerate(v4.steps):
        j = next((k for k in idx if k >= i), idx[-1])
        out.setdefault(s.pose, (v4.steps[j].pose[0], v4.steps[j].pose[1]))
    return out


def classify(grid: Grid, pose: Pose, action: str, goal: Cell, walkable: Set[Cell]) -> str:
    d0 = grid.dist((pose[0], pose[1]), goal)
    p1 = transition(pose, action, walkable)
    d1 = grid.dist((p1[0], p1[1]), goal)
    return PRESERVING if d1 < d0 else (REGRESSING if d1 > d0 else NEUTRAL)


# ============================================================================
# 4. Three numbers
# ============================================================================


@dataclass
class Score:
    c: float
    m: float
    n_shared: int
    n_judged: int

    @property
    def trap(self) -> float:
        """Misleading coverage c*m: fraction of decision points both retrieved and misleading."""
        return self.c * self.m

    @property
    def benign(self) -> float:
        """Benign coverage c*(1-m): fraction retrieved but pointing the right way."""
        return self.c * (1.0 - self.m)


def score_candidate(v4: Trajectory, cand: Trajectory, grid: Grid,
                    walkable: Set[Cell], ref: Dict[Pose, Cell]) -> Score:
    phi_t = v4.view_set()
    if not phi_t:
        return Score(0.0, 0.0, 0, 0)

    shared = phi_t & cand.view_set()
    cand_action = cand.first_action_at()
    n_bad = n_judged = 0
    for q in shared:
        a = cand_action.get(q)
        if a is None:
            continue
        k = classify(grid, q, a, ref[q], walkable)
        if k == NEUTRAL:
            continue
        n_judged += 1
        n_bad += (k == REGRESSING)

    return Score(c=len(shared) / len(phi_t),
                 m=(n_bad / n_judged) if n_judged else 0.0,
                 n_shared=len(shared), n_judged=n_judged)


def trap_points(v4: Trajectory, v0: Trajectory, hards: Sequence[Trajectory],
                grid: Grid, walkable: Set[Cell], ref: Dict[Pose, Cell]) -> List[dict]:
    """
    Trap points = poses where V0 maintains progress and at least one H points wrong.
    Locations are by design and known, so TPA / TFR can be computed offline from rollout logs.
    """
    v0_action = v0.first_action_at()
    h_actions = [(h.episode_id, h.first_action_at()) for h in hards]

    out = []
    for q in v4.view_set():
        a0 = v0_action.get(q)
        if a0 is None or classify(grid, q, a0, ref[q], walkable) != PRESERVING:
            continue
        wrong = {hid: amap[q] for hid, amap in h_actions
                 if q in amap and classify(grid, q, amap[q], ref[q], walkable) == REGRESSING}
        if wrong:
            out.append({"pose": list(q), "goal": list(ref[q]),
                        "correct_action": a0, "distractor_actions": wrong})
    return out


# ============================================================================
# 5. Selection rules (pure ranking)
# ============================================================================


def satisfies_target_goal(cand: Trajectory, target: Trajectory) -> bool:
    """
    Would following cand to completion also satisfy the target case's success condition?
    If so, it is not a distractor but a second demonstration.
    Must check by success condition: T02/T04 are category searches, so another instance
    of the same category in a different case could also succeed.
    """
    return bool(set(cand.required) & set(target.required)) or \
           bool(set(cand.categories) & set(target.categories))


def eligible(target: Trajectory, scored: List[Tuple[Trajectory, Score]],
             v0_len: int, window: Optional[float]) -> List[Tuple[Trajectory, Score]]:
    lo, hi = (0.0, 10 ** 9) if window is None else (1 - window, 1 + window)
    return [(tr, sc) for tr, sc in scored
            if tr.case_id != target.case_id
            and not satisfies_target_goal(tr, target)
            and lo <= tr.length() / max(v0_len, 1) <= hi]


def _take(ranked: List[Tuple[Trajectory, Score]], n: int, used_cases: Set[str]) -> list:
    out = []
    for tr, sc in ranked:
        if tr.case_id in used_cases:
            continue
        out.append((tr, sc))
        used_cases.add(tr.case_id)
        if len(out) == n:
            break
    return out


def select_roles(pool: List[Tuple[Trajectory, Score]]) -> Dict[str, list]:
    """
    Pure ranking, no thresholds, no exception branches.

    * Each source case contributes at most one. After expanding the pool to V0-V3
      this is essential: V0/V1/V2 of the same case follow roughly the same route
      with nearly identical c*m; without deduplication H1/H2/H3 would likely be
      three variants of one case -- "3 distractor trajectories" becomes
      "1 distractor repeated 3 times", collapsing both bank diversity and the
      "related density 1/4" claim.
    """
    used: Set[str] = set()
    H = _take(sorted(pool, key=lambda p: -p[1].trap), N_HARD, used)
    E = _take(sorted(pool, key=lambda p: p[1].c), N_EASY, used)
    L = _take(sorted(pool, key=lambda p: -p[1].benign), N_LOOK, used)
    return {"H": H, "E": E, "L": L}


def choose_length_window(targets, scored_by_case, v0_len_by_case) -> Optional[float]:
    """
    Auto-select the tightest feasible length window: start at +/-10% and relax
    stepwise until every case can fill H=4 and E=3. The operator does not choose;
    the data chooses; the selected window goes into the paper appendix.
    """
    for w in LENGTH_WINDOWS:
        if all(
            len({tr.case_id for tr, _ in
                 eligible(t, scored_by_case[t.case_id], v0_len_by_case[t.case_id], w)})
            >= N_HARD + N_EASY
            for t in targets
        ):
            return w
    return None


# ============================================================================
# 6. Main pipeline
# ============================================================================


def build(trajectories: Dict[str, Trajectory], out_dir: Path,
          walkable: Optional[Set[Cell]] = None, quiet: bool = False) -> dict:
    validate(trajectories)
    walkable = walkable or walkable_from_trajectories(trajectories.values())
    grid = Grid(walkable)

    by_cv = {(t.case_id, t.variant): t for t in trajectories.values()}
    case_ids = sorted({t.case_id for t in trajectories.values()})
    candidates = [t for t in trajectories.values() if t.variant in MEMORY_ELIGIBLE]

    targets, scored_by_case, v0_len = [], {}, {}
    for cid in case_ids:
        v4, v0 = by_cv.get((cid, "V4")), by_cv.get((cid, "V0"))
        if v4 is None or v0 is None:
            if not quiet:
                print(f"[skip] {cid}: missing V0 or V4")
            continue
        ref = progress_reference(v4)
        scored_by_case[cid] = [(c, score_candidate(v4, c, grid, walkable, ref))
                               for c in candidates if c.case_id != cid]
        v0_len[cid] = v0.length()
        targets.append(v4)

    window = choose_length_window(targets, scored_by_case, v0_len)

    bank, traps, rows = {}, {}, []
    for k, v4 in enumerate(targets, start=1):
        cid = v4.case_id
        v0 = by_cv[(cid, "V0")]
        ref = progress_reference(v4)
        pool = eligible(v4, scored_by_case[cid], v0_len[cid], window)
        roles = select_roles(pool)

        traps[cid] = trap_points(v4, v0, [tr for tr, _ in roles["H"][:3]],
                                 grid, walkable, ref)
        bank[cid] = {
            "global_index": k, "v0_slot": slot_of_v0(k),
            "pool_size": len(pool), "n_trap_points": len(traps[cid]),
            "roles": {
                role: [{"episode_id": tr.episode_id, "task_id": tr.task_id, "rank": i + 1,
                        "c": round(sc.c, 4), "m": round(sc.m, 4),
                        "trap": round(sc.trap, 4), "benign": round(sc.benign, 4),
                        "n_shared": sc.n_shared, "length": tr.length()}
                       for i, (tr, sc) in enumerate(items)]
                for role, items in roles.items()},
        }
        for role, items in roles.items():
            for i, (tr, sc) in enumerate(items):
                rows.append({"target_case": cid, "role": role, "rank": i + 1,
                             "episode_id": tr.episode_id, "c": sc.c, "m": sc.m,
                             "trap": sc.trap, "benign": sc.benign,
                             "n_shared": sc.n_shared, "n_judged": sc.n_judged,
                             "length": tr.length()})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rq3_bank.json").write_text(
        json.dumps({"length_window": window, "bank_size": BANK_SIZE, "cases": bank},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "rq3_trap_points.json").write_text(
        json.dumps(traps, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (out_dir / "rq3_bank_stats.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    report = _report(bank, traps, rows, scored_by_case, window, quiet)
    return {"bank": bank, "traps": traps, "rows": rows,
            "window": window, "ok": report}


def slot_of_v0(global_case_index: int, rotate: bool = False) -> int:
    """
    rotate=False (default): always slot 1, aligned with the completed D-R (B03/B05).
        The cost is no positional balance: if the model tends to read the first
        reference more, distractor harm is **underestimated** (conservative direction).
        Confirm what order B03/B05 actually used before deciding.
    rotate=True: the original execution version's rotation scheme; only used for
        the Tier 6.5 positional control row.
    """
    return ((global_case_index - 1) % 4) + 1 if rotate else 1


def _report(bank, traps, rows, scored_by_case, window, quiet=False) -> bool:
    """Print a health report immediately after the run. Returns True if the bank is usable."""
    def out(*a):
        if not quiet:
            print(*a)

    if not rows:
        out("No candidates produced. Check the trajectory directory and variant field.")
        return False

    def avg(role, key):
        vs = [r[key] for r in rows if r["role"] == role]
        return sum(vs) / len(vs) if vs else 0.0

    n_trap = sum(len(v) for v in traps.values())
    pools = sorted(b["pool_size"] for b in bank.values())
    wtxt = "no length filter" if window is None else f"+/-{window:.0%}"
    out(f"\ncases = {len(bank)}    auto-selected length window = {wtxt}    "
        f"median candidate pool = {pools[len(pools)//2]}")
    vc = {}
    for r in rows:
        vc[r["episode_id"].rsplit("-", 1)[-1]] = vc.get(r["episode_id"].rsplit("-", 1)[-1], 0) + 1
    out(f"Source variant distribution of selected trajectories: {dict(sorted(vc.items()))}"
        f" (V0 only means V1-V3 were not included)")
    out(f"Trap points = {n_trap}  (mean {n_trap/max(len(bank),1):.1f} / case)")
    out(f"\n{'role':<5}{'count':>6}{'c':>8}{'m':>8}{'trap':>8}")
    for role in ("H", "E", "L"):
        n = len([r for r in rows if r["role"] == role])
        out(f"{role:<5}{n:>6}{avg(role,'c'):>8.3f}{avg(role,'m'):>8.3f}{avg(role,'trap'):>8.3f}")

    # Where H's c ranks among all candidates -- descriptive, not a threshold
    allc = sorted(sc.c for lst in scored_by_case.values() for _, sc in lst)
    hc = avg("H", "c")
    pct = 100.0 * sum(1 for v in allc if v <= hc) / max(len(allc), 1)
    out(f"\nH mean c = {hc:.3f}, at the {pct:.0f}th percentile of all candidate c distribution"
        f" (overall median {allc[len(allc)//2]:.3f})")

    ok = True
    need = {"H": N_HARD, "E": N_EASY}
    for r, n in need.items():
        cids = [cid for cid, b in bank.items() if len(b["roles"][r]) < n]
        if cids:
            ok = False
            out(f"\n! {len(cids)} cases have fewer than {n} {r} entries (e.g.: {cids[:3]}). "
                f"Length window already relaxed to the limit and still insufficient, "
                f"meaning the total candidate pool is too small -- check whether only some V0 were loaded.")

    # The next two are **empirical advisories**, they do not affect bank content, not hyperparameters
    if ok and hc - avg("E", "c") < 0.15:
        ok = False
        out("\n! [Advisory, empirical] H and E c differ by less than 0.15 -- routes in this "
            "warehouse barely overlap between cases anyway, so distractor effects will be weak. "
            "Send this number back before deciding whether to run the 4 lines.")
    if ok and avg("H", "trap") < 0.05:
        ok = False
        out("\n! [Advisory, empirical] H trap below 0.05 -- no genuinely 'misleading' distractors "
            "exist in the data. Fallback: use c ranking for top-3 (test dilution only, not "
            "misleading), and state this honestly in the paper.")
    if ok:
        out("\nOK bank is usable. Send back the three output files together with this report."
            "\n  Also recommended: sample 5 cases and plot H routes alongside the target V4 "
            "route on the same map for a quick look --"
            "\n  this is the only suggested human review step, and it is spot-check only, "
            "not a per-item audit.")
    return ok


# ============================================================================
# 7. Self-test: verify the script works without a dataset
# ============================================================================


def _selftest() -> int:
    """
    Build a 10x10 toy warehouse (two shelf rows, three north-south corridors),
    8 cases, run end to end. Run this first during handoff: only connect real
    data after seeing SELFTEST PASS.
    """
    random.seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="rq3_selftest_"))
    src, dst = tmp / "traj", tmp / "out"
    src.mkdir(parents=True)

    shelves = {(x, y) for x in (2, 3, 6, 7) for y in range(2, 8)}

    def route(a, b):
        (x0, y0), (x1, y1) = a, b
        path, cur = [a], list(a)
        for _ in range(y0):
            cur[1] -= 1; path.append(tuple(cur))
        while cur[0] != x1:
            cur[0] += 1 if x1 > cur[0] else -1; path.append(tuple(cur))
        while cur[1] != y1:
            cur[1] += 1 if y1 > cur[1] else -1; path.append(tuple(cur))
        return path

    def to_steps(cells):
        out = []
        for i, c in enumerate(cells):
            if i + 1 < len(cells):
                d = (cells[i+1][0] - c[0], cells[i+1][1] - c[1])
                a = {(0, 1): "Move_Forward", (0, -1): "Move_Backward",
                     (-1, 0): "Move_Left", (1, 0): "Move_Right"}[d]
            else:
                a = "Turn_Left"
            out.append({"pose": [c[0], c[1], "N"], "action": a,
                        "milestone": i == len(cells) - 1})
        return out

    goals = [(9, 8), (0, 8), (9, 2), (0, 2), (5, 9), (5, 0), (1, 9), (8, 9)]
    starts = [(0, 5), (9, 5), (1, 1), (8, 1), (0, 0), (9, 9), (4, 4), (4, 5)]
    for j, (g, st) in enumerate(zip(goals, starts)):
        cid = f"C{j:02d}"
        # V1-V3 use adjacent start poses to simulate variants of the same case,
        # to verify "at most one per source case"
        alts = [(max(0, s0[0] - 1), s0[1]) for s0 in (st,)] + \
               [(st[0], max(0, st[1] - 1)), (min(9, st[0] + 1), st[1])]
        variants = [("V0", st), ("V4", (4, 9) if j % 2 else (4, 0))] + \
                   [(f"V{i+1}", a) for i, a in enumerate(alts)]
        for var, s in variants:
            d = {"episode_id": f"MB-WH-T01-{cid}-{var}", "task_id": "T01",
                 "case_id": f"T01-{cid}", "variant": var,
                 "required": [f"obj_{cid}"], "categories": [f"cat_{cid}"],
                 "steps": to_steps(route(s, g))}
            (src / f"{d['episode_id']}.json").write_text(json.dumps(d), encoding="utf-8")

    trajs = {}
    for p in sorted(src.glob("*.json")):
        t = load_trajectory(p)
        trajs[t.episode_id] = t

    print(f"Self-test: generated {len(trajs)} trajectories (8 cases x V0-V4)")
    res = build(trajs, dst)

    checks = [
        ("Trajectories loaded", len(trajs) == 40),   # 8 cases x (V0,V4,V1,V2,V3)
        ("Every case has a bank", len(res["bank"]) == 8),
        ("Trap points produced", sum(len(v) for v in res["traps"].values()) > 0),
        ("H c clearly higher than E", (
            sum(r["c"] for r in res["rows"] if r["role"] == "H") / max(sum(1 for r in res["rows"] if r["role"] == "H"), 1)
            - sum(r["c"] for r in res["rows"] if r["role"] == "E") / max(sum(1 for r in res["rows"] if r["role"] == "E"), 1)
        ) > 0.1),
        ("All three files written", all((dst / f).exists() for f in
                                  ("rq3_bank.json", "rq3_trap_points.json", "rq3_bank_stats.csv"))),
        ("No two entries from same source case in H", all(
            len({e["episode_id"].rsplit("-", 1)[0] for e in b["roles"]["H"]}) == len(b["roles"]["H"])
            for b in res["bank"].values())),
        ("Candidate pool includes V1-V3", any(r["episode_id"].rsplit("-", 1)[-1] != "V0" for r in res["rows"])),
    ]
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    shutil.rmtree(tmp, ignore_errors=True)

    if all(p for _, p in checks):
        print("\nSELFTEST PASS -- the script works, ready to connect real data."
              "\n(The self-test uses a toy warehouse with 8 cases; the numbers themselves"
              " are meaningless, only the PASS/FAIL matters."
              "\n  If advisories appear on real data, follow handoff doc section 3,"
              " do not decide on your own.)")
        return 0
    print("\nSELFTEST FAIL -- do not connect real data yet; send back the FAIL items above.")
    return 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(0)
    src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    trajs = {}
    for p in sorted(src_dir.rglob("*.json")):
        try:
            t = load_trajectory(p)
            trajs[t.episode_id] = t
        except Exception as e:                       # noqa: BLE001
            print(f"[skip] {p.name}: {e}")
    print(f"Loaded {len(trajs)} trajectories")
    res = build(trajs, out_dir)
    sys.exit(0 if res["ok"] else 2)
