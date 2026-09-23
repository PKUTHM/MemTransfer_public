"""EPISODIC MEMORY PROMPT — anchor the agent at the closest point in the
expert's reference trajectory.

When the agent's current first-person view closely matches one of the
expert's recorded views in the reference trajectory, surface the
immediate few steps that the expert took from there. This grounds the
agent in a known-similar moment without dumping the entire trajectory.

Naming and indexing follow the project convention: ``image_paths[i]``
is the view at pose ``poses[i]`` (before action ``i`` for ``i < N``,
final view for ``i == N``); ``actions[i]`` is the expert's action from
``poses[i]`` to ``poses[i+1]``. In the rendered prompt we use the
same 1-based Step labels used by ``reference_traj_images.py``
(``Step idx+1``).

Image budget: this section returns at most ``1 + max_actions`` images
(matched view + one resulting view per action). With the defaults
(``max_actions=6``) that's 7 images, well below the prompt budget.
"""
import os
import re

from ._helpers import image_block
from .memory_sources import origin_matches


_FILENAME_RE = re.compile(
    r"Grid_X(-?\d+)_Y(-?\d+)_(East|North|West|South)\.png$"
)


def _parse_current_pose_from_image(image_path):
    """Extract ``(x, y, heading)`` from a ``Grid_*`` PNG filename.

    Returns ``None`` if the filename doesn't match the expected format.
    """
    if not image_path:
        return None
    base = os.path.basename(image_path)
    m = _FILENAME_RE.match(base)
    if not m:
        return None
    return {
        "x": int(m.group(1)),
        "y": int(m.group(2)),
        "heading": m.group(3),
    }


def stage_progress_by_step(traj):
    """Cumulative completed-stage count at each trajectory step.

    Given a reference trajectory ``traj`` (a dict with ``stage_events`` and
    ``length``), returns an array ``prog`` of length ``length+1`` such that
    ``prog[k]`` = number of stages the expert had completed BY the time
    they finished step ``k`` (counting ``k=0`` as "haven't taken any
    action yet" and ``k=length`` as "after the last action").

    ``stage_events[i].step`` already encodes the moment each stage was
    completed — see README "Conventions" for how to read it.
    """
    se = sorted(traj.get("stage_events") or [], key=lambda e: e["step"])
    L = int(traj.get("length", 0))
    prog = []
    cum, ei = 0, 0
    for k in range(L + 1):
        while ei < len(se) and se[ei]["step"] <= k:
            cum += 1
            ei += 1
        prog.append(cum)
    return prog


def _steps_into_current_stage(prog):
    """Given ``prog`` from :func:`stage_progress_by_step`, return a parallel
    array ``steps[k]`` = how many steps the agent has taken INTO the
    current stage at trajectory index ``k``. ``k=0`` is always 0 (no
    steps in any stage yet). Subsequent entries reset to 0 at every
    stage boundary.
    """
    L = len(prog)
    out = [0] * L
    last_advance = 0
    for k in range(1, L):
        if prog[k] != prog[k - 1]:
            last_advance = k
        out[k] = k - last_advance
    return out


def _pick_from_ties(indices, stage_progress=None,
                    agent_stages=0, agent_steps_in_stage=0):
    """Pick one index from ``indices`` (a list of equally-good matches)
    using two stage-aware tiebreakers, in priority order:

      1. smallest ``|stage_progress[i] - agent_stages|``
      2. smallest ``|steps_into_current_stage[i] - agent_steps_in_stage|``
      3. (final tiebreaker) smallest index — stable order

    If ``stage_progress`` is ``None``, falls back to the smallest index.
    """
    if not indices:
        return None
    if len(indices) == 1:
        return indices[0]
    if stage_progress is None:
        return min(indices)
    steps_at = _steps_into_current_stage(stage_progress)
    def key(i):
        return (
            abs(stage_progress[i] - agent_stages),
            abs(steps_at[i] - agent_steps_in_stage),
            i,
        )
    return min(indices, key=key)


def _find_matched_index(poses, target, stage_progress=None,
                        agent_stages=0, agent_steps_in_stage=0):
    """Find the index in ``poses`` whose pose best matches ``target``.

    Returns a 3-tuple ``(index, kind, distance)``:
      - ``index``     : int  : chosen pose index
      - ``kind``      : str  : ``"exact"`` (cell+heading match),
                              ``"cell"`` (same cell, view differs),
                              ``"nearest"`` (only closest reachable pose).
      - ``distance``  : float: metres from target. ``0`` for exact/cell.

    Tiebreaker: when *multiple* candidates fall in the same tier (e.g.
    the expert visited the same cell twice with the same heading, or
    the nearest reachable cell ties with several others), apply
    :func:`_pick_from_ties` with ``stage_progress`` and the agent's
    current stage progress. This breaks ties in favour of the visit
    whose stage progress matches the agent's actual stage progress
    most closely (and, as a secondary tiebreaker, whose position
    within the current stage is most similar).

    Returns ``None`` when there is no usable pose in the trajectory.
    """
    if not poses or target is None:
        return None
    tx, ty, th = target["x"], target["y"], target["heading"]

    exact_matches = []   # all indices i where pose[i] == target exactly
    cell_matches = []    # all indices i with same cell, different heading
    candidates = []
    for i, p in enumerate(poses):
        if not p or len(p) < 3:
            continue
        px, py, ph = p[0], p[1], p[2]
        if px == tx and py == ty and ph == th:
            exact_matches.append(i)
        elif px == tx and py == ty:
            cell_matches.append(i)
        d2 = (px - tx) ** 2 + (py - ty) ** 2
        candidates.append((d2, i))

    if exact_matches:
        idx = _pick_from_ties(exact_matches, stage_progress,
                              agent_stages, agent_steps_in_stage)
        return (idx, "exact", 0.0)
    if cell_matches:
        idx = _pick_from_ties(cell_matches, stage_progress,
                              agent_stages, agent_steps_in_stage)
        return (idx, "cell", 0.0)
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    # Nearest tier: collect every candidate at the minimum squared
    # distance (within a tiny epsilon to handle floating-point ties).
    best_d2 = candidates[0][0]
    tied_near = [i for d2, i in candidates if d2 <= best_d2 + 1e-9]
    idx = _pick_from_ties(tied_near, stage_progress,
                          agent_stages, agent_steps_in_stage)
    return (idx, "nearest", float(best_d2) ** 0.5)


def _format_pose(pose_or_target):
    """Render a pose (3-tuple or ``{x, y, heading}`` dict) as a string."""
    if pose_or_target is None:
        return "(unknown pose)"
    if isinstance(pose_or_target, dict):
        x, y, h = pose_or_target.get("x"), pose_or_target.get("y"), pose_or_target.get("heading")
    else:
        x, y, h = pose_or_target[0], pose_or_target[1], pose_or_target[2]
    return f"({x}, {y}) facing {h}"


def _format_vit_label(kind, idx, score, traj_total_len, poses):
    """Three-tier ViT wording that mirrors the pose matcher's prior-run style.

    ``kind`` is one of ``"exact"`` / ``"near"`` / ``"far"`` (from
    :func:`vit_matcher.classify_cosine`).
    """
    matched_pose = poses[idx] if 0 <= idx < len(poses) else None
    pose_str = _format_pose(matched_pose)
    if kind == "exact":
        return (
            f"the prior successful run includes a view essentially identical "
            f"to yours at step {idx + 1} ({pose_str}, cosine ≈ {score:.3f})"
        )
    if kind == "near":
        return (
            f"no view in the prior run is identical to yours, but step "
            f"{idx + 1}'s view at {pose_str} is the closest semantically similar "
            f"frame (cosine ≈ {score:.3f})"
        )
    # far
    return (
        f"no good visual match for your current view anywhere in the prior "
        f"run; the closest view sits at step {idx + 1} ({pose_str}, "
        f"cosine ≈ {score:.3f}, low similarity)"
    )


def episodic_memory_blocks(image_path, reference_traj, max_actions=6, matcher="pose",
                          agent_stages=0, agent_steps_in_current_stage=0,
                          content="all"):
    """Return a list of content blocks for the episodic-memory section.

    Parameters
    ----------
    image_path : str
        Path to the agent's current first-person view (a ``Grid_X*_Y*_*.png``
        file). Used to anchor the matched step in the reference trajectory.
    reference_traj : dict or None
        The reference trajectory dict (must have ``actions``, ``poses``,
        ``image_paths``). When ``None`` or empty, returns ``[]``.
    max_actions : int
        Maximum number of subsequent expert actions to render. Defaults
        to 6 (the user-specified "next 6 actions" horizon).
    matcher : str
        ``"pose"`` (default) — match by filename/pose (fast, no torch).
        ``"vit"``   — match by ViT-B/16 cosine similarity over a precomputed
                     embedding table; falls back to ``pose`` matching if
                     the .npz table isn't available.
    agent_stages : int
        How many stages the agent has actually completed so far. Used
        ONLY as a tiebreaker when multiple reference trajectory steps
        score identically on the primary similarity metric (e.g. the
        expert revisits the same cell twice). Defaults to 0.
    agent_steps_in_current_stage : int
        How many steps the agent has taken INTO the current (incomplete)
        stage. Secondary tiebreaker — used only when the primary stage
        tiebreaker above is also tied. Defaults to 0.
    content : str
        What to include after the matched view image:
        ``"all"`` (default) — actions + resulting images
        ``"actions"`` — only action text, no resulting images
        ``"images"`` — only resulting images, no action text

    Returns
    -------
    list[dict]
        Content blocks (text + image_url). ``[]`` if the section should
        be skipped (no trajectory, no usable poses, etc.).
    """
    if not reference_traj or not reference_traj.get("actions"):
        return []
    actions = reference_traj["actions"]
    image_paths = reference_traj.get("image_paths", [])
    poses = reference_traj.get("poses", [])

    # Pre-compute the stage progress array (cumulative stages done at
    # each step) once per call, so the matcher and any tiebreaker can
    # reason in stage-terms cheaply.
    stage_progress = stage_progress_by_step(reference_traj)

    # ---- Resolve the matched index via the chosen matcher ----
    match_info = None  # (idx, label_for_prompt, kind_label)
    if matcher == "vit":
        try:
            from . import vit_matcher
            if vit_matcher.is_ready():
                vi, score = vit_matcher.find_similar_index(
                    image_path, image_paths, return_score=True,
                    stage_progress=stage_progress,
                    agent_stages=agent_stages,
                    agent_steps_in_stage=agent_steps_in_current_stage,
                )
                if vi is not None and 0 <= vi < len(actions):
                    vit_kind = vit_matcher.classify_cosine(score)
                    match_info = (vi, _format_vit_label(vit_kind, vi, score, len(actions), poses), vit_kind)
        except Exception:
            match_info = None

    if match_info is None:
        # Fall back to (or default) pose-based matching.
        target = _parse_current_pose_from_image(image_path)
        if target is None:
            return []
        match = _find_matched_index(
            poses, target,
            stage_progress=stage_progress,
            agent_stages=agent_stages,
            agent_steps_in_stage=agent_steps_in_current_stage,
        )
        if match is None:
            return []
        idx, kind, dist = match
        if idx >= len(image_paths) and idx >= len(actions):
            return []
        if kind == "exact":
            label = (
                f"the prior successful run's step {idx + 1} at "
                f"{_format_pose(poses[idx])} is an exact cell+heading match"
            )
        elif kind == "cell":
            label = (
                f"the prior successful run passed through your cell "
                f"({target['x']}, {target['y']}) facing "
                f"{poses[idx][2]} instead of your "
                f"{target['heading']} — step {idx + 1} "
                f"({_format_pose(poses[idx])}) is that visit"
            )
        else:
            label = (
                f"no exact or same-cell match for your current view in the "
                f"prior run; the closest similar view sits at "
                f"{_format_pose(poses[idx])} (distance ≈ {dist:.2f}m)"
            )
        match_info = (idx, label, kind)

    idx, match_label, kind = match_info

    available_actions = max(0, len(actions) - idx)
    n_actions = min(max_actions, available_actions)

    target = _parse_current_pose_from_image(image_path) if matcher != "vit" \
        else _format_pose(poses[idx] if 0 <= idx < len(poses) else None)

    # Note about the trajectory: this section always uses the current
    # case's OWN forward_only trajectory (loaded by run_case via
    # get_trajectory_for_case, which filters on `reachable == True`),
    # so it represents a previously successful run of THIS same case.
    traj_total_len = len(actions)

    blocks = [{
        "type": "text",
        "text": (
            "\n[EPISODIC MEMORY — prior-successful run of THIS same case]\n"
            "The following is an excerpt from a trajectory that previously "
            "completed THIS same case successfully (a prior agent ran this "
            "exact task to completion; below we surface the part of its "
            f"run that most resembles your current state, step {idx + 1} of "
            f"its {traj_total_len}-step solution ({match_label})).\n"
            "The image below is what that previously-successful agent saw at "
            "that step. The next entries are up to "
            f"{n_actions} of that agent's subsequent actions, together with "
            "the views that resulted from executing each action.\n"
        ),
    }]

    img_path0 = image_paths[idx] if 0 <= idx < len(image_paths) else ""
    blocks.append({
        "type": "text",
        "text": (
            f"[matched view (prior successful run) | step {idx + 1}/{traj_total_len}"
            f" | at {_format_pose(poses[idx])}]\n"
        ),
    })
    blk = image_block(img_path0)
    if blk is not None:
        blocks.append(blk)
    else:
        blocks.append({"type": "text", "text": "  [matched image unavailable]\n"})

    for k in range(n_actions):
        act = actions[idx + k]
        next_img_idx = idx + k + 1
        next_img_path = image_paths[next_img_idx] if 0 <= next_img_idx < len(image_paths) else ""
        if content in ("all", "actions"):
            blocks.append({
                "type": "text",
                "text": f"  → prior-successful agent did (step {idx + 1 + k}): {act}\n",
            })
        if content in ("all", "images"):
            if content == "images":
                blocks.append({
                    "type": "text",
                    "text": f"  [view after step {idx + 1 + k}]\n",
                })
            nxt_blk = image_block(next_img_path)
            if nxt_blk is not None:
                blocks.append(nxt_blk)
            else:
                blocks.append({"type": "text", "text": "  [resulting image unavailable]\n"})

    if available_actions > n_actions:
        blocks.append({
            "type": "text",
            "text": (
                f"({available_actions - n_actions} more steps from the prior "
                "successful run remain truncated.)\n"
            ),
        })
    return blocks


# ==========================================================================
# EXPERIMENT MODE — retrieval across several historical versions
# ==========================================================================
# Everything below is used by the v0-v3 -> v4 study. The single-version
# path above is left untouched so legacy v0 runs stay bit-identical.
#
# Contract (plan section 6.6): the candidate pool grows to N trajectories,
# but exactly ONE contiguous excerpt is returned, taken from whichever
# single trajectory owns the winning pose. Excerpts never span two
# trajectories, and a short tail is simply short - we never swap to
# another candidate just to obtain a longer excerpt.

#: Cross-version tie-break order for otherwise identical matches.
VERSION_PRIORITY = {"v0": 0, "v1": 1, "v2": 2, "v3": 3}

_TIER_RANK = {"exact": 0, "cell": 1, "nearest": 2}


class DecisionContext(object):
    """Per-decision state. Explicitly passed, never module-global.

    The legacy implementation stashed stage progress in module globals,
    which raced across the thread pool's concurrent cases. One instance
    per decision keeps every case's retrieval independent.
    """

    __slots__ = ("case_label", "initial_origin", "current_pose",
                 "decision_index", "stages_completed", "steps_in_current_stage")

    def __init__(self, case_label=None, initial_origin=None, current_pose=None,
                 decision_index=0, stages_completed=0, steps_in_current_stage=0):
        self.case_label = case_label
        self.initial_origin = initial_origin
        self.current_pose = current_pose
        self.decision_index = decision_index
        self.stages_completed = stages_completed
        self.steps_in_current_stage = steps_in_current_stage


class MatchResult(object):
    """Which historical pose won, and why."""

    __slots__ = ("version", "case_label", "trajectory_source", "traj_pos",
                 "matched_index", "match_kind", "distance",
                 "historical_stage_progress", "stage_difference",
                 "steps_in_stage_difference", "candidate_pose_count",
                 # ViT retrieval only. ``matcher`` records what actually ran,
                 # not what was asked for, so a fallback to pose shows up in
                 # telemetry instead of looking like a ViT result.
                 "cosine", "matcher")

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))

    def as_dict(self):
        return {n: getattr(self, n) for n in self.__slots__}


def find_best_match(trajectories, target, ctx):
    """Pick one pose out of every pose in ``trajectories``.

    Selection order, applied globally rather than per trajectory:

      1. match tier - exact (cell+heading) > same cell > nearest cell
      2. within ``nearest``: smallest Euclidean distance (1e-9 tolerance)
      3. smallest |historical stage progress - agent stage progress|
      4. smallest |historical steps-into-stage - agent steps-into-stage|
      5. version priority v0 < v1 < v2 < v3
      6. smallest step index inside that trajectory

    Returns a :class:`MatchResult`, or ``None`` when no trajectory has a
    usable pose.
    """
    if not trajectories or target is None:
        return None
    tx, ty, th = target["x"], target["y"], target["heading"]

    candidates = []
    for pos, traj in enumerate(trajectories):
        poses = traj.get("poses") or []
        if not poses:
            continue
        prog = stage_progress_by_step(traj)
        steps_at = _steps_into_current_stage(prog)
        vrank = VERSION_PRIORITY.get(traj.get("version"), len(VERSION_PRIORITY))
        for i, p in enumerate(poses):
            if not p or len(p) < 3:
                continue
            px, py, ph = p[0], p[1], p[2]
            if px == tx and py == ty and ph == th:
                tier, d2 = "exact", 0.0
            elif px == tx and py == ty:
                tier, d2 = "cell", 0.0
            else:
                tier = "nearest"
                d2 = float((px - tx) ** 2 + (py - ty) ** 2)
            sp = prog[i] if i < len(prog) else (prog[-1] if prog else 0)
            st = steps_at[i] if i < len(steps_at) else 0
            candidates.append((
                _TIER_RANK[tier], d2,
                abs(sp - (ctx.stages_completed or 0)),
                abs(st - (ctx.steps_in_current_stage or 0)),
                vrank, i, pos, tier, sp,
            ))
    if not candidates:
        return None

    best_tier = min(c[0] for c in candidates)
    pool = [c for c in candidates if c[0] == best_tier]
    if best_tier == _TIER_RANK["nearest"]:
        best_d2 = min(c[1] for c in pool)
        pool = [c for c in pool if c[1] <= best_d2 + 1e-9]

    winner = min(pool, key=lambda c: (c[2], c[3], c[4], c[5]))
    _, d2, stage_diff, steps_diff, _vrank, idx, pos, tier, sp = winner
    traj = trajectories[pos]
    return MatchResult(
        version=traj.get("version"),
        case_label=traj.get("case_label"),
        trajectory_source=traj.get("source", "forward_only"),
        traj_pos=pos, matched_index=idx, match_kind=tier,
        distance=(float(d2) ** 0.5 if tier == "nearest" else 0.0),
        historical_stage_progress=sp, stage_difference=stage_diff,
        steps_in_stage_difference=steps_diff,
        candidate_pose_count=len(candidates),
        cosine=None, matcher="pose",
    )


def find_best_match_vit(trajectories, image_path, ctx):
    """Same selection, ranked by ViT cosine instead of grid geometry.

    Pooling works exactly as in :func:`find_best_match`: every frame of
    every candidate trajectory competes, and the winner's trajectory owns
    the excerpt. Only the primary key changes.

      1. highest cosine against the agent's current view
      2. within ``vit_matcher.COSINE_TIE_EPS`` of that best score:
      3. smallest |historical stage progress - agent stage progress|
      4. smallest |historical steps-into-stage - agent steps-into-stage|
      5. version priority v0 < v1 < v2 < v3
      6. smallest step index inside that trajectory

    Cosine is comparable across scenes -- the vectors share one space and
    only the *table lookup* is per-scene -- so a candidate frame rendered
    in another scene is still ranked correctly.

    Returns ``None`` when no table can be loaded or no candidate frame is
    in one; the caller is expected to fall back to the pose matcher and
    record that it did.
    """
    if not trajectories or not image_path:
        return None
    try:
        from . import vit_matcher
    except ImportError:
        return None

    cur = vit_matcher._vector(image_path)
    if cur is None:
        return None

    candidates = []
    for pos, traj in enumerate(trajectories):
        frames = traj.get("image_paths") or []
        if not frames:
            continue
        prog = stage_progress_by_step(traj)
        steps_at = _steps_into_current_stage(prog)
        vrank = VERSION_PRIORITY.get(traj.get("version"), len(VERSION_PRIORITY))
        for i, f in enumerate(frames):
            v = vit_matcher._vector(f)
            if v is None:
                continue
            score = float(v @ cur)
            sp = prog[i] if i < len(prog) else (prog[-1] if prog else 0)
            st = steps_at[i] if i < len(steps_at) else 0
            candidates.append((
                score,
                abs(sp - (ctx.stages_completed or 0)),
                abs(st - (ctx.steps_in_current_stage or 0)),
                vrank, i, pos, sp,
            ))
    if not candidates:
        return None

    best = max(c[0] for c in candidates)
    pool = [c for c in candidates
            if c[0] >= best - vit_matcher.COSINE_TIE_EPS]
    winner = min(pool, key=lambda c: (c[1], c[2], c[3], c[4]))
    score, stage_diff, steps_diff, _vrank, idx, pos, sp = winner
    traj = trajectories[pos]
    return MatchResult(
        version=traj.get("version"),
        case_label=traj.get("case_label"),
        trajectory_source=traj.get("source", "forward_only"),
        traj_pos=pos, matched_index=idx,
        match_kind=vit_matcher.classify_cosine(score),
        distance=None,
        historical_stage_progress=sp, stage_difference=stage_diff,
        steps_in_stage_difference=steps_diff,
        candidate_pose_count=len(candidates),
        cosine=score, matcher="vit",
    )


def _historical_match_label(match, poses, target):
    """Human-readable reason, mirroring the v0 phrasing but dropping the
    "prior successful run of THIS same case" claim, which is not true of a
    run that started somewhere else (and, in G05, in another version)."""
    idx = match.matched_index
    if match.matcher == "vit":
        # ``match_kind`` is the cosine tier here, and its "exact" means
        # "visually near-identical" -- NOT the pose matcher's "same cell
        # and heading". Dispatching on the matcher first keeps the two
        # vocabularies from being reported as each other.
        return _format_vit_label(match.match_kind, idx, match.cosine,
                                 len(poses), poses)
    if match.match_kind == "exact":
        return ("step %d at %s is an exact cell+heading match"
                % (idx + 1, _format_pose(poses[idx])))
    if match.match_kind == "cell":
        return ("the historical run passed through your cell (%s, %s) facing %s "
                "instead of your %s - step %d (%s) is that visit"
                % (target["x"], target["y"], poses[idx][2], target["heading"],
                   idx + 1, _format_pose(poses[idx])))
    return ("no exact or same-cell match for your current view; the closest "
            "similar view sits at %s (distance ~ %.2fm)"
            % (_format_pose(poses[idx]), match.distance or 0.0))


def episodic_memory_multi_blocks(image_path, trajectories, max_actions=6,
                                 ctx=None, content="all", match_sink=None,
                                 matcher="pose", same_origin_wording=False):
    """Episodic section for the experiment groups (G04 single, G05 multi).

    ``trajectories`` is an ordered list of prompt-shaped trajectory dicts
    (see ``experiment.HistoricalTrajectory.as_prompt_traj``), each
    carrying ``version`` and ``case_label``. Exactly one excerpt comes
    back, from exactly one of them.

    ``matcher`` selects how the winning frame is found: ``"pose"`` ranks by
    grid geometry, ``"vit"`` by ViT-B/16 cosine over the precomputed
    per-scene tables. The pooling contract is identical either way -- only
    the ranking key differs -- so G04 (one candidate trajectory) and G05
    (four) stay directly comparable under both.

    A ``"vit"`` request falls back to ``"pose"`` when no embedding table
    can be read. That is recorded in ``MatchResult.matcher`` rather than
    hidden, because a silent fallback would make a ViT group's numbers
    indistinguishable from a pose group's.

    The caller wraps the returned blocks in the provenance markers from
    :mod:`prompt.memory_sources`, so G04 and G05 render identically apart
    from which version won.
    """
    trajectories = [t for t in (trajectories or []) if t and t.get("actions")]
    if not trajectories:
        return []
    if ctx is None:
        ctx = DecisionContext()

    # Parsed up front: the pose matcher needs it to rank, and the label
    # needs it to describe a same-cell hit. Only the pose path *requires*
    # it, so a ViT match can still proceed when the name does not parse.
    target = _parse_current_pose_from_image(image_path)

    match = None
    if matcher == "vit":
        match = find_best_match_vit(trajectories, image_path, ctx)
    if match is None:
        if target is None:
            return []
        match = find_best_match(trajectories, target, ctx)
    if match is None:
        return []
    if match_sink is not None:
        match_sink.append(match)

    traj = trajectories[match.traj_pos]
    actions = traj["actions"]
    poses = traj.get("poses") or []
    image_paths = traj.get("image_paths") or []
    idx = match.matched_index
    if idx >= len(image_paths) and idx >= len(actions):
        return []

    traj_total_len = len(actions)
    available_actions = max(0, traj_total_len - idx)
    n_actions = min(max_actions, available_actions)
    label = _historical_match_label(match, poses, target)
    matched_pose = poses[idx] if idx < len(poses) else None

    # ``same_origin_wording``: a record that started at this run's own spawn
    # (cell and heading) says so; every other record keeps "its own origin",
    # which is also the only wording when the option is off.
    same = same_origin_wording and origin_matches(traj.get("origin"),
                                                  ctx.initial_origin)
    whose_origin = ("the same origin and initial heading as yours" if same
                    else "its own origin")
    blocks = [{
        "type": "text",
        "text": (
            "\n[EPISODIC MEMORY - RETRIEVED HISTORICAL EXCERPT]\n"
            "The following is an excerpt from a historical trajectory that "
            "completed this same case starting from %s (below we "
            "surface the part of that run that most resembles your current "
            "state, step %d of its %d-step record (%s)).\n"
            "The image below is what that historical agent saw at that step. "
            "The next entries are up to %d of that agent's subsequent actions, "
            "together with the views that resulted from executing each action.\n"
            % (whose_origin, idx + 1, traj_total_len, label, n_actions)
        ),
    }, {
        "type": "text",
        "text": ("[matched view | step %d/%d | at %s]\n"
                 % (idx + 1, traj_total_len, _format_pose(matched_pose))),
    }]

    img_path0 = image_paths[idx] if 0 <= idx < len(image_paths) else ""
    blk = image_block(img_path0)
    blocks.append(blk if blk is not None
                  else {"type": "text", "text": "  [matched image unavailable]\n"})

    for k in range(n_actions):
        act = actions[idx + k]
        next_img_idx = idx + k + 1
        next_img_path = image_paths[next_img_idx] if 0 <= next_img_idx < len(image_paths) else ""
        if content in ("all", "actions"):
            blocks.append({
                "type": "text",
                "text": "  -> the historical agent did (step %d): %s\n" % (idx + 1 + k, act),
            })
        if content in ("all", "images"):
            if content == "images":
                blocks.append({"type": "text",
                               "text": "  [view after step %d]\n" % (idx + 1 + k)})
            nxt_blk = image_block(next_img_path)
            blocks.append(nxt_blk if nxt_blk is not None
                          else {"type": "text", "text": "  [resulting image unavailable]\n"})

    if available_actions > n_actions:
        blocks.append({
            "type": "text",
            "text": ("(%d more steps from this historical trajectory remain "
                     "truncated.)\n" % (available_actions - n_actions)),
        })
    return blocks
