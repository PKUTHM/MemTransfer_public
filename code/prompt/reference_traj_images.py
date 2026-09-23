"""REFERENCE TRAJECTORY IMAGES PROMPT — selected key images interleaved with the expert's action list.

Mirrors the per-step layout used in ``history.py``::

    [start] at (1, 4) facing South — view:
    <IMG base64>
      → expert did: Turn_Left

    [sampled] at (2, 4) facing East — view:
    <IMG base64>
      → expert did: Move_Forward 1m

    [stage: reached_Banana] at (-3, -9) facing South — view:
    <IMG base64>
      → expert did: Turn_180

    [task_complete | returned_to_origin] at (-4, -9) facing West — final view:
    <IMG base64>

For ``all`` / ``interval`` / ``turn_only`` modes the rendering is the same
shape, just driven by a different subset of indices.
"""
from ._helpers import image_block


def _pose_str(pose):
    if not pose:
        return "(pose unknown)"
    x, y, h = pose[0], pose[1], pose[2]
    return f"({x}, {y}) facing {h}"


def _select_smart_indices(actions, stage_events, max_images,
                          layers=None, sample_count=None):
    """Pick which IMAGE indices to include in ``smart`` mode.

    Indices are **image indices** (``0..len(actions)``): image ``i`` for
    ``i < N`` is the view BEFORE action ``i``; image ``N`` is the final
    view (the pose after the last action).

    Parameters
    ----------
    max_images : int
        Hard upper bound on the total number of images returned.
    layers : set[str] or None
        Which priority layers to include. Valid layer names:
        ``"start"``, ``"stage"``, ``"end"``, ``"turn"``.
        If None, defaults to all four (full smart behaviour).
    sample_count : int or None
        Maximum number of uniform-sampled images to add after the
        priority layers. If None, fills up to ``max_images`` (original
        behaviour). Set to 0 to disable sampling entirely.

    Priority (per the user spec, applied in order for the enabled layers):
      1. Starting image (index 0)           — layer "start"
      2. Multi-stage images                 — layer "stage"
      3. Task-completion image (index N)    — layer "end"
      4. All turn images                    — layer "turn"
      5. Uniform sampling of remaining      — always last
    """
    if layers is None:
        layers = {"start", "stage", "end", "turn"}
    if sample_count is None:
        sample_count = max_images

    n_actions = len(actions)
    n_imgs = n_actions + 1
    # ``max_images`` is a HARD cap on the returned list, including the
    # priority layers. A non-positive budget yields nothing at all.
    if max_images <= 0 or n_imgs == 0:
        return []
    if n_imgs == 1:
        return [(0, "start")] if "start" in layers else []

    selected = {}

    def _take(idx, label):
        """Claim one budget slot for ``idx``; return False when full.

        An index already claimed keeps its slot and merely gains a label,
        so overlapping layers (a turn on the final frame, a stage event on
        the start frame) never consume two slots.
        """
        if idx in selected:
            return True
        if len(selected) >= max_images:
            return False
        selected[idx] = label
        return True

    if "start" in layers:
        _take(0, "start")

    if "stage" in layers:
        # Time order within the layer, so a budget cut-off drops the
        # LATEST stage points rather than an arbitrary subset.
        events = [ev for ev in (stage_events or [])
                  if isinstance(ev.get("step"), int) and 0 <= ev["step"] < n_imgs]
        for ev in sorted(events, key=lambda e: e["step"]):
            if not _take(ev["step"], f"stage: {ev.get('label', 'stage_reached')}"):
                break

    if "end" in layers:
        if n_actions in selected:
            selected[n_actions] = selected[n_actions] + " + task_complete"
        else:
            _take(n_actions, "task_complete")

    if "turn" in layers:
        for i, act in enumerate(actions):
            if act.startswith("Turn_") and i not in selected:
                if not _take(i, "turn"):
                    break

    used = set(selected)
    remaining = [i for i in range(n_imgs) if i not in used]
    # Never let sampling push the total past the hard cap.
    available = max(0, min(sample_count, max_images) - len(selected))
    if remaining and available > 0:
        if len(remaining) <= available:
            picked = remaining
        else:
            step = len(remaining) / available
            picked = [remaining[int(k * step)] for k in range(available)]
        for i in picked:
            selected[i] = "sampled"

    return sorted(selected.items())


def _render_view_action_block(idx, label, actions, image_paths, poses, n_actions):
    """Render a single [Step K | label | view | action] block.

    Step number = ``idx + 1`` (1-based, matching the history convention).
    For the final view (``idx == n_actions``, no action follows), the
    label uses ``"Final view:"`` instead of ``"Step N+1:"``.
    """
    pose = poses[idx] if 0 <= idx < len(poses) else None
    pos_str = _pose_str(pose)
    label_str = f"[{label}]" if label else "[view]"
    is_final = idx >= n_actions
    step_str = "Final view: " if is_final else f"Step {idx + 1}: "

    blocks = [{"type": "text",
               "text": f"{step_str}{label_str} at {pos_str} — view:\n"}]
    img_path = image_paths[idx] if 0 <= idx < len(image_paths) else ""
    blk = image_block(img_path)
    if blk is not None:
        blocks.append(blk)
    else:
        blocks.append({"type": "text", "text": "  [image unavailable]\n"})

    if is_final:
        blocks.append({"type": "text", "text": "  (trajectory ends here)\n"})
    else:
        blocks.append({"type": "text",
                       "text": f"  → expert did: {actions[idx]}\n"})
    return blocks


def reference_traj_images_blocks(traj, image_mode="none", interval=3,
                                 max_smart_images=20, smart_layers=None,
                                 smart_sample_count=None, selection_sink=None):
    """Return a list of content blocks: header + per-selected-index view+action.

    The caller is responsible for ensuring ``max_smart_images`` is the
    **remaining budget** (the prompt has a hard cap of
    ``MAX_PROMPT_IMAGES = 50`` total images across current + history +
    reference). If the smart selector picks more than allowed by the
    budget, the caller should pre-cap ``max_smart_images`` before calling.

    Parameters
    ----------
    smart_layers : set[str] or None
        Which priority layers to include in smart mode. Valid names:
        ``"start"``, ``"stage"``, ``"end"``, ``"turn"``.
        None means all four (default smart behaviour).
    smart_sample_count : int or None
        Maximum number of uniform-sampled images. None means fill up
        to ``max_smart_images``. Set to 0 to disable sampling.
    selection_sink : list or None
        When given, the chosen ``(index, label)`` pairs are appended to it
        so the caller can log exactly which frames it sent without having
        to re-derive the selection.
    """
    if not traj or not traj.get("actions"):
        return []
    actions = traj["actions"]
    image_paths = traj.get("image_paths", [])
    poses = traj.get("poses", [])
    n_actions = len(actions)
    n_imgs = len(image_paths) or (n_actions + 1)

    if image_mode == "smart":
        chosen = _select_smart_indices(
            actions, traj.get("stage_events"), max_smart_images,
            layers=smart_layers, sample_count=smart_sample_count,
        )
        chosen_set = {i for i, _ in chosen}
        label_of = {i: lbl for i, lbl in chosen}
    else:
        chosen_set = set()
        for idx, act in enumerate(actions):
            if image_mode == "all":
                chosen_set.add(idx)
            elif image_mode == "interval" and interval > 0 and (idx % interval == 0):
                chosen_set.add(idx)
            elif image_mode == "turn_only" and act.startswith("Turn_"):
                chosen_set.add(idx)
        label_of = {}

    blocks = []
    chosen_sorted = sorted(chosen_set)
    if selection_sink is not None:
        selection_sink.extend((i, label_of.get(i)) for i in chosen_sorted)
    if not chosen_sorted:
        return blocks
    blocks.append({"type": "text",
                   "text": "\n[HISTORICAL EXPERT REFERENCE TRAJECTORY — key views interleaved]\n"
                           f"({len(chosen_sorted)} of {n_imgs} images shown; "
                           "label tells why each frame was selected)\n"})

    for idx in chosen_sorted:
        blocks.extend(_render_view_action_block(
            idx, label_of.get(idx), actions, image_paths, poses, n_actions
        ))

    blocks.append({"type": "text",
                   "text": f"Total length: {traj.get('length', n_actions)} steps.\n"})
    return blocks
