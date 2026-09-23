"""REFERENCE TRAJECTORY PROMPT (text) — full action sequence of the expert.

Edit ``REFERENCE_TRAJ_TEMPLATE`` to change the wording. The caller passes
``traj`` as a dict with at least ``actions`` (list[str]) and ``length``
(int). ``traj`` may also carry ``source`` ("forward_only" / "all_actions")
and ``image_paths`` (list[str]); image paths are listed in the block
header but the actual images live in the *images* variant module.
"""
REFERENCE_TRAJ_TEMPLATE_TEXT_ONLY = (
    "\n[HISTORICAL EXPERT REFERENCE TRAJECTORY]\n"
    "Here is the complete action sequence an expert agent executed under "
    "identical environment dynamics:\n"
    "Trajectory Sequence: [{actions}]\n"
    "Optimal sequence length: {length} steps."
)

REFERENCE_TRAJ_TEMPLATE_WITH_IMAGES = (
    "\n[HISTORICAL EXPERT REFERENCE TRAJECTORY]\n"
    "Here is the complete action sequence an expert agent executed under "
    "identical environment dynamics:\n"
    "Trajectory Sequence: [{actions}]\n"
    "Optimal sequence length: {length} steps.\n"
    "Below, the same trajectory is replayed with selected key images "
    "interleaved so you can see the view the expert was looking at "
    "right before each highlighted action."
)

# Experiment-mode variants. Identical layout to the two templates above,
# with one deliberate wording change: under the v4 test the historical
# route was optimal for its OWN origin, not for the agent's current one,
# so calling it "optimal" would assert something false. Used by every
# Full Context group (single- and multi-version alike) so the two stay
# comparable.
HISTORICAL_TRAJ_TEMPLATE_TEXT_ONLY = (
    "\n[HISTORICAL EXPERT REFERENCE TRAJECTORY]\n"
    "Here is the complete action sequence an expert agent executed under "
    "identical environment dynamics:\n"
    "Trajectory Sequence: [{actions}]\n"
    "Historical trajectory length: {length} steps."
)

HISTORICAL_TRAJ_TEMPLATE_WITH_IMAGES = (
    "\n[HISTORICAL EXPERT REFERENCE TRAJECTORY]\n"
    "Here is the complete action sequence an expert agent executed under "
    "identical environment dynamics:\n"
    "Trajectory Sequence: [{actions}]\n"
    "Historical trajectory length: {length} steps.\n"
    "Below, the same trajectory is replayed with selected key images "
    "interleaved so you can see the view the expert was looking at "
    "right before each highlighted action."
)

PEER_REFERENCE_TRAJ_TEMPLATE_TEXT_ONLY = (
    "\n[PEER REFERENCE TRAJECTORY — from another case of the same task type]\n"
    "Here is the action sequence an expert agent executed to complete a "
    "DIFFERENT case of the SAME task type.\n"
    "Peer Case {source_case_id}:\n"
    "  Task: {description}\n"
    "  Start: ({origin_x}, {origin_y}) facing {origin_heading}\n"
    "Note: target locations and starting position may differ from your case.\n"
    "Trajectory Sequence: [{actions}]\n"
    "Sequence length: {length} steps."
)

PEER_REFERENCE_TRAJ_TEMPLATE_WITH_IMAGES = (
    "\n[PEER REFERENCE TRAJECTORY — from another case of the same task type]\n"
    "Here is the action sequence an expert agent executed to complete a "
    "DIFFERENT case of the SAME task type.\n"
    "Peer Case {source_case_id}:\n"
    "  Task: {description}\n"
    "  Start: ({origin_x}, {origin_y}) facing {origin_heading}\n"
    "Note: target locations and starting position may differ from your case.\n"
    "Trajectory Sequence: [{actions}]\n"
    "Sequence length: {length} steps.\n"
    "Below, the same trajectory is replayed with selected key images "
    "interleaved so you can see the view the expert was looking at "
    "right before each highlighted action."
)


def reference_traj_block(traj, has_images=False, historical=False):
    """Return a text content block for the reference trajectory, or '' block if disabled.

    Parameters
    ----------
    traj : dict or None
        If None/empty, returns an empty-string text block (callers can
        still call it unconditionally). Otherwise expected to have:
        - ``actions``     : list[str]
        - ``length``      : int
        - ``source``      : str (optional, e.g. "forward_only" / "all_actions")
    has_images : bool
        If True, include the text about interleaved images.
    historical : bool
        Experiment mode: describe the length as a *historical* trajectory
        length rather than an optimal one (see the template comment).
    """
    if not traj or not traj.get("actions"):
        return {"type": "text", "text": ""}
    if historical:
        template = HISTORICAL_TRAJ_TEMPLATE_WITH_IMAGES if has_images \
            else HISTORICAL_TRAJ_TEMPLATE_TEXT_ONLY
    else:
        template = REFERENCE_TRAJ_TEMPLATE_WITH_IMAGES if has_images else REFERENCE_TRAJ_TEMPLATE_TEXT_ONLY
    text = template.format(
        actions=", ".join(traj["actions"]),
        length=traj.get("length", len(traj["actions"])),
    )
    return {"type": "text", "text": text}


def peer_reference_traj_block(traj, has_images=False):
    """Return a text content block for a peer reference trajectory.

    A peer trajectory comes from another case of the SAME task_type,
    showing how a similar task was solved.

    Parameters
    ----------
    traj : dict or None
        Expected to have:
        - ``actions``         : list[str]
        - ``length``          : int
        - ``source_case_id``  : int (the case this trajectory came from)
        - ``description``     : str (task description of the peer case)
        - ``origin``          : dict with 'coord' [x, y] and 'heading'
    has_images : bool
        If True, include the text about interleaved images.
    """
    if not traj or not traj.get("actions"):
        return {"type": "text", "text": ""}
    template = PEER_REFERENCE_TRAJ_TEMPLATE_WITH_IMAGES if has_images else PEER_REFERENCE_TRAJ_TEMPLATE_TEXT_ONLY
    
    origin = traj.get("origin", {})
    origin_coord = origin.get("coord", [0, 0])
    origin_x = int(origin_coord[0]) if len(origin_coord) > 0 else 0
    origin_y = int(origin_coord[1]) if len(origin_coord) > 1 else 0
    origin_heading = origin.get("heading", "Unknown")
    
    text = template.format(
        source_case_id=traj.get("source_case_id", "?"),
        description=traj.get("description", "Unknown task"),
        origin_x=origin_x,
        origin_y=origin_y,
        origin_heading=origin_heading,
        actions=", ".join(traj["actions"]),
        length=traj.get("length", len(traj["actions"])),
    )
    return {"type": "text", "text": text}
