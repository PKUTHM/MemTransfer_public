"""MEMORY SOURCE WRAPPERS — provenance labels around every long-term memory.

Every group that carries long-term memory (G02–G09) wraps each memory
segment in the same BEGIN/END markers and the same common note, so that
single-version and multi-version groups differ only in *how many*
segments appear — never in how a segment is introduced. The
working-memory baseline (G01) gets none of this.

Design constraint: the surrounding prompt (system prompt, mission,
current view, history, closing reminder) stays byte-for-byte identical to
the v0 evaluation. These wrappers are the *only* new text, and they live
strictly inside the memory sections.
"""

#: Collection conditions per version. Purely descriptive of how the
#: variant was designed; per-case geometry always comes from the record
#: itself, never from these labels.
VERSION_COLLECTION = {
    "v0": "the original task origin and initial heading.",
    "v1": "the same origin as V0, re-collected after changing the initial heading.",
    "v2": "an origin shifted a short distance away from V0.",
    "v3": "a different origin and corridor, to obtain experience on a different route; "
          "the actual path is whatever the record shows.",
}

COMMON_NOTE = (
    "\n[HISTORICAL EXPERIENCE — COMMON NOTE]\n"
    "The following are historical experiences collected in this same warehouse "
    "environment, from origins or initial headings that may differ from yours. "
    "Each memory segment is labelled with its own source.\n"
    "Use them together with your current mission, your current pose, the current "
    "view and your own execution history.\n"
    "A historical action sequence belongs to its own historical origin: it is NOT "
    "a route you should replay step by step.\n"
    "Where a historical task asked the expert to return to origin, that refers to "
    "that historical origin. Your current task always uses the origin you were "
    "given at init.\n"
    "Version and case labels identify where an experience came from; they are not "
    "answers to your current task.\n"
    "[END COMMON NOTE]\n"
)

#: :data:`COMMON_NOTE` for memory collected from exactly the origins under
#: test (in this dataset: V0 memory on the V0 test). Only the three sentences
#: about origins differ; the rest is the neutral note verbatim, so the two
#: stay comparable. Selected by ``prompt.same_origin_wording``.
SAME_ORIGIN_COMMON_NOTE = (
    "\n[HISTORICAL EXPERIENCE — COMMON NOTE]\n"
    "The following are historical experiences collected in this same warehouse "
    "environment, from the SAME origins and initial headings that your current "
    "test uses for each case. Each memory segment is labelled with its own source.\n"
    "Use them together with your current mission, your current pose, the current "
    "view and your own execution history.\n"
    "A historical action sequence recorded for your current case started exactly "
    "where you start now: same position, same initial heading.\n"
    "Where a historical task asked the expert to return to origin, that origin is "
    "the same as yours.\n"
    "Version and case labels identify where an experience came from; they are not "
    "answers to your current task.\n"
    "[END COMMON NOTE]\n"
)


def _pose_of(origin):
    """``(x, y, heading)`` from a record origin or a decision-context pose."""
    if not origin:
        return None
    if "coord" in origin:
        coord = origin.get("coord") or []
        if len(coord) < 2:
            return None
        return int(float(coord[0])), int(float(coord[1])), origin.get("heading")
    if "x" in origin and "y" in origin:
        return int(origin["x"]), int(origin["y"]), origin.get("heading")
    return None


def origin_matches(historical, initial):
    """Did a historical record start exactly where the current run starts?

    ``historical`` is a record origin (``{"coord": [x, y], "heading": ...}``);
    ``initial`` is the agent's spawn in either that shape or the
    ``{"x", "y", "heading"}`` shape :class:`DecisionContext` carries. Cell
    and heading must both agree: V1 shares V0's cell but not its heading,
    and that is precisely the difference it exists to test.
    """
    a = _pose_of(historical)
    return a is not None and a == _pose_of(initial)


def common_note_block(same_origin=False):
    """The shared preamble. Identical for single- and multi-version groups;
    ``same_origin`` swaps in :data:`SAME_ORIGIN_COMMON_NOTE`."""
    return {"type": "text",
            "text": SAME_ORIGIN_COMMON_NOTE if same_origin else COMMON_NOTE}


def _fmt_origin(origin):
    if not origin:
        return "(unknown)"
    coord = origin.get("coord") or []
    heading = origin.get("heading", "?")
    if len(coord) >= 2:
        return "(%s, %s) facing %s" % (int(coord[0]), int(coord[1]), heading)
    return "(unknown) facing %s" % heading


def case_scope_declaration(task_id, case_no, versions, test_version="v4"):
    """Scope note for Full Context / Episodic (same task, same case only)."""
    vs = ", ".join(v.upper() for v in versions)
    return {"type": "text", "text": (
        "\n[MEMORY SCOPE — SAME CASE]\n"
        "Current test target: %s / %s / case%02d.\n"
        "The reference trajectories or retrieved excerpts below come only from "
        "the SAME task and the SAME case, recorded in: %s.\n"
        "Different historical versions may start from different origins and "
        "headings; read each segment's own record.\n"
        % (test_version.upper(), task_id, int(case_no), vs)
    )}


def global_scope_declaration(memory_type, versions):
    """Scope note for Summary / Semantic (whole-version, cross-case memory)."""
    vs = ", ".join(v.upper() for v in versions)
    return ("\n[MEMORY SCOPE — WHOLE VERSION]\n"
            "Each %s segment below was generated independently from all 100 cases "
            "of its own version (%s). It carries cross-task, cross-case experience "
            "and is not filtered to your current test case. Case numbers inside a "
            "segment belong to that segment's own version.\n"
            % (memory_type, vs))


def version_begin(version, memory_type, scope, case_label=None,
                  task_description=None, origin=None, same_origin=False):
    """Opening marker for one version's memory segment.

    Parameters
    ----------
    version : str
        ``"v0"`` … ``"v3"``.
    memory_type : str
        Human label, e.g. ``"reference trajectory"``, ``"Summary memory"``.
    scope : str
        What experience the segment covers.
    case_label, task_description, origin :
        Only supplied for per-case memories (Full Context / Episodic). A
        whole-version text memory has no single historical origin, so
        those lines are omitted rather than filled with a placeholder.
    same_origin : bool
        The record started at the agent's own spawn; said on the origin line.
    """
    v = version.upper()
    lines = ["\n[HISTORICAL MEMORY %s — BEGIN]" % v,
             "Collection condition: %s" % VERSION_COLLECTION.get(version, "unspecified."),
             "Memory type: %s" % memory_type,
             "Experience scope: %s" % scope]
    if case_label is not None:
        lines.append("Historical case: %s" % case_label)
    if task_description is not None:
        lines.append("Historical task: %s" % task_description)
    if origin is not None:
        lines.append("Historical origin: %s%s" % (
            _fmt_origin(origin), " (same as your origin)" if same_origin else ""))
    lines.append("")
    return {"type": "text", "text": "\n".join(lines)}


def version_end(version):
    return {"type": "text", "text": "\n[HISTORICAL MEMORY %s — END]\n" % version.upper()}


def wrap_text_memory(memories, memory_type, same_origin=False):
    """Render a list of :class:`experiment.TextMemory` as one text string.

    Used for the Summary / Semantic sections, which are pure text and are
    injected inside the v0 ``[MEMORY RECALL — … INSTRUCTIONS]`` markers so
    the outer prompt shape does not change. ``same_origin`` selects
    :data:`SAME_ORIGIN_COMMON_NOTE` (memory built from the version under test).
    """
    if not memories:
        return ""
    versions = [m.version for m in memories]
    parts = [SAME_ORIGIN_COMMON_NOTE if same_origin else COMMON_NOTE,
             global_scope_declaration(memory_type, versions)]
    for mem in memories:
        head = version_begin(
            mem.version, memory_type,
            "all %d cases of version %s" % (mem.scope_case_count, mem.version.upper()),
        )
        parts.append(head["text"])
        parts.append(mem.text.strip())
        parts.append(version_end(mem.version)["text"])
    return "".join(parts)
