"""HISTORY PROMPT — past step log interleaved with first-person views.

Each step is rendered in a clear three-segment structure so the LLM
never has to guess which image corresponds to which action::

    Step 7: at (2, 9) facing South — view seen at the start of step 7:
    <IMG base64>
    → did: Turn_Right → result: success, still at (2, 9) now facing West
            (heading changed)

If the number of images is smaller than the number of step lines, the
**newest** steps carry the images and the older ones are text-only
(per the spec: "earlier history has no images, later ones do").

If the number of step lines is larger than the ``max_history_steps`` cap,
only the newest ``max_history_steps`` step lines are kept (older ones
are dropped entirely).

The log lines passed in (``logs``) typically already start with
"Step N: ..." — we strip that prefix when re-rendering so it isn't
duplicated by our outer "Step N:" header.
"""
import re
from ._helpers import image_block


NO_HISTORY_TEXT = "No historical actions taken yet. This is the first step."

_STEP_PREFIX_RE = re.compile(r"^Step\s+\d+:\s*")


def _strip_step_prefix(line):
    return _STEP_PREFIX_RE.sub("", line, count=1)


def _pose_str(pose):
    if not pose:
        return "(earlier step, pose not cached)"
    x = pose.get("x", "?")
    y = pose.get("y", "?")
    h = pose.get("heading", "?")
    return f"({x}, {y}) facing {h}"


def history_text_block(logs):
    """Legacy text-only block (kept for callers that don't want images)."""
    if not logs:
        return {"type": "text", "text": NO_HISTORY_TEXT}
    text = f"Execution Log History (last {len(logs)} steps):\n" + "\n".join(logs)
    return {"type": "text", "text": text}


def history_with_images_blocks(logs, img_history, max_history_steps=5, max_images=5):
    """Return a list of content blocks: header + per-step (position, image, action).

    Parameters
    ----------
    logs : list[str]
        The agent's step log lines (one per step, oldest first). May be
        longer than ``max_history_steps`` — only the newest N are kept.
    img_history : list[dict]
        The agent's per-step records (``"pose"`` always, ``"path"`` only
        for the last ``LOOKBACK_IMAGE_COUNT`` steps the agent retained).
        Length is at least ``min(len(logs), total_steps)``; we align by
        step number, not by list position.
    max_history_steps : int
        Cap on how many log lines to render (newest N).
    max_images : int
        Cap on how many of the kept step lines get their image shown
        (newest N within the kept lines).
    """
    if not logs and not img_history:
        return [{"type": "text", "text": NO_HISTORY_TEXT}]

    # Trim logs to the last max_history_steps entries.
    if max_history_steps > 0:
        logs = logs[-max_history_steps:]
    else:
        logs = []
    n_text = len(logs)

    # The kept log entries correspond to the last n_text steps overall.
    # img_history is indexed by step number (0-based), so the first kept
    # log entry (logs[0]) is the step at position
    # ``len(img_history) - n_text`` in img_history (clipped to 0).
    img_offset = max(0, len(img_history) - n_text)

    # Of the n_text kept entries, only the newest max_images show an image.
    n_img_show = min(max_images, n_text)
    first_with_image = max(0, n_text - n_img_show)

    header_lines = [
        "History of past steps (oldest to newest):",
    ]
    if n_text > 0:
        kept_note = (
            f"showing the latest {n_text} step(s) (out of "
            f"{len(img_history)} total steps the agent has executed)"
            if len(img_history) > n_text else
            f"showing all {n_text} step(s)"
        )
        if n_img_show > 0:
            img_note = (
                f"; the {n_img_show} newest of these show the view the agent "
                f"saw at the start of that step, the older {n_text - n_img_show} "
                f"are text-only because no images were retained for them."
            )
        else:
            img_note = "; all entries are text-only (history image count is 0)."
        header_lines.append(kept_note + img_note)
    header_lines.append("")
    blocks = [{"type": "text", "text": "\n".join(header_lines)}]

    for i, line in enumerate(logs):
        # Use absolute step numbers (1-indexed across the full history),
        # so "Step 7" really means the 7th step the agent has taken.
        step_num = img_offset + 1 + i
        img_idx = img_offset + i
        has_image = (
            i >= first_with_image
            and img_idx < len(img_history)
            and bool(img_history[img_idx].get("path"))
        )

        if img_idx < len(img_history):
            pose = img_history[img_idx].get("pose") or {}
            pos_str = _pose_str(pose)
        else:
            pose = {}
            pos_str = "(earlier step, pose not cached)"

        body = _strip_step_prefix(line)

        if has_image:
            blocks.append({
                "type": "text",
                "text": f"Step {step_num}: at {pos_str} — view seen at the start of step {step_num}:\n",
            })
            path = img_history[img_idx].get("path", "")
            blk = image_block(path)
            if blk is not None:
                blocks.append(blk)
            else:
                blocks.append({"type": "text", "text": "  [image unavailable]\n"})
            blocks.append({"type": "text",
                           "text": f"  → did: {body}\n"})
        else:
            blocks.append({
                "type": "text",
                "text": f"Step {step_num}: at {pos_str}\n  → did: {body}\n",
            })

    return blocks
