"""CURRENT IMAGE PROMPT — the first-person view at the agent's current pose.

This is the only image the agent **must** see every turn; the rest
(history / reference) are optional.

The header names the agent's current pose and facing so the LLM has
explicit grounding for "where am I, what direction am I looking".
"""
from ._helpers import image_block


def _pose_str(pose):
    if not pose:
        return "(unknown pose)"
    x = pose.get("x", "?")
    y = pose.get("y", "?")
    h = pose.get("heading", "?")
    return f"({x}, {y}) facing {h}"


def current_image_blocks(image_path, pose=None):
    """Return a list of content blocks: a header text + a base64 image_url.

    The header explicitly names the agent's current (x, y) and facing
    direction (this is the view at that pose, not at a generic one).

    Parameters
    ----------
    image_path : str
        Path to the local image file. May be missing — in that case the
        header is returned alone.
    pose : dict or None
        ``{"x": int, "y": int, "heading": str}`` for the agent's current pose.
    """
    header = f"Current view: you are at {_pose_str(pose)}.\n"
    blocks = [{"type": "text", "text": header}]
    blk = image_block(image_path)
    if blk is not None:
        blocks.append(blk)
    return blocks
