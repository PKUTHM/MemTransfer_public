"""HISTORY IMAGES PROMPT — lookback first-person views (T-1, T-2, ...).

Each entry is a dict with keys: ``path`` (str, image file path),
``action`` (str, action the agent took at that step), ``status`` (str,
server's reply, e.g. "moved" / "blocked" / "task_completed").
"""
from ._helpers import encode_image_to_base64


HISTORY_IMAGES_HEADER = "\n[IMAGE_TIMELINE_MEMORY]"


def history_images_blocks(history_imgs):
    """Return a list of content blocks: header + (text + image) per entry.

    Returns an empty list if ``history_imgs`` is empty/None, so callers
    can always extend unconditionally.
    """
    if not history_imgs:
        return []
    blocks = [{"type": "text", "text": HISTORY_IMAGES_HEADER}]
    for idx, info in enumerate(history_imgs):
        t_offset = len(history_imgs) - idx
        marker = f"\n- [Image T-{t_offset}] (Action: {info['action']} -> [{info['status']}]):"
        blocks.append({"type": "text", "text": marker})
        b64 = encode_image_to_base64(info.get("path", ""))
        if b64:
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    return blocks
