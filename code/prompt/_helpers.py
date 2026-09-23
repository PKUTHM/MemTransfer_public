"""Shared helper used by prompt modules to embed local image files as base64.

Kept private to the prompt package so prompt files don't need to know about
PIL / io / base64 details — they just call ``encode_image_to_base64(path)``.

Pre-computed base64 cache:
    If ``USE_PRECOMPUTED_IMAGES`` is True (default), ``encode_image_to_base64``
    first checks for a pre-converted ``.b64`` file in ``IMAGE_BASE64_DIR``.
    This avoids re-encoding the same image on every call. Set
    ``USE_PRECOMPUTED_IMAGES = False`` to force runtime conversion.
"""
import os
import io
import base64
from PIL import Image

try:                                    # code/ is on sys.path when the agent runs
    import scene_index
except ImportError:                     # standalone import of the prompt package
    scene_index = None

#: JPEG geometry/quality used when a PNG has to be encoded at runtime.
#: Overridden from the experiment config's ``prompt`` block via
#: :func:`set_image_encoding`; the committed .b64 cache was built at these
#: values, so changing them only affects frames that miss the cache.
DEFAULT_MAX_SIZE = 448
DEFAULT_QUALITY = 80


def set_image_encoding(max_size=None, quality=None):
    """Point the runtime encoder at the config's values."""
    global DEFAULT_MAX_SIZE, DEFAULT_QUALITY
    if max_size is not None:
        DEFAULT_MAX_SIZE = int(max_size)
    if quality is not None:
        DEFAULT_QUALITY = int(quality)

# Fallback base64 cache directory, used only for image paths that belong to no
# registered scene. Per-scene lookup goes through ``_b64_dir_for``.
IMAGE_BASE64_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset", "warehouse_01_01", "image_base64"
)

# Global flag: use pre-computed .b64 files if available
USE_PRECOMPUTED_IMAGES = True


def set_use_precomputed(value):
    """Enable/disable use of pre-computed base64 files."""
    global USE_PRECOMPUTED_IMAGES
    USE_PRECOMPUTED_IMAGES = bool(value)


def _b64_dirs_for(image_path):
    """``image_base64`` directories to try for ``image_path``, in order.

    Every scene renders the same filename set, so a basename alone does not
    say which scene a frame comes from. Looking one up in a single global
    directory would hand a v5 run the *base* scene's JPEG for every pose --
    silently, because the cache is designed to work even when the PNG is
    absent, so nothing would raise and the model would simply never see the
    obstacle. Resolving the scene from the path first is what prevents that.

    The owning scene comes first, then its fallback: a v5 obstacle scene
    only re-rendered the poses that can see the obstacle, and takes every
    other pose from the base scene (see ``scene_index.fallback_scene``).

    Falls back to :data:`IMAGE_BASE64_DIR` when the path belongs to no known
    scene (legacy single-scene runs and unit tests).
    """
    if scene_index is not None:
        chain = scene_index.lookup_chain(image_path)
        if chain:
            return [s.image_base64_dir for s in chain]
    return [IMAGE_BASE64_DIR]


def _b64_dir_for(image_path):
    """The owning scene's ``image_base64`` directory (first of the chain)."""
    return _b64_dirs_for(image_path)[0]


def _png_fallback(image_path):
    """The fallback scene's PNG for a frame the owning scene lacks, or None."""
    if scene_index is not None:
        for scene in scene_index.lookup_chain(image_path)[1:]:
            p = os.path.join(scene.image_dir, os.path.basename(image_path))
            if os.path.exists(p):
                return p
    return None


def _get_precomputed_b64(image_path):
    """Return base64 string from pre-computed file, or None if not found."""
    if not USE_PRECOMPUTED_IMAGES:
        return None
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for b64_dir in _b64_dirs_for(image_path):
        if not b64_dir:
            continue
        b64_path = os.path.join(b64_dir, stem + ".b64")
        try:
            with open(b64_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except (FileNotFoundError, IOError):
            continue
    return None


def encode_image_to_base64(image_path, max_size=None, quality=None):
    """Return JPEG base64 string for the given local image file, or '' on failure.

    If ``USE_PRECOMPUTED_IMAGES`` is True and a matching ``.b64`` file exists
    in ``IMAGE_BASE64_DIR``, reads and returns it directly (much faster).
    This works even if the original PNG doesn't exist on disk.
    Otherwise falls back to runtime PIL conversion (requires PNG to exist).
    """
    if not image_path:
        return ""
    # Resolved here, not in the signature: default arguments bind once at
    # import, so a config applied afterwards would never reach them.
    max_size = DEFAULT_MAX_SIZE if max_size is None else max_size
    quality = DEFAULT_QUALITY if quality is None else quality
    # Try pre-computed cache first (doesn't require PNG to exist)
    cached = _get_precomputed_b64(image_path)
    if cached is not None:
        return cached
    # Fall back to runtime conversion (requires PNG to exist)
    if not os.path.exists(image_path):
        image_path = _png_fallback(image_path)
        if not image_path:
            return ""
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                img = bg
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            if max(w, h) > max_size:
                if w > h:
                    new_w, new_w_h = max_size, int(h * (max_size / w))
                    new_w, new_h = max_size, new_w_h
                else:
                    new_w = int(w * (max_size / h))
                    new_h = max_size
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return ""


def image_block(path, b64=None):
    """Build a content block for an image.

    The block always carries ``_image_path`` so the agent can drop the
    base64 data when persisting the prompt (see ``agent._serialize_block``).
    """
    if b64 is None:
        b64 = encode_image_to_base64(path)
    if not b64:
        return None
    return {
        "type": "image_url",
        "_image_path": path,
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }
