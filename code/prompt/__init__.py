"""Prompt package: every part of the agent prompt lives in its own file.

Each module exports a small public surface (a string constant and/or a
builder function) that ``agent.py`` composes. Edit the strings directly to
change what the model sees; the builder functions only handle the wiring
(text↔image, optional sections, etc.).
"""
from .system import (SYSTEM_PROMPT, build_system_prompt)
from .mission import mission_block, MISSION_TEMPLATE
from .current_image import current_image_blocks
from .history import history_text_block, history_with_images_blocks
from .history_images import history_images_blocks
from .grid_memory import grid_memory_block, GRID_MEMORY_HEADER
from .reference_traj import reference_traj_block, peer_reference_traj_block
from .reference_traj_images import reference_traj_images_blocks
from .placeholders import placeholders_block
from .episodic_memory import (episodic_memory_blocks, episodic_memory_multi_blocks,
                              find_best_match, find_best_match_vit,
                              DecisionContext, MatchResult,
                              VERSION_PRIORITY)
from .memory_sources import (common_note_block, case_scope_declaration,
                             global_scope_declaration, version_begin, version_end,
                             wrap_text_memory, origin_matches, VERSION_COLLECTION,
                             COMMON_NOTE, SAME_ORIGIN_COMMON_NOTE)
from . import vit_matcher
from ._helpers import encode_image_to_base64, set_use_precomputed, IMAGE_BASE64_DIR

__all__ = [
    "SYSTEM_PROMPT", "build_system_prompt",
    "mission_block", "MISSION_TEMPLATE",
    "current_image_blocks",
    "history_text_block",
    "history_with_images_blocks",
    "history_images_blocks",
    "grid_memory_block", "GRID_MEMORY_HEADER",
    "reference_traj_block", "peer_reference_traj_block",
    "reference_traj_images_blocks",
    "placeholders_block",
    "episodic_memory_blocks", "episodic_memory_multi_blocks",
    "find_best_match", "find_best_match_vit",
    "DecisionContext", "MatchResult", "VERSION_PRIORITY",
    "common_note_block", "case_scope_declaration", "global_scope_declaration",
    "version_begin", "version_end", "wrap_text_memory",
    "VERSION_COLLECTION", "COMMON_NOTE",
    "encode_image_to_base64",
    "set_use_precomputed",
    "IMAGE_BASE64_DIR",
]
