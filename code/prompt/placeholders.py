"""PLACEHOLDERS for future memory / context mechanisms.

Each placeholder is a stub block the agent can include unconditionally.
Right now they're empty text blocks so the prompt stays unchanged; edit
the body of each function to add new memory modalities later.

Suggested future mechanisms (all commented-out stubs):

- ``semantic_memory_block``: long-term "I already know this is a coffee can
  from looking at the shelf 5 minutes ago" notes.
- ``social_memory_block``: persona / prior user instructions carried over
  across tasks.
- ``episodic_summary_block``: per-task natural-language summary written
  at the end of previous episodes.
- ``topological_map_block``: graph-based place-recognition map.
- ``uncertainty_block``: the agent's self-estimated confidence in its
  current belief state.
"""

PLACEHOLDER_HEADER = "\n[FUTURE_MEMORY_HOOKS]"


def placeholders_block():
    """Single text block listing active / inactive placeholder hooks.

    Keep this as a no-op by default so enabling new memory modules is a
    one-line change in ``agent.py``.
    """
    return {
        "type": "text",
        "text": (
            PLACEHOLDER_HEADER
            + "\n  (no extra memory mechanisms enabled — add new functions here)"
        ),
    }
