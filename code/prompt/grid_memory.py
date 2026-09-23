"""GRID MEMORY PROMPT — a 2-D ASCII map of where the agent has been and what it saw.

Edit ``format_grid_memory`` to change the visual style. The agent passes a
``grid_state`` dict built up over time; the default implementation just
renders visited cells as dots and a few key markers. This is a placeholder
that the user can extend (e.g. with item sightings, wall inference, etc.).
"""

GRID_MEMORY_HEADER = "\n[GRID_MEMORY — explored cells so far]"


def _default_render(grid_state):
    visited = grid_state.get("visited", set())
    seen_items = grid_state.get("items_seen", {})
    walls = grid_state.get("walls", set())
    bounds = grid_state.get("bounds", {"x": (-7, 7), "y": (-10, 16)})
    (xmin, xmax), (ymin, ymax) = bounds["x"], bounds["y"]
    lines = []
    lines.append(f"  X axis (East +, West -) | Y axis (North +, South -)")
    for y in range(ymax, ymin - 1, -1):
        row = []
        for x in range(xmin, xmax + 1):
            if (x, y) in walls:
                row.append("█")
            elif (x, y) in visited:
                row.append("·")
            else:
                row.append(" ")
        lines.append(f"  Y={y:>3}  " + "".join(row))
    lines.append("  " + " " * 7 + "".join(f"{x%10}" for x in range(xmin, xmax + 1)))
    if seen_items:
        lines.append("  Items seen (cell → label):")
        for (x, y), label in sorted(seen_items.items()):
            lines.append(f"    ({x},{y}) → {label}")
    return "\n".join(lines)


def grid_memory_block(grid_state):
    """Return a single text content block with the current grid memory.

    Parameters
    ----------
    grid_state : dict or None
        If None or empty, returns just the header. The dict may contain:
        - ``visited``  : set[(x, y)]
        - ``items_seen``: dict[(x, y), str]
        - ``walls``    : set[(x, y)]
        - ``bounds``   : {"x": (lo, hi), "y": (lo, hi)}
    """
    if not grid_state:
        return {"type": "text", "text": GRID_MEMORY_HEADER + "\n  (empty — extend agent.py to populate)"}
    body = _default_render(grid_state)
    return {"type": "text", "text": GRID_MEMORY_HEADER + "\n" + body}
