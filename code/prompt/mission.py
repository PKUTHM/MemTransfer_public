"""MISSION PROMPT — natural-language task + initial spawn info.

Edit ``MISSION_TEMPLATE`` to change the wording. The agent gets a Python
dict ``origin = {"x": int, "y": int, "heading": str}`` plus the raw
``description`` string from the server.
"""


MISSION_TEMPLATE = """Current Mission: {mission}
Your initial spawn: ({x}, {y}) facing {heading}.
Scene bounds X∈[-7,7], Y∈[-10,16].
Each image filename encodes your current pose; trust the visual content."""


OBSTACLE_NOTICE = """[ENVIRONMENT CHANGE — OBSTACLES]
Additional obstacles have been introduced into the current warehouse. Any historical memory provided was collected before these obstacles were added, so previously traversable routes may now be blocked.
The movement and turning rules remain the same, but traversability may have changed. Use the current view and action feedback to determine whether a route is passable. If a move is blocked, your position does not change; adjust your route instead of repeatedly following the blocked historical route.
[END ENVIRONMENT CHANGE]"""


def mission_block(mission, origin, obstacle_notice=False):
    """Return a single text content block describing the current mission.

    Parameters
    ----------
    mission : str
        Natural-language task description from the server (e.g. "Find any
        instance of Mug.").
    origin : dict
        Initial spawn info with keys ``x`` (int), ``y`` (int), ``heading``
        (one of "East"/"North"/"West"/"South").
    """
    text = MISSION_TEMPLATE.format(
        mission=mission,
        x=origin["x"],
        y=origin["y"],
        heading=origin["heading"],
    )
    if obstacle_notice:
        text += "\n\n" + OBSTACLE_NOTICE
    return {"type": "text", "text": text}
