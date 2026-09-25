"""Natural-language instructions from an episode's PDDL goal (datasets.md §5a, P4).

Template-based "explicit" level only for now: every object and place is named
exactly as the robots' tools expect. Referential / underspecified levels and
SLM paraphrasing come later.
"""

from typing import List


def explicit(goal: List[str]) -> str:
    moves, stations, empty = [], [], False
    for g in goal:
        pred, *args = g.strip("()").split()
        if pred == "obj-at":
            moves.append(f"put {args[0]} on {args[1]}")
        elif pred == "robot-at":
            stations.append(f"{args[0]} must end at {args[1]}")
        elif pred == "hand-empty":
            empty = True
    parts = []
    if moves:
        parts.append("Tidy up: " + "; ".join(moves) + ".")
    if stations:
        parts.append(" ".join(s[0].upper() + s[1:] + "." for s in stations))
    if empty:
        parts.append("When done, no robot should be holding anything.")
    return " ".join(parts)
