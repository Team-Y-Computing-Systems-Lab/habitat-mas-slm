"""Planners. All return the same Plan so results are comparable across modes:

  slm_only       (P0) the SLM writes the per-robot plan directly
  classical_only (P1) PDDL goal from the episode -> classical planner   [next]
  slm_classical  (P2) SLM -> PDDL goal/allocation -> classical planner  [next]
"""
