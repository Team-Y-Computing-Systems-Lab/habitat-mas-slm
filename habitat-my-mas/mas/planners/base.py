"""The plan format shared by every planner mode, and its validation."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Step:
    skill: str
    args: Dict[str, Any]


@dataclass
class Plan:
    """Per-robot step lists. Robots run their lists in parallel, each in order."""
    steps: Dict[str, List[Step]] = field(default_factory=dict)
    planner_calls: List[dict] = field(default_factory=list)  # SLM / planner call records
    raw: Any = None              # raw planner output, for debugging
    parse_ok: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def num_steps(self) -> int:
        return sum(len(v) for v in self.steps.values())


def validate(plan: Plan, tools: Dict[str, Dict[str, dict]]) -> List[str]:
    """Check robots, skills, and argument names against each robot's MCP tools.

    tools: {robot_id: {tool_name: input_schema}}
    """
    errors = []
    for robot, steps in plan.steps.items():
        if robot not in tools:
            errors.append(f"unknown robot {robot}")
            continue
        for i, s in enumerate(steps):
            schema = tools[robot].get(s.skill)
            if schema is None:
                errors.append(f"{robot} step {i}: no skill {s.skill}")
                continue
            props = set(schema.get("properties", {}))
            required = set(schema.get("required", []))
            missing, extra = required - set(s.args), set(s.args) - props
            if missing or extra:
                errors.append(f"{robot} step {i} {s.skill}: missing {sorted(missing)} extra {sorted(extra)}")
    return errors
