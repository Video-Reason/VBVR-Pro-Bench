"""Stable intermediate format between evaluators and video renderers.

Evaluators are free to expose task-specific diagnostic dictionaries.  An
adapter translates those dictionaries into this small schema, allowing one
renderer to handle all tasks without knowing evaluator implementation details.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


ComponentKind = Literal["quality", "gate", "penalty", "count", "status"]
Timing = Literal["live", "cumulative", "final"]
EventKind = Literal["reward", "penalty", "milestone", "verdict", "info"]


@dataclass(frozen=True)
class Component:
    id: str
    label: str
    value: float
    kind: ComponentKind = "quality"
    timing: Timing = "live"
    weight: Optional[float] = None
    note: str = ""
    children: tuple[str, ...] = ()


@dataclass(frozen=True)
class Formula:
    expression: str
    display: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class SpatialEvidence:
    kind: Literal["grid_cell", "bbox", "point", "mask", "trajectory", "frame"]
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    frame: int
    kind: EventKind
    component: str
    title: str
    detail: str = ""
    evidence: Optional[SpatialEvidence] = None


@dataclass(frozen=True)
class Explanation:
    task_name: str
    title: str
    frame: int
    final_score: float
    components: tuple[Component, ...]
    formula: Formula
    events: tuple[Event, ...] = ()
    source_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def component(self, component_id: str) -> Component:
        for item in self.components:
            if item.id == component_id:
                return item
        raise KeyError(component_id)
