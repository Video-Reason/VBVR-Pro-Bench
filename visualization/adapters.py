"""Adapters from evaluator-specific details to renderer-ready explanations."""

from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict, Sequence

from .schema import Component, Event, Explanation, Formula, SpatialEvidence
from .semantics import (component_failure_for, failure_for, label_for,
                        selection_event_title, selection_target_for, title_for)


Adapter = Callable[[str, Sequence[Dict[str, Any]], int], Explanation]
_ADAPTERS: dict[str, Adapter] = {}


def register(task_name: str):
    def decorator(fn: Adapter) -> Adapter:
        if task_name in _ADAPTERS:
            raise ValueError(f"duplicate explanation adapter: {task_name}")
        _ADAPTERS[task_name] = fn
        return fn
    return decorator


def get_adapter(task_name: str) -> Adapter:
    try:
        return _ADAPTERS[task_name]
    except KeyError as exc:
        raise KeyError(f"no explanation adapter for {task_name}") from exc


def supported_tasks() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def explain(task_name, steps, index):
    """Adapt one prefix, including evaluator-declared early-prefix errors."""
    current = steps[index]
    if current.get("error"):
        score = float(current.get("final_score", 0.0) or 0.0)
        error = str(current["error"]).replace("_", " ").strip()
        is_final = index == len(steps) - 1
        event_title = ("Task evidence could not be evaluated" if is_final
                       else "Awaiting sufficient visual evidence")
        return Explanation(
            task_name=task_name, title=_pretty_task(task_name),
            frame=int(current.get("k", index + 1)), final_score=score,
            components=(Component("prefix_evidence", "Prefix evidence", 0.0,
                                  kind="status", note=error),),
            formula=Formula("prefix_error", "Evaluator evidence unavailable",
                            ("prefix_evidence",)),
            events=(Event(
                int(current.get("k", index + 1)),
                "verdict" if is_final else "info", "prefix_evidence",
                event_title, error[:120], SpatialEvidence("frame")),),
            source_details=dict(current))
    try:
        return get_adapter(task_name)(task_name, steps, index)
    except KeyError as exc:
        # Some evaluators legitimately return only a score until enough frames
        # exist to calculate their task-specific sub-scores.
        if index == len(steps) - 1:
            raise
        score = float(current.get("final_score", 0.0) or 0.0)
        return Explanation(
            task_name=task_name, title=task_name.split("_data-generator")[0],
            frame=int(current.get("k", index + 1)), final_score=score,
            components=(Component("prefix_evidence", "Prefix evidence", 0.0,
                                  kind="status", note="sub-scores pending"),),
            formula=Formula("prefix_pending", "Awaiting task-specific evidence",
                            ("prefix_evidence",)),
            events=(), source_details={**current, "pending_field": str(exc)})


def _number(details: dict[str, Any], key: str) -> float:
    value = details[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric, got {type(value).__name__}")
    return float(value)


def _spatial_evidence(details: dict[str, Any], component_id: str) -> SpatialEvidence:
    """Prefer evaluator-emitted coordinates that identify the judged region."""
    lowered = component_id.lower()
    cell_keys = []
    if "wall" in lowered:
        cell_keys = ["wall_hit_cells"]
    elif any(word in lowered for word in ("path", "route", "task", "core")):
        cell_keys = ["missed_keys", "missed_required", "obstacle_hit_cells",
                     "wall_hit_cells"]
    for key in cell_keys:
        cells = details.get(key)
        if isinstance(cells, list) and cells and isinstance(cells[0], (list, tuple)):
            cell = cells[0]
            if len(cell) == 2:
                return SpatialEvidence("grid_cell", {
                    "row": int(cell[0]), "col": int(cell[1]), "source": key})

    if any(word in lowered for word in
           ("primary", "core", "completion", "position", "sequence",
            "delete", "insert", "substitute", "edit", "box")):
        for key in ("target_bbox", "red_box", "template_bbox"):
            bbox = details.get(key)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return SpatialEvidence("bbox", {
                    "x": float(bbox[0]), "y": float(bbox[1]),
                    "w": float(bbox[2]), "h": float(bbox[3]),
                    "source": key})
        for key in ("target_center", "final_center", "last_center"):
            point = details.get(key)
            if isinstance(point, (list, tuple)) and len(point) == 2:
                return SpatialEvidence("point", {
                    "x": float(point[0]), "y": float(point[1]), "source": key})
    return SpatialEvidence("frame")


G45 = "G-45_key_door_matching_data-generator"
O29 = "O-29_ballcolor_data-generator"
O31 = "O-31_ball_eating_data-generator"
O32 = "O-32_rolling_ball_data-generator"
O52 = "O-52_traffic_light_data-generator"
O27 = "O-27_move_2_object_to_2_target_data-generator"
G273 = "G-273_high_density_liquid_data-generator"


@register(G45)
def explain_g45(task_name: str, steps: Sequence[dict[str, Any]], index: int) -> Explanation:
    """Explain one G-45 prefix using only values emitted by its evaluator."""
    if not steps:
        raise ValueError("G-45 explanation requires at least one timeline step")
    if not 0 <= index < len(steps):
        raise IndexError(index)

    current = steps[index]
    proximity = _number(current, "proximity")
    continuity = _number(current, "continuity_factor")
    coverage = _number(current, "coverage")
    wall_multiplier = _number(current, "wall_multiplier")
    key_multiplier = _number(current, "key_multiplier")
    trajectory = (proximity + continuity + coverage) / 3.0
    wall_gate = 0.4 + 0.6 * wall_multiplier
    key_gate = 0.4 + 0.6 * key_multiplier

    events: list[Event] = []
    seen_cells: set[tuple[int, int]] = set()
    previous_gate = 1.0
    for step in steps[:index + 1]:
        for raw_cell in step.get("wall_hit_cells", []):
            cell = tuple(int(x) for x in raw_cell)
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            gate = 0.4 + 0.6 * _number(step, "wall_multiplier")
            events.append(Event(
                frame=int(step["k"]), kind="penalty", component="wall_gate",
                title=f"Wall hit ({cell[0]}, {cell[1]})",
                detail=f"wall gate {previous_gate:.3f} → {gate:.3f}",
                evidence=SpatialEvidence("grid_cell", {"row": cell[0], "col": cell[1]}),
            ))
            previous_gate = gate

    is_final = index == len(steps) - 1
    if is_final and not bool(current.get("key_visited")):
        cell = current.get("target_key_cell")
        evidence = None
        if isinstance(cell, list) and len(cell) == 2:
            evidence = SpatialEvidence("grid_cell", {"row": cell[0], "col": cell[1]})
        events.append(Event(
            frame=int(current["k"]), kind="verdict", component="key_gate",
            title="Target key not collected",
            detail=f"key gate × {key_gate:.3f}", evidence=evidence,
        ))
    if is_final and proximity < 0.999:
        events.append(Event(
            int(current["k"]), "verdict", "proximity",
            "Path strayed from the key-door route",
            "path proximity %.3f" % proximity, SpatialEvidence("trajectory")))
    if is_final and continuity < 0.999:
        events.append(Event(
            int(current["k"]), "penalty", "continuity",
            "Movement was discontinuous",
            "continuity %.3f" % continuity, SpatialEvidence("trajectory")))
    if is_final and coverage < 0.999:
        events.append(Event(
            int(current["k"]), "verdict", "coverage",
            "Key-door route coverage is incomplete",
            "route coverage %.3f" % coverage,
            _cell_evidence(current.get("target_door_cell"), "target_door_cell")))

    components = (
        Component("trajectory_quality", "Trajectory quality", trajectory,
                  children=("proximity", "continuity", "coverage")),
        Component("proximity", "Path proximity", proximity),
        Component("continuity", "Motion continuity", continuity),
        Component("coverage", "Route coverage", coverage, timing="cumulative"),
        Component("wall_gate", "Wall avoidance gate", wall_gate, kind="gate",
                  timing="cumulative", note=f"{len(seen_cells)} wall cells"),
        Component("key_gate", "Target key gate", key_gate, kind="gate",
                  timing="final" if not is_final else "live",
                  note="collected" if current.get("key_visited") else "not collected"),
    )
    return Explanation(
        task_name=task_name, title="Key–door navigation",
        frame=int(current["k"]), final_score=_number(current, "final_score"),
        components=components,
        formula=Formula(
            expression="trajectory_quality * wall_gate * key_gate",
            display="Trajectory quality × Wall gate × Key gate",
            inputs=("trajectory_quality", "wall_gate", "key_gate"),
        ),
        events=tuple(events), source_details=dict(current),
    )


def _base(task_name, title, steps, index, components, formula, events):
    current = steps[index]
    return Explanation(
        task_name=task_name, title=title, frame=int(current["k"]),
        final_score=_number(current, "final_score"), components=components,
        formula=formula, events=tuple(events), source_details=dict(current),
    )


def _require_step(steps, index):
    if not steps:
        raise ValueError("explanation requires at least one timeline step")
    if not 0 <= index < len(steps):
        raise IndexError(index)
    return steps[index]


@register(O29)
def explain_o29(task_name, steps, index):
    current = _require_step(steps, index)
    final_state = _number(current, "final_state")
    process = _number(current, "merge_process")
    weighted = 0.6 * final_state + 0.4 * process
    process_gate = 0.2 + 0.8 * process
    components = (
        Component("weighted_quality", "Weighted quality", weighted,
                  children=("final_state", "merge_process")),
        Component("final_state", "Final state", final_state, timing="final", weight=0.6,
                  children=("survivor_count", "cluster_shape", "clean_scene",
                            "final_label")),
        Component("merge_process", "Merge process", process, timing="cumulative", weight=0.4),
        Component("process_gate", "Process evidence gate", process_gate,
                  kind="gate", timing="cumulative", children=("merge_process",)),
        Component("survivor_count", "Survivor count",
                  _number(current, "final_survivor_score"), timing="final"),
        Component("cluster_shape", "Cluster shape",
                  _number(current, "final_cluster_score"), timing="final"),
        Component("clean_scene", "No stray balls",
                  _number(current, "final_clean_score"), timing="final"),
        Component("final_label", "Final label",
                  _number(current, "final_text_score"), timing="final"),
    )
    events, seen = [], set()
    for step in steps[:index + 1]:
        for merge_index in range(1, int(step.get("merge_gen_merges") or 0) + 1):
            if merge_index in seen:
                continue
            match = re.match(r"gt(\d+)/gen(\d+)=([\d.]+)",
                             str(step.get("merge_m%d_count" % merge_index, "")))
            if not match:
                continue
            seen.add(merge_index)
            expected, actual, value = int(match.group(1)), int(match.group(2)), float(match.group(3))
            events.append(Event(
                int(step["k"]), "milestone" if value >= 1 else "penalty",
                "merge_process", "Merge %d completed" % merge_index,
                "survivor %d / expected %d; count score %.2f" % (actual, expected, value),
                SpatialEvidence("frame")))
    if index == len(steps) - 1 and _number(current, "final_text_score") < 1:
        events.append(Event(int(current["k"]), "verdict", "final_label",
                            "Final label not rendered",
                            "label score %.2f" % _number(current, "final_text_score"),
                            SpatialEvidence("frame")))
    if index == len(steps) - 1:
        final_checks = (
            ("final_survivor_score", "survivor_count",
             "Incorrect number of balls remain"),
            ("final_cluster_score", "cluster_shape",
             "Merged cluster shape is incorrect"),
            ("final_clean_score", "clean_scene", "Stray balls remain"),
        )
        for field, component, title in final_checks:
            if _number(current, field) < 0.999:
                events.append(Event(
                    int(current["k"]), "verdict", component, title,
                    "%s %.3f" % (component.replace("_", " "),
                                   _number(current, field)),
                    SpatialEvidence("frame")))
        if process < 0.999 and not any(
                event.kind == "penalty" and event.component == "merge_process"
                for event in events):
            events.append(Event(
                int(current["k"]), "verdict", "merge_process",
                "Merge sequence is incomplete",
                "merge-process score %.3f" % process,
                SpatialEvidence("frame")))
    return _base(task_name, "Ball-cluster merging", steps, index, components,
                 Formula("weighted_quality * process_gate",
                         "(0.60 × Final state + 0.40 × Merge process) × Process gate",
                         ("weighted_quality", "process_gate")), events)


@register(O31)
def explain_o31(task_name, steps, index):
    current = _require_step(steps, index)
    components = (
        Component("final_state", "Final state", _number(current, "final_state"),
                  timing="final", weight=0.5,
                  children=("targets_absorbed", "eater_growth", "clean_scene")),
        Component("eat_process", "Eat process", _number(current, "eat_process"),
                  timing="cumulative", weight=0.5),
        Component("targets_absorbed", "Targets absorbed",
                  _number(current, "final_targets_score"), timing="final"),
        Component("eater_growth", "Eater growth", _number(current, "final_black_score"),
                  timing="final"),
        Component("clean_scene", "No stray pixels", _number(current, "final_clean_score"),
                  timing="final"),
    )
    events, seen = [], set()
    for step in steps[:index + 1]:
        for eat_index in range(1, int(step.get("eat_gen_eats") or 0) + 1):
            if eat_index in seen:
                continue
            seen.add(eat_index)
            events.append(Event(int(step["k"]), "milestone", "eat_process",
                                "Eat %d completed" % eat_index,
                                str(step.get("eat_eat%d" % eat_index, "")),
                                SpatialEvidence("frame")))
    if index == len(steps) - 1:
        for eat_index in range(1, int(current.get("eat_gt_eats") or 0) + 1):
            detail = str(current.get("eat_eat%d" % eat_index, ""))
            if "missing" in detail:
                events.append(Event(int(current["k"]), "verdict", "eat_process",
                                    "Ball %d never eaten" % eat_index, detail,
                                    SpatialEvidence("frame")))
        if _number(current, "final_black_score") < 1:
            events.append(Event(int(current["k"]), "verdict", "eater_growth",
                                "Eater did not grow enough",
                                "area ratio vs GT %.3f" % _number(current, "final_black_area_ratio"),
                                SpatialEvidence("frame")))
        if _number(current, "final_targets_score") < 0.999:
            events.append(Event(
                int(current["k"]), "verdict", "targets_absorbed",
                "One or more target balls remain",
                "absorption score %.3f" % _number(current, "final_targets_score"),
                SpatialEvidence("frame")))
        if _number(current, "final_clean_score") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "clean_scene",
                "Stray ball pixels remain",
                "clean-scene score %.3f" % _number(current, "final_clean_score"),
                SpatialEvidence("frame")))
    return _base(task_name, "Ball eating", steps, index, components,
                 Formula("0.5 * final_state + 0.5 * eat_process",
                         "0.50 × Final state + 0.50 × Eat process",
                         ("final_state", "eat_process")), events)


@register(O32)
def explain_o32(task_name, steps, index):
    current = _require_step(steps, index)
    gate = 0.6 + 0.4 * _number(current, "consistency")
    components = (
        Component("completion", "Task completion", _number(current, "completion"),
                  children=("position", "on_path", "direction", "waypoints")),
        Component("position", "Final position", _number(current, "pos_score"), timing="final"),
        Component("on_path", "On-path ratio", _number(current, "on_path_ratio"),
                  timing="cumulative", note="%d/%d frames" %
                  (int(current.get("on_path", 0)), int(current.get("detected", 0)))),
        Component("direction", "Moving rightward", _number(current, "direction_score"),
                  timing="cumulative"),
        Component("waypoints", "Intermediate waypoints",
                  _number(current, "intermediate_position_score"), timing="cumulative"),
        Component("consistency", "Scene consistency", _number(current, "consistency"),
                  children=("scene_preservation", "background_clean")),
        Component("scene_preservation", "Track preserved",
                  _number(current, "scene_preservation"), timing="final"),
        Component("background_clean", "Background clean",
                  _number(current, "background_clean"), timing="final"),
        Component("consistency_gate", "Consistency gate", gate, kind="gate",
                  children=("consistency",)),
    )
    events, off_track, start = [], False, None
    previous = steps[0]
    for step in steps[1:index + 1]:
        detected_delta = int(step.get("detected", 0)) - int(previous.get("detected", 0))
        on_delta = int(step.get("on_path", 0)) - int(previous.get("on_path", 0))
        now_off = detected_delta > on_delta
        if now_off and not off_track:
            start = int(step["k"])
        if off_track and not now_off and start is not None:
            events.append(Event(start, "penalty", "on_path", "Ball left the path",
                                "off-path interval starts at frame %d" % start,
                                SpatialEvidence("frame")))
            start = None
        off_track, previous = now_off, step
    if off_track and start is not None:
        events.append(Event(start, "penalty", "on_path", "Ball left the path",
                            "off-path interval starts at frame %d" % start,
                            SpatialEvidence("frame")))
    if index == len(steps) - 1:
        detected, on_path = int(current.get("detected", 0)), int(current.get("on_path", 0))
        if detected > on_path:
            events.append(Event(int(current["k"]), "verdict", "on_path",
                                "Off track for %d of %d frames" %
                                (detected - on_path, detected),
                                "on-path ratio %.4f" % _number(current, "on_path_ratio"),
                                SpatialEvidence("frame")))
        if _number(current, "scene_preservation") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "scene_preservation",
                "Track or scene elements changed",
                "track preservation %.3f" % _number(current, "scene_preservation"),
                SpatialEvidence("frame")))
        if _number(current, "background_clean") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "background_clean",
                "Background changed",
                "background score %.3f" % _number(current, "background_clean"),
                SpatialEvidence("frame")))
    return _base(task_name, "Rolling ball on a track", steps, index, components,
                 Formula("completion * consistency_gate",
                         "Completion × Consistency gate",
                         ("completion", "consistency_gate")), events)


@register(O52)
def explain_o52(task_name, steps, index):
    current = _require_step(steps, index)
    components = (
        Component("final_state", "Final state", _number(current, "final_state"),
                  timing="final", weight=0.4,
                  children=("final_color", "final_countdown")),
        Component("process", "Traffic-light process", _number(current, "process"),
                  timing="cumulative", weight=0.6,
                  children=("process_countdown", "color_change",
                            "synchronization", "background")),
        Component("final_color", "Final colours", _number(current, "final_color"), timing="final"),
        Component("final_countdown", "Final countdowns",
                  _number(current, "final_countdown"), timing="final"),
        Component("process_countdown", "Countdown sequence",
                  _number(current, "proc_countdown"), timing="cumulative"),
        Component("color_change", "Colour transitions",
                  _number(current, "proc_color_change"), timing="cumulative"),
        Component("synchronization", "Lights synchronized",
                  _number(current, "proc_sync"), timing="cumulative"),
        Component("background", "Background preserved", _number(current, "proc_bg")),
    )
    labels = {
        "proc_color_change": ("color_change", "Colour transitions incorrect"),
        "proc_sync": ("synchronization", "Lights do not change together"),
        "proc_countdown": ("process_countdown", "Countdown sequence incomplete"),
        "proc_bg": ("background", "Background or fixture changed"),
    }
    events = []
    for key, (component, title) in labels.items():
        base = steps[0].get(key)
        judged = next((s for s in steps[1:] if s.get(key) != base), steps[-1])
        if judged["k"] <= current["k"] and _number(judged, key) < 1:
            events.append(Event(int(judged["k"]), "penalty", component, title,
                                "%s %.2f" % (component.replace("_", " "), _number(judged, key)),
                                SpatialEvidence("frame")))
    if index == len(steps) - 1:
        if _number(current, "final_color") < 0.999:
            events.append(Event(
                int(current["k"]), "verdict", "final_color",
                "Final light colours are incorrect",
                "final-colour score %.3f" % _number(current, "final_color"),
                SpatialEvidence("frame")))
        if _number(current, "final_countdown") < 0.999:
            events.append(Event(
                int(current["k"]), "verdict", "final_countdown",
                "Final countdown values are incorrect",
                "final-countdown score %.3f" % _number(current, "final_countdown"),
                SpatialEvidence("frame")))
    return _base(task_name, "Traffic light cycle", steps, index, components,
                 Formula("0.4 * final_state + 0.6 * process",
                         "0.40 × Final state + 0.60 × Process",
                         ("final_state", "process")), events)


@register(O27)
def explain_o27(task_name, steps, index):
    current = _require_step(steps, index)
    movement = _number(current, "movement")
    synchronization = _number(current, "synchronization")
    sync_gate = 0.8 + 0.2 * synchronization
    components = [
        Component("movement", "Movement quality", movement,
                  children=("obj0_translation", "obj1_translation")),
        Component("synchronization", "Movement synchronization", synchronization,
                  timing="cumulative"),
        Component("synchronization_gate", "Synchronization gate", sync_gate,
                  kind="gate", timing="cumulative"),
    ]
    per_color = current.get("per_color") or {}
    for obj in ("obj0", "obj1"):
        details = per_color.get(obj) or {}
        value = float(details.get("translation_score", 0.0) or 0.0)
        components.append(Component(
            obj + "_translation", "Object %s translation" % obj[-1], value,
            timing="cumulative",
            note="landing %.2f · path %.2f" %
                 (float(details.get("landing", 0) or 0),
                  float(details.get("path_score", 0) or 0))))
    events = []
    if index == len(steps) - 1:
        for obj in ("obj0", "obj1"):
            details = per_color.get(obj) or {}
            translation = float(details.get("translation_score", 0) or 0)
            path_score = float(details.get("path_score", 1) or 0)
            landing = float(details.get("landing", 1) or 0)
            coverage = float(details.get("coverage", 1) or 0)
            teleports = int(details.get("teleport_events", 0) or 0)
            if landing < 0.999:
                events.append(Event(
                    int(current["k"]), "verdict", obj + "_translation",
                    "Object %s missed its target" % obj[-1],
                    "landing %.3f; translation %.3f" % (landing, translation),
                    _spatial_evidence(details, "position")))
            if path_score < 0.999:
                events.append(Event(
                    int(current["k"]), "verdict", obj + "_translation",
                    "Object %s deviates from the direct path" % obj[-1],
                    "lateral spread %.0f px; path score %.2f" %
                    (float(details.get("lateral_p90_px", 0) or 0), path_score),
                    SpatialEvidence("trajectory")))
            if coverage < 0.999:
                events.append(Event(
                    int(current["k"]), "penalty", obj + "_translation",
                    "Object %s was not tracked throughout" % obj[-1],
                    "trajectory coverage %.3f" % coverage,
                    SpatialEvidence("trajectory")))
            if teleports:
                events.append(Event(
                    int(current["k"]), "penalty", obj + "_translation",
                    "Object %s motion contains jumps" % obj[-1],
                    "%d teleport event(s)" % teleports,
                    SpatialEvidence("trajectory")))
        if synchronization < 1:
            events.append(Event(
                int(current["k"]), "verdict", "synchronization",
                "Objects do not move together",
                "synchronization %.2f; gate %.2f" % (synchronization, sync_gate),
                SpatialEvidence("frame")))
    return _base(task_name, "Move two objects to two targets", steps, index,
                 tuple(components),
                 Formula("movement * synchronization_gate",
                         "Movement quality × Synchronization gate",
                         ("movement", "synchronization_gate")), events)


@register(G273)
def explain_g273(task_name, steps, index):
    current = _require_step(steps, index)
    components = (
        Component("final_state", "Final state", _number(current, "final_state"),
                  timing="final", weight=0.6,
                  children=("object_position", "scene_preservation")),
        Component("process", "Physical process", _number(current, "process"),
                  timing="cumulative", weight=0.4,
                  children=("non_teleport", "one_per_column", "scene_stability")),
        Component("object_position", "Object positions",
                  _number(current, "object_position"), timing="final"),
        Component("scene_preservation", "Scene preserved",
                  _number(current, "scene_preservation"), timing="final"),
        Component("non_teleport", "Continuous falling",
                  _number(current, "non_teleport"), timing="cumulative"),
        Component("one_per_column", "One object per cup",
                  _number(current, "one_per_column"), timing="cumulative"),
        Component("scene_stability", "Intermediate scene stability",
                  _number(current, "scene_stability"), timing="cumulative"),
        Component("area_consistency", "Object area consistency",
                  _number(current, "area_consistency"), timing="cumulative"),
    )
    events = []
    if index == len(steps) - 1:
        existing = set()
        for object_index in range(int(current.get("n_objects") or 0)):
            detail = str(current.get("obj%d" % object_index, ""))
            if "not_found" in detail:
                events.append(Event(
                    int(current["k"]), "verdict", "object_position",
                    "Object %d not found at its expected final position" % object_index,
                    detail, SpatialEvidence("frame")))
                existing.add("object_position")
        if _number(current, "object_position") < 0.999 and "object_position" not in existing:
            events.append(Event(
                int(current["k"]), "verdict", "object_position",
                "Objects settled at incorrect heights",
                "position score %.3f" % _number(current, "object_position"),
                SpatialEvidence("frame")))
        if _number(current, "non_teleport") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "non_teleport",
                "Object motion contains jumps",
                "continuous-motion score %.3f" % _number(current, "non_teleport"),
                SpatialEvidence("trajectory")))
        if _number(current, "one_per_column") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "one_per_column",
                "Multiple objects entered one column",
                "one-per-column score %.3f" % _number(current, "one_per_column"),
                SpatialEvidence("frame")))
        if _number(current, "scene_stability") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "scene_stability",
                "Intermediate scene was unstable",
                "stability %.3f" % _number(current, "scene_stability"),
                SpatialEvidence("frame")))
        if _number(current, "scene_preservation") < 1:
            events.append(Event(
                int(current["k"]), "verdict", "scene_preservation",
                "Scene was not preserved",
                "background %.2f · liquid %.2f" %
                (_number(current, "pres_bg_sc"), _number(current, "pres_liquid_sc")),
                SpatialEvidence("frame")))
    return _base(task_name, "High-density liquid", steps, index, components,
                 Formula("0.6 * final_state + 0.4 * process",
                         "0.60 × Final state + 0.40 × Process",
                         ("final_state", "process")), events)


def _pretty_task(task_name):
    stem = task_name.split("_data-generator")[0]
    parts = stem.split("_", 1)
    fallback = (parts[1] if len(parts) == 2 else parts[0]).replace("_", " ").title()
    return title_for(task_name, fallback)


def _field_transition_events(steps, index, fields, threshold=0.08):
    """Keep the strongest gain and loss for each cumulative score.

    Prefix evaluators often update the same metric on dozens of frames. Showing
    every update creates repeated HUD rows without adding an explanation.
    """
    candidates = {}
    previous = {}
    for step in steps[:index + 1]:
        for field, label, _, timing, _ in fields:
            if timing != "cumulative" or field not in step:
                continue
            raw = step[field]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            value = float(raw)
            if field in previous:
                delta = value - previous[field]
                if abs(delta) >= threshold:
                    kind = "milestone" if delta > 0 else "penalty"
                    title = (label + " gained credit" if delta > 0
                             else label + " lost credit")
                    event = Event(
                        int(step["k"]), kind, field, title,
                        "%.3f → %.3f (%+.3f)" %
                        (previous[field], value, delta),
                        _spatial_evidence(step, field))
                    direction = "up" if delta > 0 else "down"
                    key = (field, direction)
                    old = candidates.get(key)
                    if old is None or abs(delta) > old[0]:
                        candidates[key] = (abs(delta), event)
            previous[field] = value
    return sorted((item[1] for item in candidates.values()),
                  key=lambda event: event.frame)


def _secondary_failure(component):
    """Concise root-cause wording for reusable secondary components."""
    if component.id in {"foreground_preservation", "foreground_gate"}:
        return "Original shapes changed"
    if component.id in {"background_preservation", "background_clean",
                        "bg_score", "background_gate"}:
        return "Background changed"
    if component.id in {"consistency", "consistency_score", "preservation",
                        "preservation_gate", "quality_gate"}:
        return "Non-target content changed"
    if component.kind == "gate":
        return "%s limited the score" % component.label
    if component.timing == "cumulative":
        return "%s incomplete" % component.label
    return "%s below target" % component.label


def _cell_evidence(cell, source):
    if isinstance(cell, (list, tuple)) and len(cell) == 2:
        return SpatialEvidence("grid_cell", {
            "row": int(cell[0]), "col": int(cell[1]), "source": source})
    return SpatialEvidence("frame")


def _navigation_final_events(task_name, details):
    """Explain grid-route failures using the evaluator's concrete diagnostics."""
    frame = int(details["k"])
    events = []
    if task_name.startswith("G-13_"):
        reached = details.get("segment_reached") or []
        waypoints = details.get("waypoint_cells") or []
        for segment, ok in enumerate(reached):
            if not ok:
                target = waypoints[segment] if segment < len(waypoints) else None
                events.append(Event(
                    frame, "verdict", "core", "Sequence waypoint not reached",
                    "route segment %d failed" % (segment + 1),
                    _cell_evidence(target, "waypoint_cells")))
                break
    elif task_name.startswith("G-15_"):
        hits = details.get("obstacle_hit_cells") or []
        if hits:
            events.append(Event(frame, "penalty", "core", "Obstacle entered",
                                "%d obstacle cells crossed" % len(hits),
                                _cell_evidence(hits[0], "obstacle_hit_cells")))
        if float(details.get("coverage", 1.0)) < 0.999:
            events.append(Event(frame, "verdict", "core", "Goal route incomplete",
                                "route coverage %.3f" % float(details["coverage"]),
                                _cell_evidence(details.get("end_cell"), "end_cell")))
    elif task_name.startswith("G-16_"):
        missed = details.get("missed_required") or []
        if missed:
            events.append(Event(frame, "verdict", "core", "Required block missed",
                                "%d required blocks missed" % len(missed),
                                _cell_evidence(missed[0], "missed_required")))
        if float(details.get("coverage", 1.0)) < 0.999:
            events.append(Event(frame, "verdict", "core", "Route coverage incomplete",
                                "route coverage %.3f" % float(details["coverage"]),
                                _cell_evidence(details.get("end_cell"), "end_cell")))
    elif task_name.startswith("G-18_"):
        if float(details.get("coverage", 1.0)) < 0.999:
            events.append(Event(frame, "verdict", "core", "Shortest route incomplete",
                                "shortest-path coverage %.3f" % float(details["coverage"]),
                                _cell_evidence(details.get("end_cell"), "end_cell")))
    elif task_name.startswith("G-41_"):
        violations = details.get("violation_details") or []
        if violations:
            violation = violations[0]
            events.append(Event(
                int(violation.get("frame", frame)), "penalty", "core",
                "Illegal %s move" % str(violation.get("type", "grid")),
                "%s → %s" % (violation.get("from"), violation.get("to")),
                _cell_evidence(violation.get("to"), "violation_details")))
        if float(details.get("model_cost", 0)) < float(details.get("optimal_cost", 0)):
            events.append(Event(
                frame, "verdict", "core", "Maximum-cost route not achieved",
                "collected %.0f of optimal %.0f" %
                (float(details.get("model_cost", 0)), float(details.get("optimal_cost", 0))),
                SpatialEvidence("trajectory")))
    elif task_name.startswith("G-47_"):
        missed = details.get("missed_keys") or []
        if missed:
            events.append(Event(frame, "verdict", "core", "Required key missed",
                                "%d keys missed before the door" % len(missed),
                                _cell_evidence(missed[0], "missed_keys")))
        hits = details.get("wall_hit_cells") or []
        if hits:
            events.append(Event(frame, "penalty", "core", "Wall crossed",
                                "%d wall cells crossed" % len(hits),
                                _cell_evidence(hits[0], "wall_hit_cells")))
    return events


def _register_formula_tasks(tasks, expression, display, fields):
    """Register tasks sharing an explicit formula and detail-field contract."""
    for task_name in tasks:
        def adapter(name, steps, index, _fields=fields, _expression=expression,
                    _display=display):
            current = _require_step(steps, index)
            resolved_fields = tuple(
                (field, label_for(name, field, label), kind, timing, weight)
                for field, label, kind, timing, weight in _fields)
            components = tuple(
                Component(field, label,
                          _number(current, field), kind=kind,
                          timing=timing, weight=weight)
                for field, label, kind, timing, weight in resolved_fields)
            events = _field_transition_events(steps, index, resolved_fields)
            if index == len(steps) - 1:
                main_failure_used = False
                for component in components:
                    if component.value < 0.999:
                        if not main_failure_used and component.kind == "quality":
                            title = failure_for(name, component.id,
                                                "%s is below target" % component.label)
                            main_failure_used = True
                        else:
                            title = component_failure_for(
                                name, component.id, _secondary_failure(component))
                        events.append(Event(
                            int(current["k"]), "verdict", component.id,
                            title,
                            "%s %.3f" % (component.label, component.value),
                            _spatial_evidence(current, component.id)))
            return _base(
                name, _pretty_task(name), steps, index, components,
                Formula(_expression, _display,
                        tuple(field[0] for field in _fields)), events)
        register(task_name)(adapter)


def _register_weighted(tasks, weights, dual_gate=False):
    fields = tuple(
        (field, field.replace("_", " ").title(), "quality",
         "cumulative" if any(word in field for word in ("process", "path", "motion"))
         else "final", weight)
        for field, weight in weights)
    weighted_expression = " + ".join("%s * %s" % (weight, field)
                                     for field, weight in weights)
    if not dual_gate:
        _register_formula_tasks(tasks, weighted_expression,
                                " + ".join("%.0f%% %s" %
                                           (weight * 100, field.replace("_", " "))
                                           for field, weight in weights), fields)
        return
    for task_name in tasks:
        def adapter(name, steps, index, _fields=fields,
                    _weighted=weighted_expression):
            current = _require_step(steps, index)
            components = [
                Component(field, label_for(name, field, label),
                          _number(current, field), kind=kind,
                          timing=timing, weight=weight)
                for field, label, kind, timing, weight in _fields]
            completion_gate = min(1.0, _number(current, "completion") / 0.5)
            foreground_gate = min(1.0, _number(current, "foreground_preservation") / 0.5)
            components.extend([
                Component("completion_gate", "Completion gate", completion_gate,
                          kind="gate", timing="final"),
                Component("foreground_gate", "Shape preservation gate", foreground_gate,
                          kind="gate", timing="final"),
            ])
            events = []
            if index == len(steps) - 1:
                completion = _number(current, "completion")
                foreground = _number(current, "foreground_preservation")
                background = _number(current, "background_preservation")
                if completion < 0.999:
                    events.append(Event(
                        int(current["k"]), "verdict", "completion",
                        failure_for(name, "completion", "Required result missing"),
                        "completion %.3f; result not detected" % completion,
                        SpatialEvidence("frame")))
                if foreground < 0.98:
                    events.append(Event(
                        int(current["k"]), "penalty", "foreground_preservation",
                        "Original shapes changed",
                        "foreground preservation %.3f" % foreground,
                        SpatialEvidence("frame")))
                if background < 0.98:
                    events.append(Event(
                        int(current["k"]), "penalty", "background_preservation",
                        "Background changed",
                        "background preservation %.3f" % background,
                        SpatialEvidence("frame")))
            return _base(
                name, _pretty_task(name), steps, index, tuple(components),
                Formula("(%s) * completion_gate * foreground_gate" % _weighted,
                        "Weighted quality × Completion gate × Shape gate",
                        tuple(field[0] for field in _fields)
                        + ("completion_gate", "foreground_gate")), events)
        register(task_name)(adapter)


def _register_gate(tasks, core_field, preservation_field, floor=0.6,
                   core_label="Task quality", preservation_label="Scene preservation"):
    for task_name in tasks:
        def adapter(name, steps, index, _core=core_field, _pres=preservation_field,
                    _floor=floor, _core_label=core_label, _pres_label=preservation_label):
            current = _require_step(steps, index)
            core = _number(current, _core)
            preservation = _number(current, _pres)
            gate = _floor + (1.0 - _floor) * preservation
            components = (
                Component("core", label_for(name, _core, _core_label), core,
                          timing="cumulative" if "process" in _core else "final"),
                Component("preservation", _pres_label, preservation, timing="final"),
                Component("quality_gate", _pres_label + " gate", gate,
                          kind="gate", timing="final"),
            )
            events = []
            if index == len(steps) - 1:
                navigation_events = _navigation_final_events(name, current)
                events.extend(navigation_events)
                if core < 0.999:
                    if not navigation_events:
                        events.append(Event(int(current["k"]), "verdict", "core",
                                            failure_for(name, _core,
                                                        _core_label + " is incomplete"),
                                            "%s %.3f" % (_core_label, core),
                                            _spatial_evidence(current, _core)))
                if preservation < 0.98:
                    events.append(Event(int(current["k"]), "penalty", "quality_gate",
                                        "Non-target content changed",
                                        "%s %.3f; gate %.3f" %
                                        (_pres_label, preservation, gate),
                                        _spatial_evidence(current, _pres)))
            return _base(name, _pretty_task(name), steps, index, tuple(components),
                         Formula("core * quality_gate",
                                 _core_label + " × " + _pres_label + " gate",
                                 ("core", "quality_gate")), events)
        register(task_name)(adapter)


def _register_custom(task_name, expression, display, values):
    """Register a non-linear formula whose components include derived gates."""
    def adapter(name, steps, index):
        current = _require_step(steps, index)
        components = tuple(
            Component(component_id, label_for(name, component_id, label),
                      float(getter(current)), kind=kind,
                      timing=timing)
            for component_id, label, getter, kind, timing in values)
        events = []
        if index == len(steps) - 1:
            main_failure_used = False
            for component in components:
                if component.value < 0.999 and component.kind != "penalty":
                    if not main_failure_used and component.kind == "quality":
                        title = failure_for(name, component.id,
                                            "%s is below target" % component.label)
                        main_failure_used = True
                    else:
                        title = component_failure_for(
                            name, component.id, _secondary_failure(component))
                    events.append(Event(int(current["k"]), "verdict", component.id,
                                        title,
                                        "%s %.3f" % (component.label, component.value),
                                        _spatial_evidence(current, component.id)))
                elif component.kind == "penalty" and component.value > 0:
                    title = component_failure_for(
                        name, component.id,
                        (component.label if "changed" in component.label.lower()
                         or "damage" in component.label.lower()
                         else component.label + " applied"))
                    events.append(Event(int(current["k"]), "penalty", component.id,
                                        title,
                                        "deduction %.3f" % component.value,
                                        _spatial_evidence(current, component.id)))
        return _base(name, _pretty_task(name), steps, index, components,
                     Formula(expression, display,
                             tuple(component.id for component in components)), events)
    register(task_name)(adapter)


_PRESERVE_FORMULA = "primary * (0.6 + 0.4 * consistency)"
_PRESERVE_FIELDS = (
    ("primary", "Task accuracy", "quality", "final", None),
    ("consistency", "Scene consistency", "gate", "final", None),
)


def _register_preservation_tasks(task_field_pairs):
    for task_name, primary_field in task_field_pairs:
        def adapter(name, steps, index, _primary=primary_field):
            current = _require_step(steps, index)
            primary = _number(current, _primary)
            if "consistency" in current:
                consistency = _number(current, "consistency")
            elif "consistency_score" in current:
                consistency = _number(current, "consistency_score")
            else:
                fore = float(current.get("fore_consistency", 0.0) or 0.0)
                back = float(current.get("back_consistency", fore) or 0.0)
                consistency = (fore + back) / 2.0
            gate = 0.6 + 0.4 * consistency
            components = [
                Component("primary", label_for(name, "primary", "Task accuracy"),
                          primary, timing="final"),
                Component("consistency", "Scene consistency", consistency,
                          timing="final"),
                Component("preservation_gate", "Preservation gate", gate,
                          kind="gate", timing="final"),
            ]
            if "correct_match_score" in current:
                components.append(Component(
                    "correct_selection", "Required targets selected",
                    float(current["correct_match_score"]), timing="final"))
            if "wrong_match_score" in current:
                wrong = max(0.0, float(current["wrong_match_score"]))
                components.append(Component(
                    "false_positive_control", "Incorrect targets avoided",
                    max(0.0, 1.0 - wrong), kind="gate", timing="final"))
            events = []
            if index == len(steps) - 1:
                correct = current.get("correct_match_score")
                wrong = float(current.get("wrong_match_score", 0.0) or 0.0)
                ambiguous = float(current.get("ambiguous_score", 0.0) or 0.0)
                if wrong > 0:
                    target = selection_target_for(name)
                    events.append(Event(int(current["k"]), "verdict", "primary",
                                        selection_event_title(name, "wrong"),
                                        "%s false-positive penalty %.3f; selection score %.3f" %
                                        (target, wrong, primary),
                                        _spatial_evidence(current, "primary")))
                elif correct is not None and float(correct) < 0.999:
                    target = selection_target_for(name)
                    events.append(Event(int(current["k"]), "verdict", "primary",
                                        selection_event_title(name, "missed"),
                                        "%s credit %.3f; selection score %.3f" %
                                        (target, float(correct), primary),
                                        _spatial_evidence(current, "primary")))
                elif primary < 0.999:
                    events.append(Event(int(current["k"]), "verdict", "primary",
                                        failure_for(name, _primary,
                                                    "Task result is inaccurate"),
                                        "task accuracy %.3f" % primary,
                                        SpatialEvidence("frame")))
                if ambiguous > 0:
                    events.append(Event(int(current["k"]), "penalty", "primary",
                                        "Ambiguous marking detected",
                                        "ambiguous-mark penalty %.3f" % ambiguous,
                                        SpatialEvidence("frame")))
                if consistency < 0.98:
                    events.append(Event(int(current["k"]), "penalty",
                                        "preservation_gate", "Non-target content changed",
                                        "scene consistency %.3f; gate %.3f" %
                                        (consistency, gate), SpatialEvidence("frame")))
            return _base(name, _pretty_task(name), steps, index, tuple(components),
                         Formula("primary * preservation_gate",
                                 "Task accuracy × Preservation gate",
                                 ("primary", "preservation_gate")), events)
        register(task_name)(adapter)


_register_preservation_tasks([
    ("G-3_stable_sort_data-generator", "arrangement"),
    ("G-9_identify_objects_in_region_data-generator", "accuracy"),
    ("G-29_chart_extreme_with_data_data-generator", "accuracy"),
    ("G-43_understand_scene_structure_data-generator", "accuracy"),
    ("G-138_spot_unique_non_repeated_color_data-generator", "accuracy"),
    ("G-140_locate_topmost_unobscured_figure_data-generator", "accuracy"),
    ("G-174_arrange_circles_by_circumference_data-generator", "arrangement"),
    ("G-194_construct_concentric_ring_data-generator", "arrangement"),
    ("G-221_outline_innermost_square_data-generator", "accuracy"),
    ("G-240_add_borders_to_unbordered_shapes_data-generator", "accuracy"),
])


_MATCH_TASKS = [
    "G-131_select_next_figure_increasing_size_sequence_data-generator",
    "G-134_select_next_figure_large_small_alternating_sequence_data-generator",
    "G-135_select_next_figure_small_large_alternating_sequence_data-generator",
    "G-136_locate_point_in_overlapping_area_data-generator",
    "G-147_identify_unique_figure_in_uniform_set_data-generator",
    "G-160_circle_largest_numerical_value_data-generator",
    "G-167_select_longest_polygon_side_data-generator",
    "G-168_identify_nearest_to_square_rectangle_data-generator",
    "G-169_locate_intersection_of_segments_data-generator",
    "G-206_identify_pentagons_data-generator",
    "G-212_find_incorrect_arrow_direction_data-generator",
    "G-217_circle_central_dot_data-generator",
    "G-218_identify_largest_angle_in_triangle_data-generator",
    "G-219_select_leftmost_shape_data-generator",
    "G-222_mark_tangent_point_of_circles_data-generator",
    "G-223_highlight_horizontal_lines_data-generator",
    "G-247_identify_chinese_character_data-generator",
    "G-248_mark_asymmetrical_shape_data-generator",
]
_register_preservation_tasks([(task, "match_score") for task in _MATCH_TASKS])


@register("G-158_identify_all_hollow_points_data-generator")
def explain_g158(task_name, steps, index):
    """Distinguish missed hollow points from incorrectly selected solid points."""
    current = _require_step(steps, index)
    match_score = _number(current, "match_score")
    consistency = _number(current, "consistency_score")
    preservation_gate = 0.6 + 0.4 * consistency
    correct = float(current.get("correct_match_score", 0.0) or 0.0)
    wrong = float(current.get("wrong_match_score", 0.0) or 0.0)
    selected = int(current.get("num_circles", 0) or 0)
    targets = int(current.get("num_target_shapes", 0) or 0)
    components = (
        Component("match_score", "Selection accuracy", match_score,
                  timing="final", children=("correct_selection", "wrong_selection"),
                  note="%d circles / %d targets" %
                  (selected, targets)),
        Component("correct_selection", "Hollow points selected", correct,
                  timing="final"),
        Component("wrong_selection", "Solid points avoided", max(0.0, 1.0 - wrong),
                  kind="gate", timing="final"),
        Component("consistency", "Scene consistency", consistency,
                  timing="final"),
        Component("preservation_gate", "Preservation gate", preservation_gate,
                  kind="gate", timing="final"),
    )
    events = []
    if index == len(steps) - 1:
        if wrong > 0:
            events.append(Event(
                int(current["k"]), "penalty", "wrong_selection",
                "Solid points incorrectly selected",
                "%d marks / %d hollow targets; FP penalty %.3f" %
                (selected, targets, wrong), SpatialEvidence("frame")))
        if correct < 0.999:
            events.append(Event(
                int(current["k"]), "verdict", "correct_selection",
                "Hollow points missed",
                "correct-selection credit %.3f" % correct,
                SpatialEvidence("frame")))
        if consistency < 0.98:
            events.append(Event(
                int(current["k"]), "penalty", "preservation_gate",
                "Non-target content changed",
                "scene consistency %.3f; gate %.3f" %
                (consistency, preservation_gate), SpatialEvidence("frame")))
    return _base(
        task_name, "Identify all hollow points", steps, index, components,
        Formula("match_score * preservation_gate",
                "Selection accuracy × Preservation gate",
                ("match_score", "preservation_gate")), events)


_register_formula_tasks(
    ["O-5_symbol_deletion_data-generator"],
    "0.6 * delete_score + 0.2 * keep_score + 0.2 * bg_score",
    "0.60 × Deleted + 0.20 × Kept + 0.20 × Background",
    (("delete_score", "Target deleted", "quality", "final", 0.6),
     ("keep_score", "Other symbols kept", "quality", "final", 0.2),
     ("bg_score", "Background preserved", "quality", "final", 0.2)))
_register_formula_tasks(
    ["O-58_symbol_delete_data-generator"],
    "0.85 * fg_score + 0.15 * bg_score",
    "0.85 × Symbol edit + 0.15 × Background",
    (("fg_score", "Symbol edit", "quality", "final", 0.85),
     ("bg_score", "Background preserved", "quality", "final", 0.15)))
_register_formula_tasks(
    ["O-59_symbol_insert_data-generator", "O-60_symbol_substitute_data-generator"],
    "0.85 * seq_score + 0.05 * template_score + 0.1 * bg_score",
    "0.85 × Sequence + 0.05 × Template + 0.10 × Background",
    (("seq_score", "Edited sequence", "quality", "final", 0.85),
     ("template_score", "Template preserved", "quality", "final", 0.05),
     ("bg_score", "Background preserved", "quality", "final", 0.10)))
_register_formula_tasks(
    ["O-61_symbol_edit_data-generator"],
    "0.7 * seq_score + 0.1 * template_score + 0.2 * bg_score",
    "0.70 × Sequence + 0.10 × Template + 0.20 × Background",
    (("seq_score", "Edited sequence", "quality", "final", 0.70),
     ("template_score", "Template preserved", "quality", "final", 0.10),
     ("bg_score", "Background preserved", "quality", "final", 0.20)))


_register_weighted(
    ["G-193_draw_next_sized_shape_data-generator",
     "G-51_predict_next_color_data-generator",
     "O-11_shape_color_then_move_data-generator"],
    (("completion", 0.60), ("foreground_preservation", 0.25),
     ("background_preservation", 0.15)))
_register_weighted(
    ["O-10_shape_outline_fill_data-generator",
     "O-12_shape_color_then_scale_data-generator",
     "O-13_shape_outline_then_move_data-generator",
     "O-14_shape_scale_then_outline_data-generator",
     "O-9_shape_scaling_data-generator"],
    (("completion", 0.60), ("foreground_preservation", 0.25),
     ("background_preservation", 0.15)), dual_gate=True)
_register_weighted(
    ["G-21_multiple_occlusions_vertical_data-generator"],
    (("mask_path_vadility", 0.50), ("occlusion_correctness", 0.30),
     ("elements_preservation", 0.20)))
_register_weighted(
    ["G-31_directed_graph_navigation_data-generator"],
    (("completion", 0.25), ("path_validity", 0.40),
     ("foreground_preservation", 0.20), ("background_preservation", 0.15)))
_register_weighted(
    ["G-39_attention_shift_different_data-generator"],
    (("box_position", 0.80), ("consistency", 0.20)))
_register_weighted(
    ["G-161_mark_second_largest_shape_data-generator"],
    (("match_score", 0.625), ("shape_score", 0.375)))
_register_weighted(
    ["G-202_mark_wave_peaks_data-generator"],
    (("match_score", 0.60), ("consistency_score", 0.40)))
_register_weighted(
    ["O-16_color_addition_data-generator"],
    (("mixing_color", 0.60), ("circle_removal", 0.20),
     ("background_clean", 0.20)))
_register_weighted(
    ["O-21_construction_blueprint_data-generator"],
    (("shape_matching", 0.70), ("correct_option_green", 0.20),
     ("other_options_red", 0.10)))
_register_weighted(
    ["O-23_domino_chain_branch_path_prediction_data-generator",
     "O-24_domino_chain_gap_analysis_data-generator"],
    (("final_state", 0.50), ("process", 0.50)))
_register_weighted(
    ["O-37_light_sequence_data-generator",
     "O-38_majority_color_data-generator",
     "O-43_object_subtraction_data-generator"],
    (("completion", 0.80), ("background_preservation", 0.20)))
_register_weighted(
    ["O-49_symmetry_completion_data-generator"],
    (("completion", 0.70), ("consistency", 0.30)))
_register_weighted(
    ["O-53_clock_data-generator"],
    (("completion", 0.40), ("process_validity", 0.40),
     ("element_preservation", 0.20)))
_register_weighted(
    ["O-55_rotation_data-generator"],
    (("final_state", 0.60), ("process", 0.40)))
_register_weighted(
    ["O-56_raven_data-generator"],
    (("completion", 0.60), ("preservation", 0.40)))
_register_weighted(
    ["O-62_gravity_physics_data-generator"],
    (("process", 0.65), ("final_position", 0.35)))


_register_gate(
    ["G-13_grid_number_sequence_data-generator",
     "G-15_grid_avoid_obstacles_data-generator",
     "G-16_grid_go_through_block_data-generator",
     "G-18_grid_shortest_path_data-generator",
     "G-41_grid_highest_cost_data-generator",
     "G-47_multiple_keys_for_one_door_data-generator"],
    "task_score", "bg_preservation", 0.6,
    "Path task score", "Background preservation")
_register_gate(
    ["G-189_draw_midpoint_perpendicular_line_data-generator",
     "O-18_glass_refraction_data-generator",
     "O-19_mirror_reflection_data-generator"],
    "red_line", "consistency", 0.6,
    "Required line", "Scene consistency")
_register_gate(
    ["G-250_color_triple_intersection_red_data-generator"],
    "red_region", "preservation", 0.6,
    "Target region colouring", "Outside-region preservation")
_register_gate(
    ["G-54_connecting_color_data-generator"],
    "correct_connections", "consistency", 0.6,
    "Correct connections", "Scene consistency")
_register_gate(
    ["O-33_counting_object_data-generator"],
    "count_correctness", "consistency", 0.6,
    "Count correctness", "Scene consistency")
_register_gate(
    ["O-45_sequence_completion_data-generator"],
    "generated_object", "consistency", 0.6,
    "Generated object", "Scene consistency")
_register_formula_tasks(
    ["O-85_2d_object_rotation_data-generator"],
    "task_score * background_gate", "Rotation quality × Background gate",
    (("task_score", "Rotation quality", "quality", "final", None),
     ("background_gate", "Background gate", "gate", "final", None)))

_register_formula_tasks(
    ["G-24_separate_objects_no_spin_data-generator"],
    "intermediate_coverage * (0.4 * alignment + 0.6 * alignment_gate * non_alignment_score)",
    "Coverage × (40% alignment + 60% gated process)",
    (("intermediate_coverage", "Intermediate coverage", "quality", "cumulative", None),
     ("alignment", "Final alignment", "quality", "final", 0.4),
     ("alignment_gate", "Alignment gate", "gate", "final", None),
     ("non_alignment_score", "Motion and background", "quality", "cumulative", 0.6)))
_register_formula_tasks(
    ["G-25_seperate_object_spinning_data-generator"],
    "0.15 * alignment + 0.85 * alignment_gate * non_alignment_score",
    "15% alignment + 85% gated process",
    (("alignment", "Final alignment", "quality", "final", 0.15),
     ("alignment_gate", "Alignment gate", "gate", "final", None),
     ("non_alignment_score", "Translation, rotation and background", "quality",
      "cumulative", 0.85)))
_register_formula_tasks(
    ["O-36_grid_shift_data-generator"],
    "0.35 * final_cell_score + 0.65 * process_score * final_cell_gate",
    "35% final cells + 65% final-gated process",
    (("final_cell_score", "Final cell layout", "quality", "final", 0.35),
     ("process_score", "Shift process", "quality", "cumulative", 0.65),
     ("final_cell_gate", "Final-layout gate", "gate", "final", None)))
_register_formula_tasks(
    ["O-39_maze_data-generator"],
    "((proximity + coverage + continuity_factor) / 3) * (0.4 + 0.6 * wall_multiplier)",
    "Mean path quality × Wall-avoidance gate",
    (("proximity", "Path proximity", "quality", "cumulative", None),
     ("coverage", "Path coverage", "quality", "cumulative", None),
     ("continuity_factor", "Motion continuity", "quality", "cumulative", None),
     ("wall_multiplier", "Wall avoidance", "gate", "cumulative", None)))
_register_gate(
    ["O-44_rotation_puzzle_data-generator"],
    "final_state_score", "rotation_process", 0.4,
    "Final rotation state", "Rotation process")
_register_gate(
    ["O-64_animal_matching_data-generator"],
    "final_quality", "process_score", 0.4,
    "Final animal placement", "Movement process")
_register_gate(
    ["O-6_2d_geometric_transformation_data-generator"],
    "final_pose", "orbital_motion", 0.4,
    "Final transformed pose", "Orbital motion")


def _v(field):
    return lambda details: _number(details, field)


_register_custom(
    "O-15_ball_bounces_given_time_data-generator",
    "0.4 * max(physics, trajectory_coverage) + 0.6 * trajectory - foreground_deduction",
    "40% physics/coverage + 60% trajectory − scene deduction",
    (("physics", "Bounce physics", _v("physics"), "quality", "cumulative"),
     ("trajectory_coverage", "Trajectory coverage", _v("traj_coverage"), "quality", "cumulative"),
     ("trajectory", "Trajectory match", _v("trajectory"), "quality", "cumulative"),
     ("foreground_deduction", "Foreground damage deduction",
      lambda d: 0.1 if _number(d, "fg_similarity") < 0.7 else 0.0,
      "penalty", "final")))
_register_custom(
    "O-22_construction_stack_data-generator",
    "max(0, min(1, main_score * process_gate - target_deduction - background_deduction))",
    "Final stack × Process gate − Preservation deductions",
    (("main_score", "Final stack match", _v("main_score"), "quality", "final"),
     ("process_gate", "Construction-process gate",
      lambda d: 0.4 + 0.6 * _number(d, "movement_validity_ratio"), "gate", "cumulative"),
     ("target_deduction", "Target changed", _v("deduction_target_frames"), "penalty", "final"),
     ("background_deduction", "Background changed", _v("deduction_background"), "penalty", "final")))
_register_custom(
    "O-25_LEGO_construction_assembly_data-generator",
    "0.8 * assembly_correctness * assembly_gate + 0.2 * consistency",
    "80% gated assembly + 20% scene consistency",
    (("assembly_correctness", "Assembly correctness", _v("assembly_correctness"), "quality", "final"),
     ("assembly_gate", "Scene-validity gate",
      lambda d: min(1.0, _number(d, "consistency") / 0.5), "gate", "final"),
     ("consistency", "Structure and background", _v("consistency"), "quality", "final")))
_register_custom(
    "O-2_pigment_color_mixing_subtractive_data-generator",
    "0.6 * mixing_color * preservation_gate + 0.2 * object_preservation + 0.2 * background_clean",
    "60% gated mixture + 20% objects + 20% background",
    (("mixing_color", "Mixed colour", _v("mixing_color"), "quality", "final"),
     ("preservation_gate", "Source-object gate",
      lambda d: min(1.0, _number(d, "object_preservation") / 0.4), "gate", "final"),
     ("object_preservation", "Source objects preserved", _v("object_preservation"), "quality", "final"),
     ("background_clean", "Background clean", _v("background_clean"), "quality", "final")))
_register_custom(
    "O-30_bookshelf_data-generator",
    "(0.625 * final_placement + 0.375 * sequential_insertion) * (0.6 + 0.4 * consistency)",
    "Weighted placement/process × Consistency gate",
    (("final_placement", "Final book placement", _v("final_placement"), "quality", "final"),
     ("sequential_insertion", "Sequential insertion", _v("sequential_insertion"), "quality", "cumulative"),
     ("consistency", "Bookshelf consistency", _v("consistency"), "gate", "final")))
_register_custom(
    "O-34_dot_to_dot_task_data-generator",
    "max(0, min(1, completeness * order * numerical - consistency_deduction))",
    "Completeness × Order × Number stability − Scene deduction",
    (("completeness", "Connection completeness", _v("connection_completeness"), "quality", "final"),
     ("order", "Connection order", _v("connection_order_penalty"), "quality", "cumulative"),
     ("numerical", "Number stability", _v("numerical_consistency_penalty"), "quality", "cumulative"),
     ("consistency_deduction", "Scene-change deduction", _v("consistency_subtract"), "penalty", "final")))
_register_custom(
    "O-46_shape_sorter_data-generator",
    "0.6 * final_layout + 0.4 * process_score * final_layout_gate",
    "60% final layout + 40% final-gated sorting process",
    (("final_layout", "Final sorted layout", _v("final_layout"), "quality", "final"),
     ("process_score", "Transport process", _v("process_score"), "quality", "cumulative"),
     ("final_layout_gate", "Final-layout gate",
      lambda d: min(1.0, _number(d, "final_layout") / 0.2), "gate", "final")))
_register_custom(
    "O-54_control_panel_data-generator",
    "(0.4 * completion + 0.6 * process_validity) * (0.6 + 0.4 * background_preservation)",
    "(40% completion + 60% process) × Background gate",
    (("completion", "Final control state", lambda d: float(d["scores"]["completion"]), "quality", "final"),
     ("process_validity", "Control sequence", lambda d: float(d["scores"]["process_validity"]), "quality", "cumulative"),
     ("background_preservation", "Background preservation",
      lambda d: float(d["scores"]["background_preservation"]), "gate", "final")))
_register_custom(
    "O-65_animal_size_sorting_data-generator",
    "arrangement * (0.6 + 0.4 * consistency) * count_penalty",
    "Arrangement × Preservation gate × Count penalty",
    (("arrangement", "Size order", _v("arrangement"), "quality", "final"),
     ("consistency", "Animal/background consistency",
      lambda d: 0.25 * _number(d, "fore_consistency")
                + 0.75 * _number(d, "back_consistency"),
      "gate", "final"),
     ("count_penalty", "Animal count", _v("count_penalty"), "gate", "final")))
_register_custom(
    "O-75_communicating_vessels_data-generator",
    "final_and_volume * equilibration_gate * (0.6 + 0.4 * consistency)",
    "Liquid result × Equilibration gate × Preservation gate",
    (("final_and_volume", "Final levels and volume", _v("final_and_volume_score"), "quality", "final"),
     ("equilibration_gate", "Equilibration process gate", _v("equilibration_gate"), "gate", "cumulative"),
     ("consistency", "Vessels and background", _v("consistency"), "gate", "final")))


@register("G-5_multi_object_placement_data-generator")
def explain_g5(task_name, steps, index):
    """Expose G-5's per-object transport mean and moved-star multiplier."""
    current = _require_step(steps, index)
    object_score = _number(current, "object_score_avg")
    star_penalty = _number(current, "star_penalty")
    star_gate = 0.4 + 0.6 * star_penalty
    per_object = current.get("per_object_scores", [])
    components = (
        Component("object_score", "Mean object transport", object_score,
                  timing="cumulative",
                  note="%d matched / %d GT" %
                       (int(current.get("n_matched", 0)),
                        int(current.get("n_gt_objects", 0)))),
        Component("star_penalty", "Unmoved-star score", star_penalty,
                  kind="gate", timing="cumulative",
                  note="%d stars moved" % int(current.get("n_moved_stars", 0))),
        Component("star_gate", "Star-protection gate", star_gate,
                  kind="gate", timing="cumulative"),
    ) + tuple(
        Component("object_%d" % (i + 1), "Object %d transport" % (i + 1),
                  float(value), timing="cumulative")
        for i, value in enumerate(per_object)
    )
    events = []
    seen_moved = 0
    for step in steps[:index + 1]:
        moved = int(step.get("n_moved_stars", 0) or 0)
        while seen_moved < moved:
            seen_moved += 1
            penalty = float(step.get("star_penalty", 0.5 ** seen_moved))
            events.append(Event(
                int(step["k"]), "penalty", "star_gate",
                "Protected star %d moved" % seen_moved,
                "star penalty %.3f; effective gate %.3f" %
                (penalty, 0.4 + 0.6 * penalty), SpatialEvidence("frame")))
    if index == len(steps) - 1:
        for i, value in enumerate(per_object):
            if float(value) < 0.999:
                details = current.get("per_object_details", [])
                detail = details[i] if i < len(details) else {}
                events.append(Event(
                    int(current["k"]), "verdict", "object_%d" % (i + 1),
                    "Object %d placement/path lost credit" % (i + 1),
                    "transport %.3f; endpoint distance %s px" %
                    (float(value), detail.get("endpoint_dist_px", "unknown")),
                    SpatialEvidence("point", {
                        "x": detail.get("endpoint_x"),
                        "y": detail.get("endpoint_y"),
                    }) if detail.get("endpoint_x") is not None else
                    SpatialEvidence("frame")))
    return _base(
        task_name, "Multi-object placement", steps, index, components,
        Formula("object_score * star_gate",
                "Mean per-object transport × Star-protection gate",
                ("object_score", "star_gate")), events)


@register("G-8_track_object_movement_data-generator")
def explain_g8(task_name, steps, index):
    """Expose the exact completion and process gates used by G-8."""
    current = _require_step(steps, index)
    completion = _number(current, "completion")
    process = _number(current, "process_score")
    process_gate = 0.4 + 0.6 * process
    components = (
        Component("completion", "Marked-object completion", completion,
                  timing="final", children=("shape_preservation",
                                             "target_alignment",
                                             "only_one_moved",
                                             "detection_gate")),
        Component("process_score", "Horizontal movement process", process,
                  timing="cumulative"),
        Component("process_gate", "Movement-process gate", process_gate,
                  kind="gate", timing="cumulative",
                  children=("process_score",)),
        Component("shape_preservation", "Rigid shape preservation",
                  _number(current, "shape_preservation"), timing="final"),
        Component("target_alignment", "Target x-alignment",
                  _number(current, "target_alignment"), timing="final"),
        Component("only_one_moved", "Bystanders stationary",
                  _number(current, "only_one_moved"), kind="gate",
                  timing="cumulative"),
        Component("detection_gate", "Tracking coverage",
                  _number(current, "detection_gate"), kind="gate",
                  timing="cumulative"),
    )
    events = []
    previous_others = 0
    for step in steps[:index + 1]:
        moved = int(step.get("n_others_moved", 0) or 0)
        if moved > previous_others:
            events.append(Event(
                int(step["k"]), "penalty", "only_one_moved",
                "%d bystander object(s) moved" % moved,
                "stationary-object multiplier %.3f" %
                float(step.get("only_one_moved", 0.0)), SpatialEvidence("frame")))
            previous_others = moved
    if index == len(steps) - 1:
        target = current.get("target_center")
        evidence = (SpatialEvidence("point", {"x": target[0], "y": target[1]})
                    if isinstance(target, (list, tuple)) and len(target) == 2
                    else SpatialEvidence("frame"))
        if _number(current, "target_alignment") < 0.999:
            events.append(Event(
                int(current["k"]), "verdict", "target_alignment",
                "Marked object missed the target x-position",
                "x gap %.2f px; alignment %.3f" %
                (_number(current, "target_x_gap"),
                 _number(current, "target_alignment")), evidence))
        if _number(current, "shape_preservation") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "shape_preservation",
                "Marked object changed shape",
                "shape preservation %.3f" % _number(current, "shape_preservation"),
                SpatialEvidence("frame")))
        if _number(current, "detection_gate") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "detection_gate",
                "Marked object was not tracked consistently",
                "tracking coverage %.3f" % _number(current, "detection_gate"),
                SpatialEvidence("trajectory")))
        if _number(current, "process_score") < 0.999:
            events.append(Event(
                int(current["k"]), "penalty", "process_score",
                "Horizontal movement process is incomplete",
                "movement-process score %.3f" % _number(current, "process_score"),
                SpatialEvidence("trajectory")))
        if (_number(current, "only_one_moved") < 0.999
                and not any(event.component == "only_one_moved" for event in events)):
            events.append(Event(
                int(current["k"]), "penalty", "only_one_moved",
                "Another object moved",
                "bystander multiplier %.3f" % _number(current, "only_one_moved"),
                SpatialEvidence("frame")))
    return _base(
        task_name, "Track marked-object movement", steps, index, components,
        Formula("completion * process_gate",
                "Completion × (0.40 + 0.60 × movement process)",
                ("completion", "process_gate")), events)


_register_formula_tasks(
    ["O-47_sliding_puzzle_data-generator"],
    "0.6 * final_state + 0.4 * process",
    "0.60 × Final puzzle state + 0.40 × Valid move process",
    (("final_state", "Final puzzle state", "quality", "final", 0.60),
     ("process", "Valid move sequence", "quality", "cumulative", 0.40)))


def recompute(explanation: Explanation) -> float:
    """Recompute supported formulas; used as an adapter correctness check."""
    if explanation.formula.expression in {"prefix_error", "prefix_pending"}:
        return explanation.final_score
    if explanation.formula.expression == "trajectory_quality * wall_gate * key_gate":
        return (
            explanation.component("trajectory_quality").value
            * explanation.component("wall_gate").value
            * explanation.component("key_gate").value
        )
    if explanation.formula.expression == "weighted_quality * process_gate":
        return (explanation.component("weighted_quality").value
                * explanation.component("process_gate").value)
    if explanation.formula.expression == "0.5 * final_state + 0.5 * eat_process":
        return (0.5 * explanation.component("final_state").value
                + 0.5 * explanation.component("eat_process").value)
    if explanation.formula.expression == "completion * consistency_gate":
        return (explanation.component("completion").value
                * explanation.component("consistency_gate").value)
    if explanation.formula.expression == "0.4 * final_state + 0.6 * process":
        return (0.4 * explanation.component("final_state").value
                + 0.6 * explanation.component("process").value)
    if explanation.formula.expression == "movement * synchronization_gate":
        return (explanation.component("movement").value
                * explanation.component("synchronization_gate").value)
    if explanation.formula.expression == "0.6 * final_state + 0.4 * process":
        return (0.6 * explanation.component("final_state").value
                + 0.4 * explanation.component("process").value)
    values = {component.id: component.value for component in explanation.components}
    tree = ast.parse(explanation.formula.expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.Name) and node.id in values:
            return float(values[node.id])
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"min", "max"} and not node.keywords):
            values_ = [evaluate(argument) for argument in node.args]
            return min(values_) if node.func.id == "min" else max(values_)
        raise NotImplementedError(explanation.formula.expression)

    return evaluate(tree)
