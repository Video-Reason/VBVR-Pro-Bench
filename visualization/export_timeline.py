#!/usr/bin/env python3
"""Convert an evaluator prefix timeline into renderer-ready explanations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from visualization.adapters import explain, get_adapter, recompute


def export(source: Path, destination: Path, every: int = 1) -> None:
    timeline = json.loads(source.read_text())
    task_name = timeline.get("task")
    if not task_name:
        raise SystemExit("timeline has no task field: %s" % source)
    adapter = get_adapter(task_name)
    steps = timeline.get("steps") or []
    if not steps:
        raise SystemExit("timeline has no steps: %s" % source)

    indexes = list(range(0, len(steps), max(1, every)))
    if indexes[-1] != len(steps) - 1:
        indexes.append(len(steps) - 1)
    frames = [explain(task_name, steps, index) for index in indexes]
    final_recomputed = recompute(frames[-1])
    difference = abs(final_recomputed - frames[-1].final_score)
    if difference > 5e-4:
        raise SystemExit(
            "adapter formula mismatch: recomputed %.6f vs evaluator %.6f" %
            (final_recomputed, frames[-1].final_score))

    payload = {
        "schema_version": 1,
        "source_timeline": str(source),
        "task_name": task_name,
        "n_source_steps": len(steps),
        "n_explanation_frames": len(frames),
        "final_formula_check": {
            "evaluator": frames[-1].final_score,
            "recomputed": round(final_recomputed, 8),
            "difference": round(difference, 10),
        },
        "frames": [frame.to_dict() for frame in frames],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s (%d explanation frames)" % (destination, len(frames)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--every", type=int, default=1,
                        help="export every Nth prefix; final prefix is always included")
    args = parser.parse_args()
    output = args.out or args.timeline.with_name(args.timeline.stem + "_explanation.json")
    export(args.timeline, output, args.every)
