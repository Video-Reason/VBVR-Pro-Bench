#!/usr/bin/env python3
"""One-call evaluator -> Explanation JSON -> HUD pipeline for web backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from visualization.export_timeline import export
from visualization.score_video import parser as score_parser, run as score


def process(*, video: Path, task: str, gt_dir: Path, output_dir: Path,
            device="cpu",
            prefix_stride=5, max_frames=200, explanation_every=1,
            render_stride=1, make_video=True):
    """Process one upload and return paths suitable for an API response.

    The original task evaluator remains the only source of scores. This
    function only coordinates scoring, adaptation, JSON export, and rendering.
    """
    output_dir = Path(output_dir)
    gt_dir = Path(gt_dir)
    gt_video = gt_dir / "ground_truth.mp4"
    gt_first = gt_dir / "first_frame.png"
    gt_final = gt_dir / "final_frame.png"
    metadata = gt_dir / "metadata.json"
    prompt_file = gt_dir / "prompt.txt"
    timeline = output_dir / "timeline.json"
    frames = output_dir / "frames"
    explanation_json = output_dir / "explanation.json"
    explanation_video = output_dir / "explanation.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--video", str(video), "--task", task,
        "--gt-dir", str(gt_dir), "--gt-video", str(gt_video),
        "--gt-first", str(gt_first), "--gt-final", str(gt_final),
        "--output", str(timeline), "--frames-dir", str(frames),
        "--stride", str(prefix_stride), "--max-frames", str(max_frames),
        "--device", device,
    ]
    if metadata.exists():
        argv.extend(["--metadata", str(metadata)])
    if prompt_file.exists():
        argv.extend(["--prompt-file", str(prompt_file)])

    score(score_parser().parse_args(argv))
    export(timeline, explanation_json, explanation_every)
    if make_video:
        # Keep JSON-only service deployments independent of renderer imports.
        from visualization.render_explanation import render
        render(timeline, explanation_video, stride=render_stride)

    payload = json.loads(explanation_json.read_text())
    return {
        "task_name": task,
        "score": payload["final_formula_check"]["evaluator"],
        "formula_check": payload["final_formula_check"],
        "timeline": str(timeline.resolve()),
        "explanation_json": str(explanation_json.resolve()),
        "explanation_video": (str(explanation_video.resolve())
                              if make_video else None),
    }


def main(args):
    result = process(
        video=args.video, task=args.task, gt_dir=args.gt_dir,
        output_dir=args.output_dir, device=args.device,
        prefix_stride=args.prefix_stride, max_frames=args.max_frames,
        explanation_every=args.explanation_every,
        render_stride=args.render_stride, make_video=not args.no_video)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prefix-stride", type=int, default=5,
                    help="score every Nth video prefix (1 = every frame)")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--explanation-every", type=int, default=1)
    ap.add_argument("--render-stride", type=int, default=1)
    ap.add_argument("--no-video", action="store_true")
    return ap


if __name__ == "__main__":
    main(parser().parse_args())
