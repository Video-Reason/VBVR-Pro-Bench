#!/usr/bin/env python3
"""Run any registered VBVR evaluator on every prefix of an arbitrary video.

The output is the evaluator-faithful timeline consumed by explanation adapters.
No scoring rule is reimplemented here: every point calls the concrete task's
``_evaluate_task_specific`` method and records ``_last_task_details``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


JSON_SCALARS = (str, int, float, bool, type(None))


def clean(value):
    if isinstance(value, JSON_SCALARS):
        return value
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    try:
        return value.item()
    except (AttributeError, ValueError):
        return str(value)


def load_inputs(args):
    from vbvr_pro_bench.utils import get_video_frames, load_image, normalize_frame_size

    generated = get_video_frames(str(args.video), max_frames=args.max_frames)
    if not generated:
        raise SystemExit("no frames decoded from %s" % args.video)
    gt_frames = get_video_frames(str(args.gt_video), max_frames=args.max_frames)
    first = load_image(str(args.gt_first))
    final = load_image(str(args.gt_final))
    target = first if first is not None else final
    if target is None:
        raise SystemExit("neither GT first nor GT final frame could be loaded")
    if generated[0].shape != target.shape:
        generated = [normalize_frame_size(frame, target) for frame in generated]
    if gt_frames and gt_frames[0].shape != target.shape:
        gt_frames = [normalize_frame_size(frame, target) for frame in gt_frames]
    return generated, gt_frames, first, final


def eval_info(args):
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text().strip()
    metadata = [str(path) for path in args.metadata]
    return {
        "video_path": str(args.video),
        "task_name": args.task,
        "no_ssim_fallback": True,
        "gt_path": str(args.gt_dir),
        "gt_video_path": str(args.gt_video),
        "gt_first_frame": str(args.gt_first),
        "gt_final_frame": str(args.gt_final),
        "metafile_path": metadata,
        "prompt": prompt,
    }


def run(args):
    import cv2
    from vbvr_pro_bench.evaluators import TASK_EVALUATOR_MAP

    if args.task not in TASK_EVALUATOR_MAP:
        raise SystemExit("unknown task: %s" % args.task)
    generated, gt_frames, first, final = load_inputs(args)
    info = eval_info(args)
    evaluator = TASK_EVALUATOR_MAP[args.task](device=args.device, task_name=args.task)

    frames_dir = args.frames_dir or args.output.with_name(args.output.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(generated):
        cv2.imwrite(str(frames_dir / ("frame_%03d.png" % index)), frame)

    indexes = list(range(1, len(generated) + 1, max(1, args.stride)))
    if indexes[-1] != len(generated):
        indexes.append(len(generated))
    steps, started = [], time.time()
    for position, end in enumerate(indexes, 1):
        try:
            score = evaluator._evaluate_task_specific(
                generated[:end], gt_frames, first, final, info)
            details = clean(getattr(evaluator, "_last_task_details", {}))
            evaluator_score = round(float(score), 8)
            emitted_score = details.get("final_score")
            if isinstance(emitted_score, (int, float)) and abs(
                    float(emitted_score) - evaluator_score) > 5e-4:
                details["evaluator_detail_final_score"] = emitted_score
            # The evaluator return value is authoritative. Diagnostic fields
            # may contain an intermediate score in third-party evaluators.
            details["final_score"] = round(evaluator_score, 4)
            details["evaluator_score"] = evaluator_score
        except Exception as exc:
            if not args.keep_errors:
                raise
            details = {
                "error": "%s: %s" % (type(exc).__name__, exc),
                "final_score": 0.0,
            }
        steps.append({"k": end, **details})
        if position == 1 or position == len(indexes) or position % 20 == 0:
            print("%d/%d prefixes; frame=%d score=%s elapsed=%.1fs" %
                  (position, len(indexes), end, details.get("final_score"),
                   time.time() - started), flush=True)

    payload = {
        "schema_version": 1,
        "task": args.task,
        "video_path": str(args.video.resolve()),
        "gt_dir": str(args.gt_dir.resolve()),
        "gt_video_path": str(args.gt_video.resolve()),
        "gt_first_frame": str(args.gt_first.resolve()),
        "gt_final_frame": str(args.gt_final.resolve()),
        "metadata": [str(path.resolve()) for path in args.metadata],
        "prompt": info["prompt"],
        "frames_dir": str(frames_dir.resolve()),
        "n_frames": len(generated),
        "stride": args.stride,
        "steps": steps,
        "final": steps[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s" % args.output)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--gt-video", type=Path, required=True)
    ap.add_argument("--gt-first", type=Path, required=True)
    ap.add_argument("--gt-final", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, action="append", default=[])
    prompts = ap.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--keep-errors", action="store_true")
    return ap


if __name__ == "__main__":
    run(parser().parse_args())
