#!/usr/bin/env python3
"""
Evaluation script for the VBVR-Pro-Bench interleaved image setting.

Scores predicted images against the GT image sequence using the task_specific
dimension only, for fair comparison with the video setting.

Ground truth (VBVR-Pro-Bench-Image):
    {gt_image_base}/{split}/{task_name}/{idx}/
        first_frame.png     conditioning image
        frame_1.png ...     reference output steps
        metadata.json       task parameters and symbolic ground truth

Model output:
    {model_path}/{split}/{task_name}/{idx}/
        001.png, 002.png, ...   one file per output step

where {split} is In-Domain_50 or Out-of-Domain_50 and {idx} is the 5-digit
instance id. Steps are ordered by the number in the filename, so both 1.png and
001.png work. An instance with no prediction scores 0.

Usage:
    # One model
    python run_evaluation_image.py \
        --model_path /path/to/model_images/my_model \
        --gt_image_base /path/to/VBVR-Pro-Bench-Image \
        --output_dir ./interleave_eval_results

    # Every model under a directory
    python run_evaluation_image.py \
        --models_base /path/to/model_images \
        --gt_image_base /path/to/VBVR-Pro-Bench-Image
"""

import os
import sys
import re
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


sys.path.insert(0, str(Path(__file__).parent))

from vbvr_bench.evaluators import (
    get_evaluator, TASK_EVALUATOR_MAP, get_task_category,
    is_out_of_domain, get_split,
)


FOLDERS = ["In-Domain_50", "Out-of-Domain_50"]


def _gt_frames(gt_dir):
    fs = glob.glob(os.path.join(gt_dir, "frame_*.png"))
    return sorted(fs, key=lambda p: int(re.search(r"frame_(\d+)", os.path.basename(p)).group(1)))


def _step_key(path):
    """Order predicted step images by the trailing number in the filename, so
    1.png < 2.png < 10.png with or without zero padding."""
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", stem)
    return (0, int(nums[-1]), stem) if nums else (1, 0, stem)


def _collect_preds(task_path):
    """Index predictions by instance id.

    Expected layout:  {task_path}/{idx}/*.png   with idx a 5-digit id.
    Every .png inside an instance directory is one output step of that instance.
    """
    preds = {}
    for item in sorted(os.listdir(task_path)):
        full = os.path.join(task_path, item)
        if os.path.isdir(full) and re.fullmatch(r"\d{5}", item):
            pngs = glob.glob(os.path.join(full, "*.png"))
            if pngs:
                preds[item] = sorted(pngs, key=_step_key)
    return preds


def _new_summary():
    return {sp: {"scores": [], "by_task": {}, "by_category": {}}
            for sp in ("In_Domain", "Out_of_Domain", "overall")}


def _agg(summary, split, task_name, category, score):
    for key in (split, "overall"):
        summary[key]["scores"].append(score)
        summary[key]["by_task"].setdefault(task_name, []).append(score)
        summary[key]["by_category"].setdefault(category, []).append(score)


def _finalize(summary):
    for sp in ("In_Domain", "Out_of_Domain", "overall"):
        sc = summary[sp]["scores"]
        summary[sp]["mean_score"] = sum(sc) / len(sc) if sc else 0.0
        summary[sp]["num_samples"] = len(sc)
        for d in ("by_task", "by_category"):
            for k, v in list(summary[sp][d].items()):
                summary[sp][d][k] = sum(v) / len(v) if isinstance(v, list) and v else 0.0


def evaluate_folder_model(model_name, model_path, gt_image_base, output_dir, device="cuda"):
    results = {
        "model_name": model_name, "model_path": model_path,
        "timestamp": datetime.now().isoformat(), "samples": [],
        "summary": _new_summary(),
    }
    # Enumerate (folder, task, idx) over the full GT set: an idx with no prediction
    # yields empty pred_paths and scores 0.
    jobs = []
    for folder in FOLDERS:
        gt_folder = os.path.join(gt_image_base, folder)
        if not os.path.isdir(gt_folder):
            continue
        split = "In_Domain" if folder.startswith("In") else "Out_of_Domain"
        for task_name in sorted(os.listdir(gt_folder)):
            gt_task = os.path.join(gt_folder, task_name)
            if not os.path.isdir(gt_task):
                continue
            task_path = os.path.join(model_path, folder, task_name)
            preds = _collect_preds(task_path) if os.path.isdir(task_path) else {}
            for idx in sorted(os.listdir(gt_task)):
                if not (re.fullmatch(r"\d{5}", idx) and os.path.isdir(os.path.join(gt_task, idx))):
                    continue
                jobs.append((folder, split, task_name, idx, preds.get(idx, [])))

    for folder, split, task_name, idx, pred_paths in tqdm(jobs, desc=f"Evaluating {model_name}"):
        gt_dir = os.path.join(gt_image_base, folder, task_name, idx)
        category = get_task_category(task_name) if task_name in TASK_EVALUATOR_MAP else "unknown"
        eval_info = {
            "input_image": os.path.join(gt_dir, "first_frame.png"),
            "pred_images": pred_paths,
            "gt_images": _gt_frames(gt_dir),
            "task_name": task_name,
            "gt_path": gt_dir,
            "metafile_path": os.path.join(gt_dir, "metadata.json"),
        }
        if not pred_paths:
            r = {"score": 0.0, "error": "no prediction", "dimensions": {}}
        else:
            try:
                evaluator = get_evaluator(task_name, device)
                r = evaluator.evaluate_interleave(eval_info)
            except Exception as e:
                r = {"score": 0.0, "error": str(e)[:200], "dimensions": {}}
        score = float(r.get("score", 0.0))
        results["samples"].append({
            "task_name": task_name, "video_file": f"{idx}.png", "video_idx": idx,
            "split": split, "category": category,
            "folder": folder,
            "score": score,
            "dimensions": r.get("dimensions", {}),
            "details": r.get("details", {}),
            "error": r.get("error", None),
        })
        _agg(results["summary"], split, task_name, category, score)

    _finalize(results["summary"])
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{model_name}_vbvr_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    s = results["summary"]
    err = sum(1 for x in results["samples"] if x.get("error"))
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    print(f"  In-Domain:      {s['In_Domain']['mean_score']:.4f}  ({s['In_Domain']['num_samples']} samples)")
    print(f"  Out-of-Domain:  {s['Out_of_Domain']['mean_score']:.4f}  ({s['Out_of_Domain']['num_samples']} samples)")
    print(f"  Overall:        {s['overall']['mean_score']:.4f}  ({s['overall']['num_samples']} samples, err {err})")
    if s["overall"]["by_category"]:
        print("  By Category:")
        for cat, sc in sorted(s["overall"]["by_category"].items(), key=lambda x: -x[1]):
            print(f"    {cat:<16}: {sc:.4f}")
    print(f"  ->  {out}")
    return results


def evaluate_folder(models_base, model_path, models, gt_image_base, output_dir, device):
    if model_path:
        name = os.path.basename(model_path.rstrip("/"))
        evaluate_folder_model(name, model_path, gt_image_base, output_dir, device)
        return
    dirs = [d for d in sorted(os.listdir(models_base))
            if os.path.isdir(os.path.join(models_base, d))]
    if models:
        dirs = [d for d in dirs if d in set(models)]
    print(f"Found {len(dirs)} image models: {dirs}")
    for d in dirs:
        evaluate_folder_model(d, os.path.join(models_base, d), gt_image_base, output_dir, device)


def main():
    parser = argparse.ArgumentParser(
        description="Run VBVR-Pro-Bench evaluation for interleaved image generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected model output layout:

  {model_path}/
  ├── In-Domain_50/{task_name}/{idx}/001.png, 002.png, ...
  └── Out-of-Domain_50/{task_name}/{idx}/001.png, 002.png, ...

{idx} is the 5-digit instance id, matching the GT tree. Every .png inside an
instance directory is one output step, ordered by the number in its filename.
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_path", type=str,
                       help="Path to a single model's image directory")
    group.add_argument("--models_base", type=str,
                       help="Base directory containing one subdirectory per model")

    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Restrict --models_base to these model names")
    parser.add_argument("--gt_image_base", type=str, required=True,
                        help="GT root (VBVR-Pro-Bench-Image), containing "
                             "In-Domain_50/ and Out-of-Domain_50/")
    parser.add_argument("--output_dir", type=str, default="./interleave_eval_results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu); affects optional OCR only")

    args = parser.parse_args()

    evaluate_folder(
        models_base=args.models_base,
        model_path=args.model_path,
        models=args.models,
        gt_image_base=args.gt_image_base,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
