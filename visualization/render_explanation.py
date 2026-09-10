#!/usr/bin/env python3
"""Render any adapted evaluator timeline with the shared live-scoring HUD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # hudkit
sys.path.insert(0, str(HERE.parent))   # visualization package

import cv2
from PIL import Image, ImageDraw

import hudkit as K
from visualization.adapters import explain, get_adapter


def frame_directory(timeline, source):
    value = timeline.get("frames_dir")
    if not value:
        raise SystemExit("timeline has no frames_dir")
    path = Path(value)
    if not path.is_absolute():
        # A relative frames_dir is resolved against the timeline, then against
        # a sibling frames/ directory.
        candidates = [source.parent / path, HERE.parent / "frames" / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    return path


def source_seconds(timeline, n_frames):
    video = timeline.get("video_path")
    if video and Path(video).exists():
        capture = cv2.VideoCapture(str(video))
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if count and fps:
            return count / fps
    return n_frames / K.FPS


def event_dict(event):
    data = {
        "k": event.frame, "kind": event.kind, "component": event.component,
        "title": event.title, "detail": event.detail,
    }
    if event.evidence:
        data["evidence"] = {"kind": event.evidence.kind, "data": event.evidence.data}
    return data


def evidence_point(event, box, cropbox, shape, details):
    evidence = event.get("evidence") or {}
    kind, data = evidence.get("kind"), evidence.get("data") or {}
    x, y, w, h = box
    x0, y0, x1, y1 = cropbox

    if kind == "grid_cell" and details.get("grid_size"):
        grid = float(details["grid_size"])
        sx = (float(data["col"]) + 0.5) / grid * shape[1]
        sy = (float(data["row"]) + 0.5) / grid * shape[0]
    elif kind == "point" and "x" in data and "y" in data:
        sx, sy = float(data["x"]), float(data["y"])
    elif kind == "bbox" and all(key in data for key in ("x", "y", "w", "h")):
        sx = float(data["x"]) + float(data["w"]) / 2
        sy = float(data["y"]) + float(data["h"]) / 2
    else:
        return None
    return (x + (sx - x0) / max(x1 - x0 + 1, 1) * w,
            y + (sy - y0) / max(y1 - y0 + 1, 1) * h)


def component_colour(component):
    if component.timing == "final":
        return K.AMBER
    if component.kind in {"gate", "penalty"} and component.value < 0.999:
        return K.RED
    if component.value >= 0.999:
        return K.GREEN
    return K.BLUE


def render(timeline_path, output, seconds=None, stride=1):
    timeline = json.loads(timeline_path.read_text())
    task_name = timeline["task"]
    steps = timeline["steps"]
    adapter = get_adapter(task_name)
    fdir = frame_directory(timeline, timeline_path)
    frames = [cv2.imread(str(path)) for path in sorted(fdir.glob("frame_*.png"))]
    if not frames or any(frame is None for frame in frames):
        raise SystemExit("no readable frames in %s" % fdir)
    cropbox = K.trim_box(frames, mode="bg")
    shape = frames[0].shape
    duration = seconds or source_seconds(timeline, len(frames))
    order = K.frame_order(len(steps), duration, stride, total=duration + K.TAIL)
    writer = K.open_writer(output)
    thumbs = {}

    for output_index, index in enumerate(order):
        explanation = explain(task_name, steps, index)
        current = steps[index]
        frame_number = int(current["k"])
        frame = frames[min(frame_number - 1, len(frames) - 1)]
        previous = explain(task_name, steps, max(index - 1, 0))
        canvas = Image.new("RGB", (K.W, K.H), K.BG)
        draw = ImageDraw.Draw(canvas)
        K.header(draw, explanation.title, frame_number, int(steps[-1]["k"]))
        video_box = K.video_panel(canvas, draw, frame, cropbox)
        draw = ImageDraw.Draw(canvas)

        event_rows = [event_dict(event) for event in explanation.events]
        for event in event_rows:
            if 0 <= frame_number - event["k"] < 12:
                point = evidence_point(event, video_box, cropbox, shape,
                                       explanation.source_details)
                if point:
                    K.ring(draw, point[0], point[1], 28, K.RED, 4,
                           event["title"].upper(), 13, box=video_box)

        formula_text = K.fit(explanation.formula.display, 18, K.CW - 48, True)
        K.score_card(
            draw, K.VY, explanation.final_score,
            explanation.final_score - previous.final_score, frame_number,
            [(formula_text, K.MUT)], note="task-specific evaluator")

        y = K.VY + 168
        K.card(draw, K.CX, y, K.CW, K.CARD_H)
        K.caption(draw, K.CX + 24, y + 18, "SCORE COMPONENTS", K.CW - 48)
        visible = list(explanation.components)[:6]
        for row_index, component in enumerate(visible):
            note = component.note or (
                "%d%% weight" % round(component.weight * 100)
                if component.weight is not None else component.timing)
            K.score_row(
                draw, K.CX, K.CW, y + K.ROW_TOP + row_index * K.ROW_STEP,
                K.fit(component.label, 16, 210), component.value,
                component_colour(component), note=note, label_w=230,
                live=component.timing != "final")

        log_y = y + K.CARD_H + 16

        def mark(event, box):
            return evidence_point(event, box, cropbox, shape,
                                  explanation.source_details)

        K.deduction_log(canvas, ImageDraw.Draw(canvas), log_y, event_rows,
                        frame_number, frames, cropbox, thumbs, mark=mark,
                        empty="none yet — no scoring event has fired")
        K.write(writer, canvas)
        if output_index % 60 == 0:
            print("%d/%d" % (output_index, len(order)), flush=True)

    writer.release()
    print("wrote %s (%d frames, %.1fs)" %
          (output, len(order), len(order) / K.FPS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    render(args.timeline, args.out, args.seconds, args.stride)
