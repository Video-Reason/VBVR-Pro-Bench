"""Drawing kit for the live-scoring HUD.

Palette, layout constants and drawing primitives only, so every task renders in
one visual system. Also provides the H.264 writer used for the output MP4.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 24
TAIL = 0.7                                  # seconds held on the final frame
SPEED = 2.00                                # playback speed of the clip itself

BG = (255, 255, 255)
CARD = (255, 255, 255)
EDGE = (228, 228, 228)                      # HAIR #E4E4E4
TINT = (237, 242, 246)                      # #EDF2F6
TRACK = (229, 234, 238)
INK = (26, 26, 26)                          # #1A1A1A
MUT = (110, 123, 133)                       # #6E7B85
BLUE = (44, 95, 124)                        # #2C5F7C
RED = (192, 57, 43)                         # #C0392B
RED_TINT = (251, 241, 240)                  # #FBF1F0
GREEN = (46, 125, 79)
AMBER = (180, 83, 9)

# ---- layout shared by every case, in px -----------------------------------
# The right column is the same width as the video and the pair is centred: a
# panel wider than the footage it explains reads as the tail wagging the dog.
VS = 820                                    # video panel (square)
GAP = 40
VX = (W - (2 * VS + GAP)) // 2              # 120 px margins either side
VY = 82
CX = VX + VS + GAP                          # right column
CW = VS
HALF = (CW - 16) // 2
BY, BH, TH = 924, 150, 96                   # bottom strip, thumbnail size
MW = 250                                    # meter length
PAD = 10                                    # video panel inner padding

# The system font directory is not guaranteed to exist — it has already moved
# once under this project — so resolve DejaVu from wherever it actually is.
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/dejavu",
             os.path.expanduser("~/.fonts"),
             os.environ.get("VBVR_FONT_DIR", "")]
_F = {}


def _font_path(name):
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"{name} not found in any of {FONT_DIRS}")


def font(size, bold=False):
    key = (round(size), bold)
    if key not in _F:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        _F[key] = ImageFont.truetype(_font_path(name), round(size))
    return _F[key]


def text(d, xy, s, size=20, bold=False, fill=INK, anchor="la"):
    d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def fit(s, size, maxw, bold=False):
    """Trim a string with an ellipsis so it cannot run into the next column."""
    f = font(size, bold)
    if f.getlength(s) <= maxw:
        return s
    while s and f.getlength(s + "…") > maxw:
        s = s[:-1]
    return s.rstrip() + "…"


def card(d, x, y, w, h, r=8, fill=CARD, edge=EDGE):
    d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill, outline=edge)


def caption(d, x, y, s, w=None):
    """Small blue section label, optionally with a rule under it."""
    text(d, (x, y), s, 14, True, BLUE)
    if w:
        d.line([(x, y + 22), (x + w, y + 22)], fill=EDGE)


def meter(d, x, y, w, value, colour, h=9):
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=TRACK)
    fw = int(w * max(0.0, min(1.0, value)))
    if fw > h:
        d.rounded_rectangle([x, y, x + fw, y + h], radius=h // 2, fill=colour)


def score_row(d, x, w, y, label, value, colour, note=None, digits=3,
              label_w=132, sub=None, live=False, text_value=None):
    """One metered sub-score line, laid out inside a card of width `w`.

    The meter length is derived from the card, never hard-coded: the cards are
    now half as wide as they were, and a fixed-length bar ran under the value
    column.  A long note pushes the meter right; if that leaves no room the
    meter is dropped rather than drawn inverted.
    """
    lx = x + 24
    if live:                       # a dot marks a metric that is still moving
        d.ellipse([lx, y + 7, lx + 7, y + 14], fill=BLUE)
        lx += 14
    text(d, (lx, y), label, 17 if digits == 3 else 16, False, INK)
    if sub:
        text(d, (lx, y + 21), sub, 14, False, MUT)
    vx = x + w - 24                                   # right edge of the value
    mx = x + 24 + label_w
    val = text_value if text_value is not None else f"{value:.{digits}f}"
    m_end = vx - int(font(17, True).getlength(val)) - 16
    if note:
        note = fit(note, 13, m_end - mx)        # never run under the value
        text(d, (mx, y + 2), note, 13, False, RED if value < 0.999 else MUT)
        mx += int(font(13).getlength(note)) + 16
    if m_end - mx >= 60 and text_value is None:
        meter(d, mx, y + 7, m_end - mx, value, colour)
    text(d, (vx, y), val, 17, True, colour, anchor="ra")


def pill(d, x, y, label, colour, size=17):
    """Status badge — a filled rounded chip, for per-frame verdicts."""
    w = int(font(size, True).getlength(label)) + 28
    d.rounded_rectangle([x, y, x + w, y + size + 16], radius=(size + 16) // 2,
                        fill=colour)
    text(d, (x + 14, y + 8), label, size, True, (255, 255, 255))
    return w


def ring(d, cx, cy, rad, colour, width=3, label=None, size=15, box=None):
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=colour,
              width=width)
    if label:                       # flip the label below when there is no
        above = box is None or cy - rad - 24 > box[1]      # room above
        text(d, (cx, cy - rad - 6 if above else cy + rad + 6), label, size,
             True, colour, anchor="md" if above else "ma")


def trim_box(frames, mode="dark", sample=24):
    """Box of the uniform margin a renderer leaves around the scene.

    The box is the *union* over frames sampled across the whole clip, and is
    then fixed for the run: taking it from the first frame alone clips anything
    that moves into fresh territory later (O-32's ball ends up outside frame
    0's content box), and recomputing it per frame makes the panel jitter.

    A square source is left alone entirely — its framing is part of the shot,
    and trimming it changes what the viewer is being shown.  Only letterboxed
    or otherwise padded sources get trimmed: ``mode="dark"`` keeps pixels
    brighter than a black surround, ``mode="bg"`` keeps pixels that differ from
    the modal background colour.
    """
    if not isinstance(frames, list):
        frames = [frames]
    h, w = frames[0].shape[:2]
    if abs(w - h) <= 2:
        return 0, 0, w - 1, h - 1      # square source: keep the original frame
    step = max(1, len(frames) // sample)
    box = None
    for f in frames[::step] + [frames[-1]]:
        if mode == "dark":
            keep = f.max(axis=2) > 40
        else:
            bg = np.median(f.reshape(-1, 3), axis=0)
            keep = np.abs(f.astype(int) - bg).sum(axis=2) > 25
        ys, xs = np.nonzero(keep)
        if not len(xs):
            continue
        b = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3]))
    f = frames[0]
    if box is None:
        return 0, 0, f.shape[1] - 1, f.shape[0] - 1
    pad = int(0.012 * max(f.shape[:2]))
    return (max(box[0] - pad, 0), max(box[1] - pad, 0),
            min(box[2] + pad, f.shape[1] - 1),
            min(box[3] + pad, f.shape[0] - 1))


def crop(frame, box):
    x0, y0, x1, y1 = box
    return frame[y0:y1 + 1, x0:x1 + 1]


def paste(canvas, frame, box, cropbox):
    """Draw a BGR frame, cropped and scaled, into a panel; keeps its aspect."""
    x, y, w, h = box
    im = Image.fromarray(cv2.cvtColor(crop(frame, cropbox), cv2.COLOR_BGR2RGB))
    scale = min(w / im.width, h / im.height)
    tw, th = max(1, int(im.width * scale)), max(1, int(im.height * scale))
    ox, oy = x + (w - tw) // 2, y + (h - th) // 2
    canvas.paste(im.resize((tw, th), Image.LANCZOS), (ox, oy))
    return ox, oy, tw, th                       # where it actually landed


def video_panel(canvas, d, frame, cropbox):
    """Frame in a card sized to the frame, not to a fixed square.

    A wide scene in a square card leaves most of the card empty, which reads as
    a layout mistake rather than as the shape of the content.
    """
    x0, y0, x1, y1 = cropbox
    iw, ih = x1 - x0 + 1, y1 - y0 + 1
    scale = min((VS - 2 * PAD) / iw, (VS - 2 * PAD) / ih)
    tw, th = max(1, int(iw * scale)), max(1, int(ih * scale))
    cx, cy = VX + (VS - tw) // 2 - PAD, VY + (VS - th) // 2 - PAD
    card(d, cx, cy, tw + 2 * PAD, th + 2 * PAD, fill=TINT)
    im = Image.fromarray(cv2.cvtColor(crop(frame, cropbox), cv2.COLOR_BGR2RGB))
    canvas.paste(im.resize((tw, th), Image.LANCZOS), (cx + PAD, cy + PAD))
    return cx + PAD, cy + PAD, tw, th


def header(d, title, k, total):
    """Task name on the left, position in the clip on the right — nothing else."""
    text(d, (VX, 22), title, 30, True, INK)
    text(d, (CX + CW, 34), f"frame {k} / {total}", 18, False, MUT, anchor="ra")
    d.line([(VX, 66), (CX + CW, 66)], fill=EDGE)


def score_card(d, y, score, delta, k, chain, note="what the evaluator returns"):
    """The one piece of real typographic weight on the page."""
    card(d, CX, y, CW, 152)
    caption(d, CX + 24, y + 18, "SCORE SO FAR")
    text(d, (CX + 24, y + 44), f"{score:.3f}", 62, True, INK)
    if abs(delta) > 5e-3:
        text(d, (CX + 232, y + 74),
             f"{'▲' if delta > 0 else '▼'} {abs(delta):.3f}", 22, True,
             GREEN if delta > 0 else RED)
    text(d, (CX + CW - 24, y + 22), note, 14, False, MUT, anchor="ra")
    text(d, (CX + CW - 24, y + 42), f"if the video ended at frame {k}", 14,
         False, MUT, anchor="ra")
    rx = CX + CW - 24
    for seg, col in reversed(chain):
        rx -= font(21, True).getlength(seg)
        text(d, (rx, y + 100), seg, 21, True, col)


CARD_H = 344                                # metric cards
ROW_TOP, ROW_STEP = 62, 46                  # up to six rows per card


# The event log is bottom-aligned with the video panel so the two columns read
# as one block. That fixes the available height, which in turn fixes the row
# count at three legible rows rather than four cramped ones.
LOG_BOTTOM = VY + VS
ROW_H, ROW_THUMB, ROW_LIMIT = 74, 62, 3


def deduction_log(canvas, d, y, events, k, frames, cropbox, thumbs,
                  mark=None, empty="none yet — the run is still clean",
                  limit=ROW_LIMIT):
    """Every deduction, each carrying the frame it was raised on.

    This used to be two panels: a list here and a strip of triggering frames
    across the bottom of the page, which repeated each deduction's title and
    detail beside its thumbnail.  The reader had to match one against the other
    to learn nothing new, so the thumbnail moved into the row and the strip is
    gone.  ``mark`` is an optional callback (event, box) -> (x, y) that rings
    the offending element inside a thumbnail.
    """
    h = LOG_BOTTOM - y
    card(d, CX, y, CW, h)
    caption(d, CX + 24, y + 18, "DEDUCTIONS", CW - 48)
    fired = [e for e in events if e["k"] <= k]
    if not fired:
        text(d, (CX + 24, y + 56), empty, 17, False, GREEN)
    for j, e in enumerate(fired[-limit:]):
        ry = y + 52 + j * ROW_H
        fresh = 0 <= k - e["k"] < 12
        d.rounded_rectangle([CX + 16, ry - 2, CX + CW - 16, ry + ROW_H - 8],
                            radius=6, fill=RED_TINT,
                            outline=RED if fresh else RED_TINT)
        key = events.index(e)
        if key not in thumbs:
            src = crop(frames[min(e["k"] - 1, len(frames) - 1)], cropbox)
            im = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
            scale = ROW_THUMB / max(im.width, im.height)
            thumbs[key] = im.resize((max(1, int(im.width * scale)),
                                     max(1, int(im.height * scale))),
                                    Image.LANCZOS)
        th = thumbs[key]
        tx = CX + 28
        ty = ry + (ROW_H - 10 - th.height) // 2
        canvas.paste(th, (tx, ty))
        d = ImageDraw.Draw(canvas)
        if mark is not None:
            spot = mark(e, (tx, ty, th.width, th.height))
            if spot:
                ring(d, spot[0], spot[1], 8, RED, 2)
        d.rectangle([tx, ty, tx + th.width - 1, ty + th.height - 1],
                    outline=RED, width=2)
        lx = tx + ROW_THUMB + 18
        # Reserve a bounded right-hand column for diagnostics. Evaluator
        # details can be arbitrarily long; the HUD must never let them collide
        # with the event title or run beyond the card.
        detail_text = fit(str(e["detail"]), 15, min(350, CW // 2), True)
        dw = int(font(15).getlength(detail_text))
        text(d, (lx, ry + 10), fit(e["title"], 17,
                                   max(120, CW - (lx - CX) - dw - 56),
                                   True), 17, True, RED)
        text(d, (lx, ry + 36), f"frame {e['k']}", 14, False, MUT)
        text(d, (CX + CW - 32, ry + 24), detail_text, 15, False, MUT,
             anchor="ra")
    return d


class H264Writer:
    """Pipe raw frames to ffmpeg's libx264.

    cv2.VideoWriter's "mp4v" is MPEG-4 Part 2, which browsers refuse to play in
    a <video> tag and some slide software silently drops.  H.264 High profile in
    yuv420p with the moov atom moved to the front plays everywhere: browsers,
    PowerPoint, Keynote, QuickTime.
    """

    @staticmethod
    def _ffmpeg():
        """Resolve an ffmpeg with libx264: VBVR_FFMPEG, then the binary that
        imageio-ffmpeg ships, then whatever is on PATH."""
        import shutil
        override = os.environ.get("VBVR_FFMPEG", "")
        if override and os.path.exists(override):
            return override
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                return exe
        except Exception:
            pass
        return shutil.which("ffmpeg")

    def __init__(self, path, w=W, h=H, fps=FPS, crf=18):
        import subprocess
        exe = self._ffmpeg()
        if not exe:
            raise SystemExit("no ffmpeg with libx264 found; "
                             "pip install imageio-ffmpeg")
        self.p = subprocess.Popen(
            [exe, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.0",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise SystemExit("ffmpeg failed while encoding")


def open_writer(path):
    return H264Writer(path)


MIN_PLAY = 0.55                             # slowest the clip may be played


def frame_order(n_steps, seconds, stride=1, tail=None, total=None,
                min_play=MIN_PLAY):
    """Playback that runs for as long as the source clip, plus a tail.

    The sampled frames are re-timed onto the output frame rate rather than
    played one-per-output-frame, so playback runs at a fixed multiple of the
    source speed. The only added time is the tail that holds the final score.
    """
    speed = SPEED
    if total:
        # `total` is the target output duration. The clip is slowed to fill it,
        # down to a floor past which playback stops reading as motion; whatever
        # the floor cannot absorb becomes a hold on the completed final panel.
        speed = max(min_play, seconds / max(0.1, total - TAIL))
        tail = total - seconds / speed
    n_out = max(1, round(seconds * FPS / speed)) if seconds else n_steps
    order = [min(n_steps - 1, i * n_steps // n_out) for i in range(n_out)]
    # `tail` overrides the default hold on the last frame: the clip still plays
    # at the same rate, it simply rests on the finished state a moment longer
    order += [n_steps - 1] * round((TAIL if tail is None else tail) * FPS)
    return order[::stride] if stride > 1 else order


def write(writer, canvas):
    writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
