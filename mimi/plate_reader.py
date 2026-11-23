"""Plate reader (perception): camera frame -> notes + confidence (proposal 4.2, "Vision").

    1. find the 4 ArUco markers and warp the frame to a top-down image in stud units
    2. classify every stud cell of the note grid in HSV (empty white / one of 8 calibrated colors / unknown)
    3. color runs inside a pitch lane become notes (1 stud = 1 beat)
    4. confidence = share of cells in the measures we care about that were read cleanly
    5. low confidence -> the VLM gets the rectified, labelled plate for a second opinion

Also: is the baton (any brick) in the PLAY slot, and is a hand in the way (markers hidden or the image moving).
"""

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C
from .models import VLM_MODEL, ask, images_content, parse_json
from .music import Note

CALIBRATION_FILE = "calibration.json"

# OpenCV HSV (H 0-179). Overwritten by calibrate.py --colors under the event lighting.
DEFAULT_COLORS = {
    "red": (0, 200, 170), "orange": (11, 220, 220), "yellow": (26, 210, 230), "light green": (40, 190, 200),
    "dark green": (65, 200, 120), "light blue": (100, 180, 210), "dark blue": (112, 200, 120), "black": (0, 0, 30),
}
EMPTY, DARK, UNKNOWN = "empty", "dark", "unknown"  # DARK = very dark cell when black is not a pitch


@dataclass
class PlateRead:
    notes: list = field(default_factory=list)
    confidence: float = 0.0
    markers_ok: bool = False
    moving: bool = False
    baton: bool = False
    issues: list = field(default_factory=list)  # human-readable reasons for lost confidence
    warped: np.ndarray | None = None
    source: str = "hsv"                           # "hsv", "vlm" or "wizard"

    @property
    def hand_in_way(self):
        return not self.markers_ok or self.moving


def load_colors():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        return {name: tuple(v) for name, v in data["colors"].items()}
    return dict(DEFAULT_COLORS)


def hsv_center(values, axis):
    """Typical HSV of a set of OpenCV HSV pixels: median saturation and value, and the CIRCULAR mean hue.
    Hue wraps around (179 and 0 are both red), so a plain median of red pixels can land near 90 (green)."""
    values = np.asarray(values, dtype=float)
    angle = values[..., 0] * (np.pi / 90.0)  # OpenCV hue 0-179 -> 0-2pi
    hue = (np.degrees(np.arctan2(np.sin(angle).mean(axis=axis), np.cos(angle).mean(axis=axis))) / 2.0) % 180.0
    return np.stack([hue, np.median(values[..., 1], axis=axis), np.median(values[..., 2], axis=axis)], axis=-1)


def stud_to_px(x, y):
    m, p = C.CANVAS_MARGIN_STUDS, C.PX_PER_STUD
    return (x + m) * p, (y + m) * p


class PlateReader:
    def __init__(self):
        self.colors = load_colors()
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        side = (C.PLATE_STUDS + 2 * C.CANVAS_MARGIN_STUDS) * C.PX_PER_STUD
        self.canvas_size = (side, side)
        self.last_homography = None

    # ------------------------------------------------------------ geometry

    def homography(self, frame):
        """(H, all four markers found). Reuses the last good H if a marker is hidden."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = {}
        if ids is not None:
            for quad, marker_id in zip(corners, ids.flatten()):
                if int(marker_id) in C.MARKER_CENTERS_STUDS:
                    found[int(marker_id)] = quad.reshape(4, 2).mean(axis=0)
        if len(found) == 4:
            src = np.float32([found[i] for i in range(4)])
            dst = np.float32([stud_to_px(*C.MARKER_CENTERS_STUDS[i]) for i in range(4)])
            self.last_homography = cv2.getPerspectiveTransform(src, dst)
            return self.last_homography, True
        return self.last_homography, False

    def warp(self, frame):
        H, ok = self.homography(frame)
        if H is None:
            return None, False
        return cv2.warpPerspective(frame, H, self.canvas_size), ok

    def column_runs(self, hsv, col):
        """Bricks seen in one beat column of the flattened image, as [(lane, top-face HSV, run length in studs)].
        The camera sees each brick at an angle, so its color spills over the cells beyond it (away from the
        camera). The run's end nearest the camera is the brick's cell; its far end is the brick's top face."""
        p, lanes, spl = C.PX_PER_STUD, len(C.PITCHES), C.STUDS_PER_LANE
        grid_top, grid_bottom = C.GRID_Y0, C.GRID_Y0 + lanes * spl
        from_top = C.CAMERA_SIDE == "top"
        y_lo = grid_top - 0.5 if from_top else grid_top - C.BRICK_SPILL_STUDS
        y_hi = grid_bottom + C.BRICK_SPILL_STUDS if from_top else grid_bottom + 0.5
        x = C.GRID_X0 + col * C.STUDS_PER_BEAT
        (x0, y0), (x1, y1) = stud_to_px(x + 0.3, y_lo), stud_to_px(x + 0.7 * C.STUDS_PER_BEAT, y_hi)
        rows = hsv_center(hsv[int(y0):int(y1), int(x0):int(x1)], axis=1)  # one HSV per pixel row
        filled = (rows[:, 1] >= C.CELL_MIN_COLOR_SAT) | (rows[:, 2] < C.BLACK_MAX_VALUE)
        # bridge tiny gaps (stud shadows, seams), then cut into runs
        gap = max(1, int(0.15 * p))
        for i in range(1, len(filled) - gap):
            if not filled[i] and filled[i - 1] and filled[i:i + gap + 1].any():
                filled[i] = True
        runs, start = [], None
        for i, f in enumerate(np.append(filled, False)):
            if f and start is None:
                start = i
            elif not f and start is not None:
                runs.append((start, i))
                start = None
        found = []
        for start, end in runs:
            length = (end - start) / p
            if length < 0.5:
                continue
            near = y_lo + (start if from_top else end - spl * p) / p  # top edge of the brick's cell
            k = int(round(near - grid_top - 0.2))                      # cell rows from the top of the grid
            lane = lanes - 1 - k // spl if 0 <= k < lanes * spl else None
            if lane is None:
                continue
            # the top face, staying clear of the brick's far edge and the shadow beyond it
            edge = max(1, int(0.2 * p))
            if from_top:
                face = rows[max(start, end - p - edge):max(start + 1, end - edge)]
            else:
                face = rows[min(start + edge, end - 1):min(end, start + p + edge)]
            found.append((lane, hsv_center(face, axis=0), length))
        return found

    # ------------------------------------------------------------ cells

    def _patch(self, hsv, x, y, w=1, h=1):
        """Median HSV of the middle of a stud rectangle (avoids seams and stud shadows)."""
        p = C.PX_PER_STUD
        x0, y0 = stud_to_px(x, y)
        x0, y0 = int(x0 + 0.25 * p), int(y0 + 0.25 * p)
        x1, y1 = int(x0 + (w - 0.5) * p), int(y0 + (h - 0.5) * p)
        region = hsv[y0:y1, x0:x1].reshape(-1, 3)
        return hsv_center(region, axis=0) if len(region) else np.zeros(3)

    def classify(self, hsv_value):
        """(label, distance) for one cell. Distance is 0 for clear empty/black cells.
        Very dark cells have no reliable hue: they are the "black" pitch if black is a lane color,
        otherwise DARK (not a note)."""
        h, s, v = (float(c) for c in hsv_value)
        if v < C.BLACK_MAX_VALUE:
            return ("black" if "black" in C.LANE_COLORS else DARK), 0.0
        if s < C.CELL_MIN_COLOR_SAT:
            return EMPTY, 0.0
        best, best_d = UNKNOWN, 1e9
        for name, (ch, cs, cv) in self.colors.items():
            if name == "black":  # matched by brightness above, not by hue
                continue
            dh = min(abs(h - ch), 180 - abs(h - ch))
            d = (4.0 * dh ** 2 + 0.15 * (s - cs) ** 2 + 0.05 * (v - cv) ** 2) ** 0.5
            if d < best_d:
                best, best_d = name, d
        if best_d > C.CELL_MAX_COLOR_DIST:
            return UNKNOWN, best_d
        return best, best_d

    def cell_grid(self, warped, frame=None):
        """labels[lane][beat_column] for the whole note grid, from the color runs in each beat column."""
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lanes = len(C.PITCHES)
        columns = C.MEASURES * C.BEATS_PER_MEASURE
        labels = [[EMPTY] * columns for _ in range(lanes)]
        bad = [[False] * columns for _ in range(lanes)]
        longest = C.STUDS_PER_LANE + C.BRICK_SPILL_STUDS
        for col in range(columns):
            for lane, face, length in self.column_runs(hsv, col):
                label = self.classify(face)[0]
                if label == EMPTY:  # a colored side with a washed-out top: still a brick, color unknown
                    label = UNKNOWN
                labels[lane][col] = label
                # a run much longer than one brick plus its spill = two bricks touching in this column
                bad[lane][col] = label in (UNKNOWN, DARK) or length > longest
        return labels, bad

    def staircase_faces(self, warped):
        """Top-face HSV of the calibration staircase (lane i at beat i+1), for calibrate.py colors."""
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        faces = {}
        for lane in range(len(C.PITCHES)):
            runs = [face for run_lane, face, _ in self.column_runs(hsv, lane) if run_lane == lane]
            faces[lane] = runs[0] if runs else None
        return faces

    # ------------------------------------------------------------ notes

    def notes_from_cells(self, labels, bad, measures):
        """Color runs inside each lane -> notes. Returns (notes, clean_fraction, issues)."""
        notes, issues = [], []
        checked = set()
        for m in measures:
            lo = m * C.BEATS_PER_MEASURE
            for lane, row in enumerate(labels):
                col = lo
                while col < lo + C.BEATS_PER_MEASURE:
                    checked.add((lane, col))
                    if row[col] == EMPTY:
                        col += 1
                        continue
                    end = col
                    while end + 1 < lo + C.BEATS_PER_MEASURE and row[end + 1] == row[col]:
                        end += 1
                        checked.add((lane, end))
                    length = end - col + 1
                    if row[col] != C.LANE_COLORS[lane] and row[col] not in (UNKNOWN, DARK):
                        issues.append(f"{row[col]} brick in the {C.PITCHES[lane]} lane at beat {col - lo + 1}")
                        for c in range(col, end + 1):
                            bad[lane][c] = True
                    # bricks are 1, 2 or 4 studs long; a run of 3 is two same-colored bricks side by side
                    parts = {1: [1], 2: [2], 3: [3], 4: [4]}[length]  # one brick per run (1x1 ... 1x4)
                    start = col
                    for beats in parts:
                        notes.append(Note(start, lane, beats))
                        start += beats
                    col = end + 1
        # a melody is one line: two notes sounding on the same beat means a misread or a stray brick
        for n in notes:
            for other in notes:
                if other is not n and other.start <= n.start < other.start + other.beats and other.lane != n.lane:
                    issues.append(f"two notes on beat {n.start % C.BEATS_PER_MEASURE + 1}")
                    bad[n.lane][n.start] = True
        occupied = [(l, c) for (l, c) in checked if labels[l][c] != EMPTY]
        n_bad = sum(bad[l][c] for (l, c) in checked)
        clean = 1.0 - n_bad / max(C.BEATS_PER_MEASURE, len(occupied))
        return sorted(set(notes)), max(0.0, clean), sorted(set(issues))

    def baton_in_slot(self, warped):
        """Is the baton in the PLAY slot?
        Light slot (white outline): any brick counts, when most of the slot is colored or dark.
        Black pad (PLAY_SLOT_BLACK_PAD): the empty pad is dark, so the baton is any COLORED brick on it."""
        x, y, w, h = C.PLAY_SLOT_STUDS
        x0, y0 = (int(v) for v in stud_to_px(x, y))
        x1, y1 = (int(v) for v in stud_to_px(x + w, y + h))
        region = cv2.cvtColor(warped[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        if region.size == 0:
            return False
        colored = (region[..., 1] >= C.CELL_MIN_COLOR_SAT) & (region[..., 2] >= C.BLACK_MAX_VALUE)
        if C.PLAY_SLOT_BLACK_PAD:
            return float(colored.mean()) >= C.BATON_ON_PAD_FRACTION
        filled = colored | (region[..., 2] < C.BLACK_MAX_VALUE)
        return float(filled.mean()) >= C.BATON_FILL_FRACTION

    # ------------------------------------------------------------ public

    def read(self, frame, measures, previous_warped=None):
        """One frame -> PlateRead for the given measures (0 = tutorial ... 3 = round 3)."""
        warped, markers_ok = self.warp(frame)
        if warped is None:
            return PlateRead(issues=["plate markers not found"])
        labels, bad = self.cell_grid(warped, frame)
        notes, confidence, issues = self.notes_from_cells(labels, bad, measures)
        moving = False
        if previous_warped is not None:
            diff = cv2.absdiff(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), cv2.cvtColor(previous_warped, cv2.COLOR_BGR2GRAY))
            moving = float((diff > 40).mean()) > C.MOTION_PIXEL_FRACTION
        if not markers_ok:
            issues.append("a marker is covered")
        return PlateRead(notes, confidence, markers_ok, moving, self.baton_in_slot(warped), issues, warped)

    def baton_only(self, frame):
        warped, _ = self.warp(frame)
        return warped is not None and self.baton_in_slot(warped)


# ---------------------------------------------------------------- drawing and VLM second opinion

def annotate(warped, notes=(), active_measure=None):
    """Grid lines, pitch labels and beat numbers on the rectified plate (for the VLM and the log)."""
    img = warped.copy()
    p, lanes = C.PX_PER_STUD, len(C.PITCHES)
    cols = C.MEASURES * C.BEATS_PER_MEASURE
    x0, y0 = (int(v) for v in stud_to_px(C.GRID_X0, C.GRID_Y0))
    for lane in range(lanes + 1):
        y = y0 + lane * C.STUDS_PER_LANE * p
        cv2.line(img, (x0, y), (x0 + cols * p, y), (90, 90, 90), 1)
    for col in range(cols + 1):
        x = x0 + col * p
        thick = 3 if col % C.BEATS_PER_MEASURE == 0 else 1
        cv2.line(img, (x, y0), (x, y0 + lanes * C.STUDS_PER_LANE * p), (90, 90, 90), thick)
        if col < cols:
            cv2.putText(img, str(col % C.BEATS_PER_MEASURE + 1), (x + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for lane in range(lanes):
        y = y0 + (lanes - 1 - lane) * C.STUDS_PER_LANE * p + C.STUDS_PER_LANE * p // 2 + 5
        cv2.putText(img, C.PITCHES[lane], (x0 - 34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    if active_measure is not None:
        ax = x0 + active_measure * C.BEATS_PER_MEASURE * p
        cv2.rectangle(img, (ax, y0), (ax + C.BEATS_PER_MEASURE * p, y0 + lanes * C.STUDS_PER_LANE * p), (255, 0, 255), 2)
    for n in notes:
        x = x0 + n.start * p
        y = y0 + (lanes - 1 - n.lane) * C.STUDS_PER_LANE * p
        cv2.rectangle(img, (x + 2, y + 2), (x + n.beats * p - 2, y + C.STUDS_PER_LANE * p - 2), (0, 0, 0), 2)
    return img


def vlm_second_opinion(warped, measure):
    """Ask the VLM to read one measure of the rectified plate. Returns a list of Notes or None."""
    pitch_list = ", ".join(f"{p} ({c})" for p, c in zip(C.PITCHES, C.LANE_COLORS))
    prompt = (
        "This is a top-down photo of a LEGO music board. The grid has 8 rows (pitch lanes), labelled on "
        f"the left from bottom to top: {pitch_list}. Columns are beats, numbered 1-4 in each measure; "
        "thick lines separate measures. A brick's length is its duration: 1 column = 1 beat, "
        "2 columns = 2 beats, 3 columns = 3 beats, 4 columns = 4 beats. Ignore hands, shadows and anything outside the grid.\n"
        f"Read ONLY the measure outlined in magenta (measure {measure + 1}).\n"
        'Reply with only JSON: {"notes": [{"pitch": "E4", "beat": 1, "beats": 1}, ...]} '
        "listing every brick in that measure from left to right; empty list if there are none.")
    try:
        reply = ask(VLM_MODEL, [{"role": "user", "content": images_content([annotate(warped, active_measure=measure)], prompt)}])
    except Exception as error:
        print(f"  (VLM unavailable: {error})")
        return None
    print("  vlm:", reply.strip().replace("\n", " ")[:300])
    data = parse_json(reply)
    if not isinstance(data.get("notes"), list):
        return None
    notes = []
    for item in data["notes"]:
        try:
            pitch = str(item["pitch"]).upper()
            pitch = pitch if pitch in C.PITCHES else {"C": "C4", "D": "D4", "E": "E4", "F": "F4", "G": "G4", "A": "A4", "B": "B4"}.get(pitch)
            beat, beats = int(item["beat"]), int(item["beats"])
        except (KeyError, TypeError, ValueError):
            return None
        if pitch is None or not 1 <= beat <= C.BEATS_PER_MEASURE or beats not in C.DURATION_WORDS:
            return None
        notes.append(Note(measure * C.BEATS_PER_MEASURE + beat - 1, C.PITCHES.index(pitch), beats))
    return sorted(notes)
