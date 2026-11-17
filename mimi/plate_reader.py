"""Plate reader (perception): camera frame -> notes + confidence (proposal 4.2, "Vision").

    1. find the 4 ArUco markers and warp the frame to a top-down image in stud units
    2. classify every stud cell of the note grid in HSV (empty white / one of 8 calibrated colors / unknown)
    3. color runs inside a pitch lane become notes (1 stud = 1 beat)
    4. confidence = share of cells in the measures we care about that were read cleanly
    5. low confidence -> the VLM gets the rectified, labelled plate for a second opinion

Also: is the baton brick in the PLAY slot, and is a hand in the way (markers hidden or the image moving).
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
    "red": (0, 200, 170), "orange": (11, 220, 220), "yellow": (26, 210, 230), "lime": (40, 190, 200),
    "green": (65, 200, 140), "azure": (100, 180, 210), "purple": (135, 120, 140), "pink": (165, 110, 230),
}
EMPTY, BLACK, UNKNOWN = "empty", "black", "unknown"


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

    # ------------------------------------------------------------ cells

    def _patch(self, hsv, x, y, w=1, h=1):
        """Median HSV of the middle of a stud rectangle (avoids seams and stud shadows)."""
        p = C.PX_PER_STUD
        x0, y0 = stud_to_px(x, y)
        x0, y0 = int(x0 + 0.25 * p), int(y0 + 0.25 * p)
        x1, y1 = int(x0 + (w - 0.5) * p), int(y0 + (h - 0.5) * p)
        region = hsv[y0:y1, x0:x1].reshape(-1, 3)
        return np.median(region, axis=0) if len(region) else np.zeros(3)

    def classify(self, hsv_value):
        """(label, distance) for one cell. Distance is 0 for clear empty/black cells."""
        h, s, v = (float(c) for c in hsv_value)
        if v < 60:
            return BLACK, 0.0
        if s < C.CELL_MIN_COLOR_SAT:
            return EMPTY, 0.0
        best, best_d = UNKNOWN, 1e9
        for name, (ch, cs, cv) in self.colors.items():
            dh = min(abs(h - ch), 180 - abs(h - ch))
            d = (4.0 * dh ** 2 + 0.15 * (s - cs) ** 2 + 0.05 * (v - cv) ** 2) ** 0.5
            if d < best_d:
                best, best_d = name, d
        if best_d > C.CELL_MAX_COLOR_DIST:
            return UNKNOWN, best_d
        return best, best_d

    def cell_grid(self, warped):
        """labels[lane][beat_column] for the whole note grid."""
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lanes = len(C.PITCHES)
        columns = C.MEASURES * C.BEATS_PER_MEASURE
        labels = [[EMPTY] * columns for _ in range(lanes)]
        bad = [[False] * columns for _ in range(lanes)]
        for lane in range(lanes):
            top = C.GRID_Y0 + (lanes - 1 - lane) * C.STUDS_PER_LANE
            for col in range(columns):
                x = C.GRID_X0 + col * C.STUDS_PER_BEAT
                rows = [self.classify(self._patch(hsv, x, top + r, C.STUDS_PER_BEAT, 1))[0]
                        for r in range(C.STUDS_PER_LANE)]
                colored = [r for r in rows if r not in (EMPTY,)]
                if not colored:
                    continue
                label = colored[0]
                # a real brick fills the whole 2-stud lane with one color
                bad[lane][col] = len(colored) != len(rows) or len(set(rows)) > 1 or label in (UNKNOWN, BLACK)
                labels[lane][col] = label
        return labels, bad

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
                    if row[col] != C.LANE_COLORS[lane] and row[col] not in (UNKNOWN, BLACK):
                        issues.append(f"{row[col]} brick in the {C.PITCHES[lane]} lane at beat {col - lo + 1}")
                        for c in range(col, end + 1):
                            bad[lane][c] = True
                    # bricks are 1, 2 or 4 studs long; a run of 3 is two same-colored bricks side by side
                    parts = {1: [1], 2: [2], 3: [2, 1], 4: [4]}[length]
                    if length == 3:
                        issues.append(f"3-beat run in {C.PITCHES[lane]} lane: guessing half + quarter")
                        bad[lane][col] = True
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
        x, y, w, h = C.PLAY_SLOT_STUDS
        x0, y0 = (int(v) for v in stud_to_px(x, y))
        x1, y1 = (int(v) for v in stud_to_px(x + w, y + h))
        region = cv2.cvtColor(warped[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        if region.size == 0:
            return False
        return float((region[..., 2] < 60).mean()) >= C.BATON_DARK_FRACTION

    # ------------------------------------------------------------ public

    def read(self, frame, measures, previous_warped=None):
        """One frame -> PlateRead for the given measures (0 = tutorial ... 3 = round 3)."""
        warped, markers_ok = self.warp(frame)
        if warped is None:
            return PlateRead(issues=["plate markers not found"])
        labels, bad = self.cell_grid(warped)
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
        y = y0 + (lanes - 1 - lane) * C.STUDS_PER_LANE * p + p + 5
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
        "2 columns = 2 beats, 4 columns = 4 beats. Ignore hands, shadows and anything outside the grid.\n"
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
        if pitch is None or not 1 <= beat <= C.BEATS_PER_MEASURE or beats not in (1, 2, 4):
            return None
        notes.append(Note(measure * C.BEATS_PER_MEASURE + beat - 1, C.PITCHES.index(pitch), beats))
    return sorted(notes)
