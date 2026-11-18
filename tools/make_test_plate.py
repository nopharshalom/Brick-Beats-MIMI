"""Draw a synthetic, perspective-distorted photo of the plate (markers + bricks) for testing the
plate reader and --dry-run without the robot or real bricks.

    python tools/make_test_plate.py "E1 D1 C2 | C1 D1 E1 F1 | G1 E1 C2 | G4" test_plate.jpg --baton orange

Measures are separated by '|' (tutorial | round 1 | round 2 | round 3), wizard note format (R = rest).
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimi import config as C  # noqa: E402
from mimi.music import parse_wizard_notes  # noqa: E402
from mimi.plate_reader import DEFAULT_COLORS, stud_to_px  # noqa: E402


def hsv_to_bgr(hsv):
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0])


def render(measures_text, baton=None):
    side = (C.PLATE_STUDS + 2 * C.CANVAS_MARGIN_STUDS) * C.PX_PER_STUD
    img = np.full((side, side, 3), 200, np.uint8)  # table
    p = C.PX_PER_STUD
    x0, y0 = (int(v) for v in stud_to_px(0, 0))
    cv2.rectangle(img, (x0, y0), (x0 + C.PLATE_STUDS * p, y0 + C.PLATE_STUDS * p), (245, 245, 245), -1)
    for i in range(C.PLATE_STUDS):  # studs
        for j in range(C.PLATE_STUDS):
            cv2.circle(img, (x0 + i * p + p // 2, y0 + j * p + p // 2), p // 3, (232, 232, 232), 1)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
    for marker_id, (mx, my) in C.MARKER_CENTERS_STUDS.items():
        size = int(3.5 * p)
        marker = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, marker_id, size), cv2.COLOR_GRAY2BGR)
        cx, cy = (int(v) for v in stud_to_px(mx, my))
        img[cy - size // 2 - 8:cy + size - size // 2 + 8, cx - size // 2 - 8:cx + size - size // 2 + 8] = 255
        img[cy - size // 2:cy - size // 2 + size, cx - size // 2:cx - size // 2 + size] = marker
    for m, text in enumerate(measures_text.split("|")):
        for n in parse_wizard_notes(text.strip(), m) or []:
            color = hsv_to_bgr(DEFAULT_COLORS[C.LANE_COLORS[n.lane]])
            bx, by = (int(v) for v in stud_to_px(C.GRID_X0 + n.start, C.GRID_Y0 + (len(C.PITCHES) - 1 - n.lane) * C.STUDS_PER_LANE))
            cv2.rectangle(img, (bx + 1, by + 1), (bx + n.beats * p - 2, by + C.STUDS_PER_LANE * p - 2), color, -1)
    if baton:
        x, y, w, h = C.PLAY_SLOT_STUDS
        bx, by = (int(v) for v in stud_to_px(x, y))
        cv2.rectangle(img, (bx, by), (bx + int(w * p), by + int(h * p)), hsv_to_bgr(DEFAULT_COLORS[baton]), -1)
    # seen from MIMI's side: upside down and in perspective
    img = cv2.rotate(img, cv2.ROTATE_180)
    src = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    w, h = 1280, 960
    dst = np.float32([[300, 120], [w - 300, 120], [w - 60, h - 40], [60, h - 40]])
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(180, 180, 180))
    return cv2.GaussianBlur(out, (3, 3), 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notes")
    parser.add_argument("out")
    parser.add_argument("--baton", nargs="?", const="orange", choices=list(DEFAULT_COLORS),
                        help="put a brick of this color (default orange) in the PLAY slot")
    args = parser.parse_args()
    cv2.imwrite(args.out, render(args.notes, args.baton))
    print("saved", args.out)


if __name__ == "__main__":
    main()
