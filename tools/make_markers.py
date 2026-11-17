"""Print-ready ArUco markers 0-3 (DICT_4X4_50) for the plate corners.

    python tools/make_markers.py      # -> markers/marker_0.png ... marker_3.png and markers/all.png

Print at 100%. Glue each one so its CENTER sits at MARKER_CENTERS_STUDS in mimi/config.py
(0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left, seen from the students' side; MIMI at the top).
A stud is 8 mm, so the default centers are 2.5 studs = 20 mm outside each plate corner.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimi import config as C  # noqa: E402

SIDE_MM = 30   # printed marker size; keep it smaller than 2 x the corner offset
DPI = 300


def main():
    os.makedirs("markers", exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
    px = int(SIDE_MM / 25.4 * DPI)
    tiles = []
    for marker_id in range(4):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, px)
        img = cv2.copyMakeBorder(img, 40, 80, 40, 40, cv2.BORDER_CONSTANT, value=255)  # white quiet zone + label
        cv2.putText(img, f"id {marker_id}", (40, img.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
        cv2.imwrite(f"markers/marker_{marker_id}.png", img)
        tiles.append(img)
    cv2.imwrite("markers/all.png", np.hstack(tiles))
    print(f"saved markers/ ({SIDE_MM} mm at {DPI} dpi)")


if __name__ == "__main__":
    main()
