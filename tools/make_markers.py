"""Print-ready ArUco markers 0-3 (DICT_4X4_50) for the plate corners.

    python tools/make_markers.py      # -> markers/markers_letter.pdf (print this) + PNGs

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
    save_pdf(dictionary)


def save_pdf(dictionary):
    """markers/markers_letter.pdf: all 4 markers at exact size on a US Letter page, with a 50 mm check
    line. Print it from a PDF viewer at 100% / Actual size (image viewers usually rescale PNGs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    page_w, page_h = 215.9, 279.4  # mm
    fig = plt.figure(figsize=(page_w / 25.4, page_h / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)
    ax.axis("off")
    for marker_id in range(4):
        x = 25 + (marker_id % 2) * 90
        y = 30 + (marker_id // 2) * 70
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, 200)
        ax.imshow(img, cmap="gray", extent=(x, x + SIDE_MM, y + SIDE_MM, y), interpolation="nearest")
        ax.add_patch(plt.Rectangle((x - 8, y - 8), SIDE_MM + 16, SIDE_MM + 16, fill=False,
                                   edgecolor="#AAAAAA", linewidth=0.5, linestyle="--"))  # cut line (keeps white border)
        ax.text(x + SIDE_MM / 2, y + SIDE_MM + 5, f"id {marker_id}", ha="center", fontsize=11)
    ax.plot([25, 75], [180, 180], color="black", linewidth=1)
    for x in (25, 75):
        ax.plot([x, x], [177, 183], color="black", linewidth=1)
    ax.text(25, 190, "This line must measure 50 mm, and each black square 30 mm.\n"
                     "If not, reprint at 100% / Actual size (no 'fit to page').\n"
                     "Cut along the dashed lines; keep the white border.", fontsize=10, va="top")
    os.makedirs("markers", exist_ok=True)
    fig.savefig("markers/markers_letter.pdf")
    plt.close(fig)
    print("saved markers/markers_letter.pdf (print this one)")


if __name__ == "__main__":
    main()
