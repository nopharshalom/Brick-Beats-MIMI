"""Draw the report figures from mimi/config.py, so they always match the code.

    python tools/make_figures.py
        -> docs/fig1_setting_and_staff.pdf / .png   Fig. 1 of the report: (a) top-down setting, (b) plate-as-staff
        -> docs/plate_layout.pdf / .png             build sheet: the plate to scale, markers, grid, PLAY slot

Use the PDFs in Overleaf (vector, sharp at any size) and the PNGs in the README.
Change colors, grid or slot positions in mimi/config.py, then run this again.
"""

import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mimi import config as C  # noqa: E402
from mimi.music import melody  # noqa: E402

OUT_DIR = "docs"

# How each brick color is drawn (close to the LEGO colors bought for the game).
SWATCH = {
    "red": "#C91A09", "orange": "#FE8A18", "yellow": "#F2CD37", "light green": "#A5CA18",
    "dark green": "#237841", "light blue": "#36AEBF", "dark blue": "#0A3463", "black": "#1B1B1B",
}
PITCH_LABELS = ["C4", "D", "E", "F", "G", "A", "B", "C5"]
INK, GRID, FAINT = "#2C2C2A", "#9A9890", "#D3D1C7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def tint(hex_color, amount=0.72):
    """Mix a color with white (for the lane backgrounds)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{int(c + (255 - c) * amount):02X}" for c in (r, g, b))


def box(ax, x, y, w, h, text, face="white", edge=INK, size=8, rounded=True, **kw):
    style = "round,pad=0,rounding_size=1.2" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=face, edgecolor=edge, linewidth=0.8, **kw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DIR}/{name}.pdf and .png")


# ---------------------------------------------------------------- Fig. 1

def panel_setting(ax):
    """(a) Top-down setting. Free units, y down."""
    ax.set_xlim(0, 100)
    ax.set_ylim(64, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((2, 8), 68, 48, boxstyle="round,pad=0,rounding_size=2",
                                facecolor="#F6F1E7", edgecolor=INK, linewidth=0.9))
    ax.text(4, 11.5, "Table (6 ft)", fontsize=9, fontweight="bold", color=INK)
    box(ax, 13, 3, 46, 5, "cardboard “concert hall” backdrop", face="#F4C0D1", size=7, rounded=False)
    ax.add_patch(Circle((36, 16), 4.2, facecolor="white", edgecolor=INK, linewidth=0.9))
    ax.text(36, 16, "MIMI", ha="center", va="center", fontsize=8, color=INK)
    ax.annotate("", xy=(36, 25.5), xytext=(36, 20.4), arrowprops=dict(arrowstyle="-|>", color="#185FA5", lw=1))
    ax.text(37.2, 23.4, "camera FOV", fontsize=7, color="#185FA5")
    # plate with the note grid and corner markers
    px, py, ps = 26, 25.5, 20
    ax.add_patch(Rectangle((px, py), ps, ps, facecolor="white", edgecolor=INK, linewidth=0.8))
    for i in range(1, 8):
        ax.plot([px, px + ps], [py + i * ps / 8] * 2, color=FAINT, lw=0.5)
    for i in range(1, 6):
        ax.plot([px + i * ps / 6] * 2, [py, py + ps], color=FAINT, lw=0.5)
    for cx, cy in ((px, py), (px + ps, py), (px + ps, py + ps), (px, py + ps)):
        ax.add_patch(Rectangle((cx - 1, cy - 1), 2, 2, facecolor=INK))
    ax.text(px + ps / 2, py + ps + 2.6,"32×32 Lego baseplate (“score”)", ha="center", fontsize=6.5, color=INK)
    # PLAY slot: tape outline, any brick inside = the baton
    ax.add_patch(Rectangle((51, 33), 3.2, 6, facecolor="white", edgecolor=INK, linewidth=0.8, linestyle="--"))
    ax.add_patch(Rectangle((51.5, 33.5), 2.2, 5, facecolor=SWATCH["orange"], edgecolor=INK, linewidth=0.5))
    ax.text(55.5, 35.2, "PLAY slot", fontsize=7.5, color=INK, va="center")
    ax.text(55.5, 37.8, "(any brick =\nbaton)", fontsize=6.5, color=INK, va="top")
    box(ax, 5, 28, 15, 13, "brick trays\n(sorted by\ncolor/length)", face="#FAEEDA", size=7)
    for s in range(4):
        box(ax, 20 + s * 8.2, 50.5, 7.4, 3.6, f"Seat {s + 1}", face="#C0DD97", size=7, rounded=False)
    ax.text(36, 60, "Students (3–4) stand here", ha="center", fontsize=8, color=INK)
    box(ax, 76, 9, 21, 8, "Wizard laptop\n(Nophar)", size=7.5)
    box(ax, 79, 24, 15, 5, "speaker", size=7.5)
    box(ax, 76, 36, 21, 8, "Host/surveys\n(Wiam)", size=7.5)


def panel_staff(ax):
    """(b) Plate-as-staff mapping: lanes, beats, measures, example bricks."""
    lanes, beats = len(C.PITCHES), C.MEASURES * C.BEATS_PER_MEASURE
    lane_h = C.STUDS_PER_LANE
    ax.set_xlim(-3.2, beats + 0.4)
    ax.set_ylim(lanes * lane_h + 11.5, -2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    for lane in range(lanes):
        y = (lanes - 1 - lane) * lane_h
        color = SWATCH[C.LANE_COLORS[lane]]
        ax.add_patch(Rectangle((0, y), beats, lane_h, facecolor=tint(color), edgecolor="none"))
        ax.add_patch(Rectangle((-1.1, y + 0.2), 0.7, lane_h - 0.4, facecolor=color, edgecolor="none"))
        ax.text(-1.5, y + lane_h / 2, PITCH_LABELS[lane], ha="right", va="center", fontsize=7.5, color=INK)
    for col in range(beats + 1):
        thick = col % C.BEATS_PER_MEASURE == 0
        ax.plot([col, col], [0, lanes * lane_h], color=INK if thick else "white", lw=1.4 if thick else 0.6)
    ax.plot([0, beats], [0, 0], color=INK, lw=0.8)
    ax.plot([0, beats], [lanes * lane_h] * 2, color=INK, lw=0.8)
    for m, name in enumerate(["Tutorial", "Round 1", "Round 2", "Round 3"]):
        ax.text(m * 4 + 2, -1.0, name, ha="center", fontsize=6.5, color=INK)
    examples = melody(C.TUTORIAL_MELODY, 0) + melody([(4, 0, 4)], 3)  # tutorial E-D-C(half), whole-note G
    for n in examples:
        y = (lanes - 1 - n.lane) * lane_h
        ax.add_patch(Rectangle((n.start + 0.08, y + 0.12), n.beats - 0.16, lane_h - 0.24,
                               facecolor=SWATCH[C.LANE_COLORS[n.lane]], edgecolor=INK, linewidth=0.6))
    names = ", ".join(C.LANE_COLORS)
    split = names.index(", ", len(names) // 2) + 2
    ax.text(0, lanes * lane_h + 1.6,
            f"Row = pitch (1 stud tall). Color = pitch:\n{names[:split]}\n{names[split:]}.\n"
            "Brick length = duration: 1×1 quarter (1 beat), 1×2 half,\n"
            "1×3 dotted half (3 beats), 1×4 whole (4 beats).\n"
            "Each measure = one round, kept for the finale.",
            fontsize=6.5, color=INK, va="top", linespacing=1.35)


def figure_1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})
    panel_setting(a)
    panel_staff(b)
    fig.subplots_adjust(wspace=0.08)
    fig.canvas.draw()  # apply the equal aspect so the panel positions are final
    for ax, title in ((a, "(a) Top-down setting"), (b, "(b) Plate-as-staff mapping")):
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, 0.04, title, ha="center", fontsize=10, fontweight="bold")
    save(fig, "fig1_setting_and_staff")


# ---------------------------------------------------------------- build sheet: plate to scale

def plate_layout():
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-6, 56)
    ax.set_ylim(39.5, -8)
    n = C.PLATE_STUDS
    ax.add_patch(Rectangle((0, 0), n, n, facecolor="white", edgecolor=INK, linewidth=1))
    for i in range(n):
        for j in range(n):
            ax.add_patch(Circle((i + 0.5, j + 0.5), 0.3, facecolor="none", edgecolor="#E3E1DA", linewidth=0.4))
    # ArUco markers, real patterns, 30 mm = 3.75 studs
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, C.ARUCO_DICT))
    side = 30 / 8
    for marker_id, (mx, my) in C.MARKER_CENTERS_STUDS.items():
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, 120)
        ax.imshow(img, cmap="gray", extent=(mx - side / 2, mx + side / 2, my + side / 2, my - side / 2),
                  interpolation="nearest", zorder=3)
        ax.plot([mx, 0 if mx < 0 else n], [my, 0 if my < 0 else n], color=GRID, lw=0.6, ls=":")
        tx = mx + (side / 2 + 0.5) * (1 if mx < 0 else -1)
        ax.text(tx, my, f"id {marker_id}", ha="left" if mx < 0 else "right", va="center", fontsize=8, color=INK)
    # note grid
    lanes, beats = len(C.PITCHES), C.MEASURES * C.BEATS_PER_MEASURE
    gx, gy, lh = C.GRID_X0, C.GRID_Y0, C.STUDS_PER_LANE
    for lane in range(lanes):
        y = gy + (lanes - 1 - lane) * lh
        ax.add_patch(Rectangle((gx - 1.3, y + 0.25), 0.8, lh - 0.5, facecolor=SWATCH[C.LANE_COLORS[lane]], edgecolor="none"))
        ax.text(gx - 1.7, y + lh / 2, f"{PITCH_LABELS[lane]}  {C.LANE_COLORS[lane]}", ha="right", va="center", fontsize=6.5, color=INK)
        if lane:
            ax.plot([gx, gx + beats], [y + lh] * 2, color=FAINT, lw=0.6)
    ax.add_patch(Rectangle((gx, gy), beats, lanes * lh, facecolor="none", edgecolor=INK, linewidth=0.9))
    for m in range(1, C.MEASURES):
        ax.plot([gx + 4 * m] * 2, [gy, gy + lanes * lh], color=INK, lw=1.3)
    for m, name in enumerate(["Tutorial", "Round 1", "Round 2", "Round 3"]):
        ax.text(gx + 4 * m + 2, gy + lanes * lh + 1.0, name, ha="center", fontsize=6.5, color=INK)
    for note in melody(C.TUTORIAL_MELODY, 0) + melody([(4, 0, 4)], 3):
        y = gy + (lanes - 1 - note.lane) * lh
        ax.add_patch(Rectangle((gx + note.start + 0.06, y + 0.08), note.beats - 0.12, lh - 0.16,
                               facecolor=SWATCH[C.LANE_COLORS[note.lane]], edgecolor=INK, linewidth=0.5))
    # dimensions
    arrow = dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(0, gy - 1.2), xytext=(gx, gy - 1.2), arrowprops=arrow)
    ax.text(gx / 2, gy - 1.7, f"{gx} studs", ha="center", fontsize=7.5, color=INK)
    ax.annotate("", xy=(gx + beats + 1.2, 0), xytext=(gx + beats + 1.2, gy), arrowprops=arrow)
    ax.text(gx + beats + 1.7, gy / 2, f"{gy} studs", va="center", fontsize=7.5, color=INK)
    ax.text(n / 2, gy + lanes * lh + 3.2, f"{beats} beats (1 stud each) × {lanes} lanes (1 stud each)",
            ha="center", fontsize=7.5, color=INK)
    ax.text(n / 2, n - 1.2, f"{n} × {n} studs ({n * 8} mm)", ha="center", fontsize=7.5, color=INK)
    # PLAY slot
    sx, sy, sw, sh = C.PLAY_SLOT_STUDS
    ax.add_patch(Rectangle((sx, sy), sw, sh, facecolor="white", edgecolor=INK, linewidth=0.9, linestyle="--"))
    ax.add_patch(Rectangle((sx + 0.15, sy + 0.15), sw - 0.3, sh - 0.3, facecolor=SWATCH["orange"], edgecolor=INK, linewidth=0.5))
    ax.annotate("", xy=(n, sy + sh + 1), xytext=(sx, sy + sh + 1), arrowprops=arrow)
    ax.text((n + sx) / 2, sy + sh + 2.2, f"{sx - n:g}", ha="center", fontsize=7.5, color=INK)
    # notes column
    tx = 40
    notes = [
        (-4.5, "ArUco markers", True), (-3.0, "30 mm wide, ids 0–3", False),
        (-1.8, "centers 20 mm diagonally", False), (-0.6, "outside each corner", False),
        (sy + 0.8, "PLAY slot", True), (sy + 2.0, f"{sw:g} × {sh:g} studs, white tape outline", False),
        (sy + 3.2, "ANY brick here = baton", False),
        (24, "Example bricks", True), (25.2, "Tutorial: E, D, C half", False), (26.4, "Round 3: whole-note G", False),
        (29.6, "1 stud = 8 mm", True),
    ]
    for y, text, bold in notes:
        ax.text(tx, y, text, fontsize=8, color=INK, fontweight="bold" if bold else "normal", va="center")
    ax.text(n / 2, -6.6, "MIMI sits here (top edge)", ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(n / 2, 38.4, "Students stand here, seats 1–4", ha="center", fontsize=9, fontweight="bold", color=INK)
    save(fig, "plate_layout")


if __name__ == "__main__":
    figure_1()
    plate_layout()
