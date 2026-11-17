"""Pre-study setup (proposal 6.2): check the camera view and fit the brick colors under event lighting.

    python calibrate.py view            # head to the READ pose, save the raw + rectified + labelled plate
    python calibrate.py view --listen   # same from the LISTEN pose: is the PLAY slot still visible?
    python calibrate.py colors          # fit HSV centers from the calibration staircase -> calibration.json
    python calibrate.py test            # read the plate 3 times and print the notes (the "3 test plates")
    python calibrate.py test --image photo.jpg   # same, offline from a photo

Calibration staircase for `colors`: put one SHORT (1x2) brick of each color in its own lane, lane i at
beat i+1 of the plate (low C red on beat 1, D orange on beat 2, ... high C pink on beat 8).
Tune READ_PITCH / LISTEN_PITCH / GRID_* / PLAY_SLOT_STUDS in mimi/config.py until the outputs look right.
"""

import argparse
import json

import cv2

from mimi import config as C
from mimi.music import to_text
from mimi.plate_reader import CALIBRATION_FILE, PlateReader, annotate


def grab(args):
    if args.image:
        return cv2.imread(args.image), None
    from mimi.robot import Body, connect
    mini = connect()
    body = Body(mini)
    body.listening_tilt() if args.listen else body.reading_look()
    body.set_mode("still")
    frame = None
    for _ in range(80):
        frame = mini.media.get_frame()
        if frame is not None:
            break
    return frame, body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["view", "colors", "test"])
    parser.add_argument("--listen", action="store_true", help="use the LISTEN head pose instead of READ")
    parser.add_argument("--image", help="use a saved photo instead of the robot camera")
    args = parser.parse_args()

    frame, body = grab(args)
    if frame is None:
        raise SystemExit("No camera frame.")
    reader = PlateReader()
    try:
        if args.what == "view":
            cv2.imwrite("calib_raw.jpg", frame)
            warped, ok = reader.warp(frame)
            print("all 4 markers found" if ok else "NOT all markers found: move the head or the plate")
            if warped is not None:
                read = reader.read(frame, list(range(C.MEASURES)))
                cv2.imwrite("calib_rectified.jpg", annotate(warped, read.notes))
                print("baton in PLAY slot:", read.baton)
                print("saved calib_raw.jpg and calib_rectified.jpg: grid lines must sit on the stud rows")

        elif args.what == "colors":
            warped, ok = reader.warp(frame)
            if not ok:
                raise SystemExit("Need all 4 markers for color calibration.")
            hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            colors = {}
            for lane, name in enumerate(C.LANE_COLORS):
                top = C.GRID_Y0 + (len(C.PITCHES) - 1 - lane) * C.STUDS_PER_LANE
                value = reader._patch(hsv, C.GRID_X0 + lane, top, 1, C.STUDS_PER_LANE)
                colors[name] = [int(v) for v in value]
                print(f"  {name:7s} lane {C.PITCHES[lane]}: HSV {colors[name]}")
            hues = sorted((v[0], k) for k, v in colors.items())
            gaps = [(min(abs(a[0] - b[0]), 180 - abs(a[0] - b[0])), a[1], b[1]) for a, b in zip(hues, hues[1:] + hues[:1])]
            closest = min(gaps)
            print(f"closest pair: {closest[1]} / {closest[2]} ({closest[0]} hue units apart)"
                  + ("  <- may be confused, check the test read" if closest[0] < 8 else ""))
            with open(CALIBRATION_FILE, "w") as f:
                json.dump({"colors": colors}, f, indent=2)
            print(f"saved {CALIBRATION_FILE}")

        else:  # test
            previous = None
            for i in range(3):
                read = reader.read(frame if args.image else body.mini.media.get_frame(), list(range(C.MEASURES)), previous)
                previous = read.warped
                print(f"read {i + 1}: conf {read.confidence:.2f}  hand/moving={read.hand_in_way}  baton={read.baton}")
                for m in range(C.MEASURES):
                    print(f"   measure {m}: {to_text([n for n in read.notes if n.start // C.BEATS_PER_MEASURE == m])}")
                for issue in read.issues:
                    print("   issue:", issue)
    finally:
        if body is not None:
            body.neutral()
            body.stop()


if __name__ == "__main__":
    main()
