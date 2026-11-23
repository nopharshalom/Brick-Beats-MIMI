"""Pre-study setup (proposal 6.2): check the camera view and fit the brick colors under event lighting.

    python calibrate.py view            # head to the READ pose, save the raw + rectified + labelled plate
    python calibrate.py view --listen   # same from the LISTEN pose: is the PLAY slot still visible?
    python calibrate.py colors          # fit HSV centers from the calibration staircase -> calibration.json
    python calibrate.py test            # read the plate 3 times and print the notes (the "3 test plates")
    python calibrate.py test --image photo.jpg   # same, offline from a photo
    python calibrate.py view --webcam            # any of the above with the laptop camera (no robot)
    python calibrate.py preview --webcam         # live camera window: found markers outlined, q to quit

Calibration staircase for `colors`: put one 1x1 brick of each color in its own lane, lane i at
beat i+1 of the plate (low C red on beat 1, D orange on beat 2, ... high C black on beat 8).
Tune READ_PITCH / LISTEN_PITCH / GRID_* / PLAY_SLOT_STUDS in mimi/config.py until the outputs look right.
"""

import argparse
import json

import cv2

from mimi import config as C
from mimi.music import to_text
from mimi.plate_reader import CALIBRATION_FILE, PlateReader, annotate


def frame_source(args):
    """(function returning the next frame, Body or None). Robot camera by default."""
    if args.image:
        return (lambda: cv2.imread(args.image)), None
    if args.webcam is not None:
        from mimi.robot import open_webcam
        capture = open_webcam(args.webcam)

        def next_webcam_frame():
            capture.grab()
            ok, frame = capture.read()
            return frame if ok else None
        return next_webcam_frame, None
    from mimi.robot import Body, connect
    mini = connect()
    body = Body(mini)
    body.listening_tilt() if args.listen else body.reading_look()
    body.set_mode("still")

    def next_robot_frame():
        for _ in range(80):
            frame = mini.media.get_frame()
            if frame is not None:
                return frame
        return None
    return next_robot_frame, body


def live_preview(next_frame, reader):
    """Live camera window to aim the camera: markers outlined in green, status line on top.
    q / Esc quits, s saves calib_raw.jpg."""
    print("Live preview: q or Esc to quit, s to save the frame. Aim until it says 'all 4 markers'.")
    while True:
        frame = next_frame()
        if frame is None:
            print("Camera stopped sending frames.")
            break
        shown = frame.copy()
        corners, ids, _ = reader.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = sorted(int(i) for i in ids.flatten()) if ids is not None else []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(shown, corners, ids, (0, 255, 0))
        wanted = sorted(C.MARKER_CENTERS_STUDS)
        missing = [i for i in wanted if i not in found]
        if frame.mean() < 15:
            status, color = "IMAGE IS BLACK: open the camera shutter / camera key", (0, 0, 255)
        elif not missing:
            status, color = "all 4 markers found", (0, 200, 0)
        else:
            status, color = f"missing marker id {', '.join(map(str, missing))}", (0, 0, 255)
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 60), (255, 255, 255), -1)
        cv2.putText(shown, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        scale = 960 / shown.shape[1]
        cv2.imshow("MIMI camera (q to quit, s to save)", cv2.resize(shown, None, fx=scale, fy=scale))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            cv2.imwrite("calib_raw.jpg", frame)
            print("saved calib_raw.jpg")
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["preview", "view", "colors", "test"])
    parser.add_argument("--listen", action="store_true", help="use the LISTEN head pose instead of READ")
    parser.add_argument("--image", help="use a saved photo instead of the robot camera")
    parser.add_argument("--webcam", nargs="?", const=0, type=int, metavar="INDEX",
                        help="use the laptop camera (default 0) instead of the robot camera")
    args = parser.parse_args()

    next_frame, body = frame_source(args)
    frame = next_frame()
    if frame is None:
        raise SystemExit("No camera frame.")
    if frame.mean() < 15:
        print("The camera image is almost black: open the privacy shutter / camera key, or point it at the plate.")
    reader = PlateReader()
    try:
        if args.what == "preview":
            live_preview(next_frame, reader)
        elif args.what == "view":
            cv2.imwrite("calib_raw.jpg", frame)
            warped, ok = reader.warp(frame)
            print("all 4 markers found" if ok else "NOT all markers found: move the head or the plate")
            if warped is not None:
                read = reader.read(frame, list(range(C.MEASURES)))
                cv2.imwrite("calib_rectified.jpg", annotate(warped, read.notes))
                print("brick (baton) in PLAY slot:", read.baton)
                print("saved calib_raw.jpg and calib_rectified.jpg: grid lines must sit on the stud rows")

        elif args.what == "colors":
            warped, ok = reader.warp(frame)
            if not ok:
                raise SystemExit("Need all 4 markers for color calibration.")
            faces = reader.staircase_faces(warped)  # the bricks' top faces, not their shaded sides
            colors = {}
            for lane, name in enumerate(C.LANE_COLORS):
                if faces[lane] is None:
                    raise SystemExit(f"No brick found for {name} (lane {C.PITCHES[lane]}, beat {lane + 1}). "
                                     "Check the staircase in calib_rectified.jpg from `calibrate.py view`.")
                colors[name] = [int(v) for v in faces[lane]]
                print(f"  {name:11s} lane {C.PITCHES[lane]}: HSV {colors[name]}")
                if name != "black" and colors[name][2] < C.BLACK_MAX_VALUE + 20:
                    print(f"    <- {name} is almost as dark as black: add light, or it may be read as black")
            if "black" in colors and colors["black"][2] >= C.BLACK_MAX_VALUE:
                print(f"  <- black reads too bright (value {colors['black'][2]}): lower the lamp glare or raise BLACK_MAX_VALUE")
            hues = sorted((v[0], k) for k, v in colors.items() if k != "black")  # black has no hue
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
                read = reader.read(frame if i == 0 else next_frame(), list(range(C.MEASURES)), previous)
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
