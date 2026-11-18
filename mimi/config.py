"""Every number the study protocol (proposal 6.2, "FSM variables") or the testbed calibration can change.

Edit this file, not the code. Calibration values that are measured on the day (HSV color centers)
live in calibration.json, written by calibrate.py.
"""

# ---------------------------------------------------------------- robot connection
ROBOT_IP = "192.168.8.187"  # same robot as Lab 3; change if it moved

# ---------------------------------------------------------------- FSM variables (proposal 6.2)
MAX_PLAYBACKS = 3          # build playbacks allowed per Echo round
ROUND_TIMEOUT_S = 60       # seconds to match, counted from the end of the second target playback
TEMPO_BPM = 90
READ_CONFIDENCE_MIN = 0.80  # below this, the plate read goes to the VLM for a second opinion
LLM_TIMEOUT_S = 2.0        # hint must arrive within this, else the template fallback is said
STARS_TO_WIN = 2           # of 3 rounds -> "Orchestra Graduates"
REPLAY_TARGET_AFTER_MISMATCH = True  # play MIMI's melody once more after a mismatch (LO3: compare by ear)

# ---------------------------------------------------------------- music
# Pitch lanes from the bottom of the plate (lane 0) to the top (lane 7). Fig. 1b.
PITCHES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
PITCH_WORDS = ["low C", "D", "E", "F", "G", "A", "B", "high C"]  # how MIMI says them
# Brick color per lane (the vision pipeline and the students both use it).
LANE_COLORS = ["red", "orange", "yellow", "light green", "dark green", "light blue", "dark blue", "black"]
BEATS_PER_MEASURE = 4
MEASURES = 4               # tutorial + 3 Echo rounds, kept on the plate for the finale
DURATION_WORDS = {1: "quarter", 2: "half", 4: "whole"}

# Target melodies. Each note is (lane, start_beat, beats) inside one 4-beat measure, start_beat 0-3.
# Avoid two neighbouring notes with the same pitch: same-colored bricks side by side look like one
# long brick to the camera (see README, "Known limitations").
TUTORIAL_MELODY = [(2, 0, 1), (1, 1, 1), (0, 2, 2)]  # E - D - C(half), proposal 6.1 step 4
MELODY_BANK = {
    1: [  # Round 1: quarter notes only, steps
        [(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)],
        [(4, 0, 1), (3, 1, 1), (2, 2, 1), (1, 3, 1)],
        [(2, 0, 1), (3, 1, 1), (4, 2, 1), (2, 3, 1)],
    ],
    2: [  # Round 2: quarter + half
        [(4, 0, 1), (2, 1, 1), (0, 2, 2)],
        [(0, 0, 2), (2, 2, 1), (4, 3, 1)],
        [(3, 0, 1), (4, 1, 1), (5, 2, 2)],
    ],
    3: [  # Round 3: a whole note, or a leap
        [(4, 0, 4)],
        [(0, 0, 1), (4, 1, 1), (2, 2, 2)],
        [(7, 0, 2), (4, 2, 1), (0, 3, 1)],
    ],
}

# ---------------------------------------------------------------- plate geometry (studs)
# The rectified ("top-down") image is in stud units, in the STUDENTS' view of the plate:
# x to the right along the beats, y down, so the top pitch lane (high C) has the smallest y.
PX_PER_STUD = 20
PLATE_STUDS = 32
# ArUco DICT_4X4_50 ids 0-3, glued at the plate corners. Value = marker CENTER in stud coordinates.
# 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left (students' view; MIMI sits at the top).
ARUCO_DICT = "DICT_4X4_50"
MARKER_CENTERS_STUDS = {0: (-2.5, -2.5), 1: (34.5, -2.5), 2: (34.5, 34.5), 3: (-2.5, 34.5)}
# Warped canvas: the plate plus a margin so the markers and the PLAY slot fit.
CANVAS_MARGIN_STUDS = 6
# Note grid: beat 1 of measure 1 starts at GRID_X0, the top of the high-C lane at GRID_Y0.
# 4 measures x 4 beats = 16 beat columns, 1 stud per beat; 8 lanes x 2 studs = 16 stud rows.
GRID_X0, GRID_Y0 = 8, 8
STUDS_PER_BEAT = 1
STUDS_PER_LANE = 2
# PLAY slot where the conductor puts the baton: (x, y, w, h) in studs, may be off the plate.
# The baton can be ANY brick (black is a pitch color): mark the slot with an outline of white or light-gray
# tape only, never filled with colored tape, so an empty slot reads as empty.
PLAY_SLOT_STUDS = (35.5, 14, 2, 4)
BATON_FILL_FRACTION = 0.5  # share of PLAY-slot pixels that must be colored or dark to count as "baton in"

# ---------------------------------------------------------------- vision thresholds
MOTION_PIXEL_FRACTION = 0.02  # share of plate pixels changing between two frames = a hand is moving over it
CELL_MIN_COLOR_SAT = 70     # below this saturation a cell is white baseplate (empty)
BLACK_MAX_VALUE = 60        # darker than this (HSV value) = a black brick; dark blue must stay above it
CELL_MAX_COLOR_DIST = 40.0  # farther than this from every calibrated color = unknown cell
FRAMES_THAT_MUST_AGREE = 2

# ---------------------------------------------------------------- motion (degrees; SDK limits are ±40 pitch/roll, ±65 head-body yaw)
YAW_LIMIT = 45
TILT_LIMIT = 25
LOOK_GROUP_PITCH = -5      # head slightly up, toward the standing students
READ_PITCH = 28            # downward "reading" look: whole plate + markers in view (calibrate!)
LISTEN_PITCH = 18          # while students build: can still see the PLAY slot (calibrate!)
LISTEN_ROLL = 10           # the "listening" tilt
# Where each seat is, as body yaw in degrees (+ = MIMI turns to its left). Seat 1 is on MIMI's right.
SEAT_YAW = {1: -30, 2: -10, 3: 10, 4: 30}
BREATH_PITCH_DEG = 1.5     # idle breathing amplitude
BREATH_PERIOD_S = 4.0
BOB_PITCH_DEG = 7          # conducting head bob amplitude during playback
DOA_TURN_GAIN = 0.5        # how much of the sound direction MIMI turns toward (0 = ignore sound)
DOA_HOLD_S = 2.5           # how long MIMI keeps looking toward a voice before drifting back
FACE_TRACKING_WHILE_SPEAKING = True  # daemon face tracking blended in while addressing the group
FACE_TRACKING_WEIGHT = 0.5

# ---------------------------------------------------------------- speech
MAX_HINT_WORDS = 15
TTS_VOICE = "alloy"
TTS_CACHE_DIR = "tts_cache"
