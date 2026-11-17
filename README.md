# Brick Beats with Maestro MIMI

Controller code for the Reachy Mini conductor game in our INFO 5356 proposal. A white Lego baseplate
is a music staff: each row is a pitch, each column is a beat, and brick length is the note's duration.
MIMI plays a 4-beat melody twice and the team builds it. The round's conductor then puts the baton
brick in the PLAY slot, and MIMI reads the plate with its camera and plays back exactly what was built.

The LLM/VLM setup (Ollama by default, OpenAI with `--api`) is the same as
[Lab 3](https://github.com/nopharshalom/Lab-3-GenAI-for-HRI). The models are used for less here than
in Lab 3: perception and feedback choice are deterministic, the **VLM** only gives a second opinion on
low-confidence plate reads, and the **LLM** only words the short hint after a mismatch.

## How the code matches the proposal (Section 4, Fig. 2)

| Proposal | Code |
|---|---|
| Plate reader: ArUco homography → HSV grid → notes + confidence | `mimi/plate_reader.py` |
| VLM second opinion on low-confidence reads | `plate_reader.vlm_second_opinion` |
| Game FSM: Greet, Tutorial, [PlayTarget → Listen → ReadPlate → Compare → Match/Mismatch]×3, Finale, Failsafe | `mimi/game.py` |
| Melody comparator, synthesized notes at 90 BPM | `mimi/music.py` |
| Dialogue manager: scripted lines + LLM hint (≤15 words, grade 6, 2 s template fallback) | `mimi/dialogue.py` |
| Behavior arbiter (deliberative poses override reactive ones) and reactive layer (breathing, beat bob, turning toward voices, face tracking while speaking) | `mimi/robot.py` (`Body`) |
| Speaker: pre-rendered TTS clips + `push_audio_sample` notes | `mimi/robot.py` (`Speaker`), `tools/prerender_tts.py` |
| Wizard-of-Oz keyboard: correct a read, repeat, skip, FAILSAFE | `mimi/wizard.py` |
| Time-stamped CSV logging, plate images | `mimi/session_log.py` → `logs/`, `captures/` |
| DVs: playbacks-to-match, stars, Wizard interventions, read latency/accuracy, Gini | `analyze.py` |
| All FSM variables (max playbacks, timeout, tempo, melody bank, thresholds, LLM timeout) | `mimi/config.py` |

## Setup

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

Free models (same as Lab 3): install [Ollama](https://ollama.com), then pull the two models:

```bash
ollama pull llama3.2
```

```bash
ollama pull gemma3:4b
```

OpenAI models: put `OPENAI_API_KEY=...` in `.env` and add `--api` to any command.
Set `ROBOT_IP` in `mimi/config.py`.

## Before the event

1. Print the markers with `python tools/make_markers.py` and glue them at the plate corners (see that file for where).
2. Pre-render the voice once with `python tools/prerender_tts.py --api`. After that the game plays these
   clips on the robot even with the free models.
3. On the day, under the event lighting:
   - `python calibrate.py view` saves `calib_rectified.jpg`. The grid lines must sit on the stud rows.
     Tune `READ_PITCH` and `GRID_X0/GRID_Y0` in `mimi/config.py` until they do.
   - `python calibrate.py view --listen` checks the PLAY slot is still visible at the listening pose
     (`LISTEN_PITCH`, `PLAY_SLOT_STUDS`).
   - Build the color staircase (one short brick per color, lane *i* on beat *i+1*), then run
     `python calibrate.py colors` to write `calibration.json`.
   - Build each of the 3 test plates and run `python calibrate.py test`.
   - Check `SEAT_YAW` by watching MIMI turn to each seat (use `--start round1` and a wrong build).

## Running a session

```bash
python run_mimi.py --session S01 --condition directed
```

```bash
python run_mimi.py --session S02 --condition group --api
```

Alternate `--condition` by session (RQ1). `--start round2` resumes after a crash.

**Wizard keys** (type the key in the terminal, then press Enter): `p` = play (use when the baton
isn't detected), `c` = correct the read (e.g. `c E1 D1 C2`, where `R` is a rest), `x` = mark the
last read as wrong, `r` = repeat, `n` = skip ahead, `f` = FAILSAFE, `g` = resume, `q` = quit.

After the sessions, run `python analyze.py` to write `session_summary.csv`.

### Without the robot

```bash
python tools/make_test_plate.py "E1 D1 C2 | C1 D1 E1 F1 | G1 E1 C2 | G4" test_plate.jpg --baton
```

```bash
python run_mimi.py --session T --condition directed --dry-run --plate-image test_plate.jpg
```

In a dry run there is no motion, and the notes play on the laptop. Press `p` to request a playback.
Leave out `--plate-image` to type every read with `c`.

## Design choices we made while coding (check these against the proposal)

- **The 60 s round clock only runs while students build.** It pauses while MIMI reads, talks or plays,
  so a slow read or model call never costs the team time.
- **In the GROUP condition the LLM hint never names a beat or a seat.** It gets "somewhere in the
  measure, one note should be higher", so the two RQ1 conditions only differ in where the feedback is aimed.
- **MIMI replays its melody after each mismatch** (`REPLAY_TARGET_AFTER_MISMATCH`), so students can
  compare the two melodies by ear (LO3). This doesn't count toward the 3 playbacks.
- **The hint request starts while the build plays back.** It is used only if the LLM answered
  within 2 s; otherwise MIMI says a template hint. Each hint's source is logged.
- **The baton has to be lifted and put back** to trigger another read, so a baton left in the slot
  doesn't keep triggering.
- **The finale plays what is on the plate**, including rounds that were lost.

## Known limitations / not yet tested on the robot

- Everything except the robot I/O has been tested on synthetic plate photos. That covers the plate
  reader, the comparator, the synthesizer, the hint fallback and a full dry-run game. **The robot
  calls have not run on hardware yet:** `push_audio_sample` notes, DoA from `/api/state/doa`,
  `start_head_tracking`, body yaw. The signs in `SEAT_YAW` and the DoA→yaw mapping also need checking on the robot.
- Two same-colored bricks side by side look like one long brick to the camera. The melody bank avoids
  repeated neighbouring pitches for this reason. A 3-beat run is read as half + quarter and flagged as low confidence.
- A hand is detected only when it hides a marker or moves between the two frames. A hand held still
  over the plate shows up as unknown cells, which lower the confidence and send the read to the VLM.
