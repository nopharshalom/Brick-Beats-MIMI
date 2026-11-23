"""The game FSM (deliberative layer, proposal 4.1):

    GREET -> TUTORIAL -> [PLAY_TARGET -> LISTEN -> READ_PLATE -> COMPARE -> MATCH / MISMATCH] x3 -> FINALE
    + FAILSAFE (Wizard 'f'), from anywhere; Wizard 'g' resumes at the start of the interrupted stage.

The round clock (60 s) only runs while the students are building: it pauses while MIMI reads, talks
or plays, so slow reads or model calls never cost the team time.
"""

import random
import time

from . import config as C
from .dialogue import TUTORIAL_STEPS, PendingHint, line
from .music import compare, in_measure, melody, parse_wizard_notes, synthesize, to_text
from .plate_reader import PlateRead, PlateReader, annotate, vlm_second_opinion
from .robot import Body, Speaker, connect
from .session_log import SessionLog
from .wizard import Wizard


class Failsafe(Exception):
    pass


class Quit(Exception):
    pass


class Game:
    def __init__(self, args):
        self.args = args
        self.condition = args.condition
        self.log = SessionLog(args.session, args.condition)
        self.mini = connect(args.dry_run, args.plate_image, args.sim, args.webcam)
        laptop_media = args.dry_run or args.sim  # frames from --plate-image, sound on the laptop, no mic
        self.body = Body(self.mini, laptop_media)
        self.speaker = Speaker(self.mini, laptop_media)
        # one reader per head pose: the homography is only valid for the pose it was found in
        self.readers = {"read": PlateReader(), "listen": PlateReader()}
        self.wizard = Wizard(self.log)
        self.rng = random.Random(args.seed)
        self.stars = {}           # round -> 0/1
        self.plate = {}           # measure -> last accepted notes (finale fallback)
        self.last_line = ""

    # ------------------------------------------------------------ helpers

    def say(self, text, remember=True):
        if remember:
            self.last_line = text
        self.log.event("say", text=text)
        self.speaker.say(text)
        self.check_wizard()

    def check_wizard(self, *wanted):
        """Handle FAILSAFE / quit / repeat anywhere; return a wanted command (key, arg) or None."""
        cmd = self.wizard.poll(*wanted, "r")
        if cmd is None:
            return None
        key = cmd[0]
        if key == "f":
            raise Failsafe()
        if key == "q":
            raise Quit()
        if key == "r":
            if self.last_line:
                self.speaker.say(self.last_line)
            return self.check_wizard(*wanted)
        return cmd

    def play(self, notes, measures, label):
        """Play notes with the conducting bob. `measures`: the measures to play, in order."""
        first = measures[0] * C.BEATS_PER_MEASURE
        total = len(measures) * C.BEATS_PER_MEASURE
        audio = synthesize(notes, self.speaker.sample_rate, first_beat=first, total_beats=total)
        self.log.event("playback", what=label, notes=to_text(notes))
        self.body.look_at_group()
        self.body.set_mode("bob")
        try:
            self.speaker.play(audio)
        finally:
            self.body.set_mode("idle")
        self.check_wizard()

    def frame(self):
        for _ in range(20):
            frame = self.mini.media.get_frame()
            if frame is not None:
                return frame
            time.sleep(0.1)
        return None

    def peek(self, measures):
        """A quick single-frame look from the listening pose: (PlateRead or None)."""
        frame = self.frame()
        if frame is None:
            return None
        return self.readers["listen"].read(frame, measures)

    def read_plate(self, measures, active):
        """Look down and read until two frames agree with no hand in the way (proposal 4.2).
        Low confidence -> VLM second opinion for the active measure. Wizard 'c' overrides.
        Returns a PlateRead, or None if the plate could not be read (Wizard can then correct)."""
        self.body.reading_look()
        self.say(line("reading"), remember=False)
        start = time.time()
        reader, warned, result = self.readers["read"], False, None
        for _ in range(8):
            cmd = self.check_wizard("c")
            if cmd:
                return self.wizard_read(cmd[1], active, start)
            f1 = self.frame()
            if f1 is None:
                break
            r1 = reader.read(f1, measures)
            time.sleep(0.3)
            f2 = self.frame()
            r2 = reader.read(f2, measures, previous_warped=r1.warped) if f2 is not None else r1
            if r2.hand_in_way or r1.hand_in_way:
                if not warned:
                    self.body.goto(pitch=C.LOOK_GROUP_PITCH, duration=0.6)
                    self.say(line("hands"), remember=False)
                    self.body.reading_look()
                    warned = True
                time.sleep(1.0)
                continue
            if r1.notes == r2.notes:
                result = r2
                break
        if result is None:
            self.log.event("read_failed", latency_s=round(time.time() - start, 2))
            self.body.look_at_group()
            self.body.set_mode("idle")
            self.say(line("cannot_read"))
            return None
        if result.confidence < C.READ_CONFIDENCE_MIN:
            vlm_notes = vlm_second_opinion(result.warped, active)
            if vlm_notes is not None:
                keep = [n for n in result.notes if not active * C.BEATS_PER_MEASURE <= n.start < (active + 1) * C.BEATS_PER_MEASURE]
                result.notes, result.source = sorted(keep + vlm_notes), "vlm"
        image = self.log.save_image(annotate(result.warped, result.notes, active), "read")
        seat_notes = {s: sum(1 for n in in_measure(result.notes, active) if n.start % C.BEATS_PER_MEASURE + 1 == s)
                      for s in range(1, 5)}
        self.log.event("read", source=result.source, notes=to_text(in_measure(result.notes, active)),
                       all_notes=to_text(result.notes), confidence=round(result.confidence, 3),
                       issues=result.issues, latency_s=round(time.time() - start, 2), image=image,
                       seat_notes=seat_notes)
        print(f"  read ({result.source}, conf {result.confidence:.2f}): {to_text(in_measure(result.notes, active))}")
        self.body.set_mode("idle")
        return result

    def wizard_read(self, text, active, start):
        notes = parse_wizard_notes(text, active)
        if notes is None:
            print("[wizard] could not parse that; example: E1 D1 C2")
            return None
        others = [n for m, ns in self.plate.items() if m != active for n in ns]
        self.log.event("read", source="wizard", notes=to_text(notes), confidence=1.0,
                       latency_s=round(time.time() - start, 2))
        self.body.set_mode("idle")
        return PlateRead(notes=sorted(others + notes), confidence=1.0, markers_ok=True, source="wizard")

    # ------------------------------------------------------------ stages

    def greet(self):
        self.log.transition("GREET")
        self.body.look_at_group()
        self.body.face_tracking(True)
        self.body.happy_wiggle()
        self.say(line("greet"))
        self.body.bow()
        for key in ("intro_1", "intro_2", "intro_3"):
            self.say(line(key))
        self.body.face_tracking(False)

    def tutorial(self):
        self.log.round = 0
        self.log.transition("TUTORIAL")
        self.say(line("tutorial_start"))
        target = melody(C.TUTORIAL_MELODY, measure=0)
        for i, note in enumerate(target):
            seat = note.start + 1
            self.body.look_at_seat(min(seat, 4))
            self.say(TUTORIAL_STEPS[i].format(color=C.LANE_COLORS[note.lane], pitch=C.PITCH_WORDS[note.lane]))
            self.body.listening_tilt()
            deadline = time.time() + 25
            while time.time() < deadline:
                if self.check_wizard("n"):  # a 'c' stays queued for the read below
                    break
                seen = self.peek([0])
                if seen and note in seen.notes:
                    self.log.event("tutorial_step_seen", step=i + 1)
                    self.body.happy_wiggle()
                    break
                time.sleep(1.0)
        result = self.read_plate([0], active=0)
        built = in_measure(result.notes, 0) if result else target
        self.plate[0] = built
        self.say(line("tutorial_done"))
        self.play(built, [0], "tutorial build")

    def echo_round(self, n):
        self.log.round = n
        measure = n  # measure 0 = tutorial, measures 1-3 = rounds 1-3
        target = melody(self.rng.choice(C.MELODY_BANK[n]), measure)
        conductor = (n - 1) % 4 + 1
        self.log.event("round_start", target=to_text(target), conductor_seat=conductor)

        # PLAY_TARGET
        self.log.transition("PLAY_TARGET")
        self.body.look_at_group()
        self.say(line("round_intro", n=n))
        self.body.look_at_seat(conductor)
        self.say(line("conductor", seat=conductor))
        self.play(target, [measure], "target 1")
        self.say(line("round_again"), remember=False)
        self.play(target, [measure], "target 2")
        self.say(line("round_build"))

        playbacks, build_time, warned, diff = 0, 0.0, False, None
        while True:
            # LISTEN: students build; MIMI waits for the baton (or Wizard 'p'/'c'/'n')
            self.log.transition("LISTEN")
            self.body.listening_tilt()
            baton_was_in = bool((seen := self.peek([measure])) and seen.baton)
            trigger = None
            last = time.time()
            while trigger is None:
                now = time.time()
                build_time += now - last
                last = now
                if build_time >= C.ROUND_TIMEOUT_S:
                    trigger = ("timeout", "")
                    break
                if not warned and build_time >= C.ROUND_TIMEOUT_S - 20:
                    warned = True
                    self.say(line("time_warning"), remember=False)
                    last = time.time()  # speaking time does not count
                cmd = self.check_wizard("p", "c", "n")
                if cmd:
                    trigger = cmd
                    break
                seen = self.peek([measure])
                baton = bool(seen and seen.baton)
                if baton and not baton_was_in:
                    trigger = ("baton", "")
                baton_was_in = baton
                time.sleep(0.3)
            self.log.event("trigger", by=trigger[0], build_time_s=round(build_time, 1))
            if trigger[0] in ("timeout", "n"):
                break

            # READ_PLATE
            self.log.transition("READ_PLATE")
            result = self.wizard_read(trigger[1], measure, time.time()) if trigger[0] == "c" \
                else self.read_plate(list(range(measure + 1)), active=measure)
            if result is None:
                continue  # not counted as a playback; Wizard can correct with 'c'
            built = in_measure(result.notes, measure)
            self.plate[measure] = built

            # COMPARE (start the LLM hint now so it is ready after the playback)
            self.log.transition("COMPARE")
            diff = compare(target, built, measure)
            pending = PendingHint(diff, self.condition) if diff else None
            self.log.event("compare", match=diff is None, built=to_text(built), target=to_text(target),
                           diff=None if diff is None else {"beat": diff.beat, "kind": diff.kind,
                                                           "direction": diff.direction() or diff.length()})
            playbacks += 1
            self.say(line("playing_build"), remember=False)
            self.play(built, [measure], f"build playback {playbacks}")

            if diff is None:  # MATCH
                self.log.transition("MATCH")
                self.body.happy_wiggle()
                self.say(line("match"))
                self.stars[n] = 1
                self.log.event("round_result", won=True, playbacks=playbacks, build_time_s=round(build_time, 1))
                return

            # MISMATCH
            self.log.transition("MISMATCH")
            self.body.curious_tilt()
            if self.condition == "directed":
                self.body.look_at_seat(diff.seat)
                self.say(line("mismatch_directed", seat=diff.seat, beat=diff.beat))
            else:
                self.body.look_at_group()
                self.say(line("mismatch_group"))
            hint, source, raw, latency = pending.result()
            self.log.event("hint", text=hint, source=source, raw=raw, llm_latency_s=latency)
            self.say(hint)
            self.body.look_at_group()
            if playbacks >= C.MAX_PLAYBACKS:
                break
            if C.REPLAY_TARGET_AFTER_MISMATCH:
                self.say(line("replay_target"), remember=False)
                self.play(target, [measure], "target replay")
            left = C.MAX_PLAYBACKS - playbacks
            self.say(line("playbacks_left", left=left, s="" if left == 1 else "s"))

        # LOST: MIMI demonstrates the answer
        self.log.transition("LOST")
        self.stars[n] = 0
        self.log.event("round_result", won=False, playbacks=playbacks, build_time_s=round(build_time, 1))
        self.body.droop()
        self.body.look_at_group()
        self.say(line("lost"))
        self.play(target, [measure], "answer demo")
        if diff is not None:
            if diff.target is not None:
                self.say(line("lost_answer", beat=diff.beat, color=C.LANE_COLORS[diff.target.lane],
                              duration=C.DURATION_WORDS.get(diff.target.beats, "long")))
            else:
                self.say(line("lost_rest", beat=diff.beat))

    def finale(self):
        self.log.round = 4
        self.log.transition("FINALE")
        self.body.look_at_group()
        self.say(line("finale_intro"))
        measures = list(range(C.MEASURES))
        result = self.read_plate(measures, active=C.MEASURES - 1)
        notes = result.notes if result else [n for m in measures for n in self.plate.get(m, [])]
        self.play(notes, measures, "finale symphony")
        stars = sum(self.stars.values())
        won = stars >= C.STARS_TO_WIN
        self.log.event("game_result", stars=stars, won=won, per_round=self.stars)
        if won:
            self.body.happy_wiggle()
            self.say(line("win", stars=stars))
        else:
            self.say(line("lose", stars=stars, s="" if stars == 1 else "s"))
        self.body.bow()
        self.say(line("goodbye"))

    def failsafe(self):
        self.log.transition("FAILSAFE")
        self.body.face_tracking(False)
        try:
            self.speaker.say(line("failsafe"))
        except Exception as error:
            print(f"  (could not speak: {error})")
        self.body.neutral()
        print("[wizard] FAILSAFE. Type g + Enter to resume this stage, q + Enter to quit.")
        while True:
            cmd = self.wizard.poll("g", "q")
            if cmd and cmd[0] == "q":
                raise Quit()
            if cmd and cmd[0] == "g":
                self.speaker.say(line("resume"))
                return
            time.sleep(0.2)

    # ------------------------------------------------------------ main loop

    def run(self):
        self.wizard.start()
        stages = [("greet", self.greet), ("tutorial", self.tutorial)]
        stages += [(f"round{n}", lambda n=n: self.echo_round(n)) for n in (1, 2, 3)]
        stages += [("finale", self.finale)]
        start = {"greet": 0, "tutorial": 1, "round1": 2, "round2": 3, "round3": 4, "finale": 5}[self.args.start]
        i = start
        try:
            while i < len(stages):
                try:
                    stages[i][1]()
                    i += 1
                except Failsafe:
                    self.failsafe()  # then re-run the interrupted stage from its start
        except (Quit, KeyboardInterrupt):
            print("\nStopped.")
            self.log.event("quit", stage=stages[min(i, len(stages) - 1)][0])
        finally:
            try:
                self.body.neutral()
            finally:
                self.body.stop()
            print(f"Log saved to {self.log.path}")
