"""MIMI's body: the behavior arbiter, the reactive layer, and the speaker (proposal 4.1, Fig. 2).

Deliberative code (the game FSM) calls `body.goto(...)` for poses with meaning ("read the plate",
"turn to Seat 3"). That pose becomes the ANCHOR. The reactive layer runs in a background thread and
streams small offsets around the anchor with `set_target`: idle breathing, conducting head bob at the
tempo, and turning toward a voice (microphone direction of arrival). While a deliberative move runs,
the arbiter lock keeps the reactive layer out, so deliberative poses always win.
"""

import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.request
import wave
from dataclasses import dataclass, replace

import numpy as np

from . import config as C
from .models import TTS_MODEL, client

# Hide GStreamer's harmless RTCP spam from the camera stream (as in Lab 3). Before reachy_mini starts it.
os.environ.setdefault("GST_DEBUG", "rtpsession:0")


def clamp(value, limit):
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class Pose:
    yaw: float = 0.0        # head yaw relative to body, degrees, + = left
    pitch: float = C.LOOK_GROUP_PITCH  # + = look down
    roll: float = 0.0       # + = tilt
    body_yaw: float = 0.0   # degrees, + = left
    antennas: tuple = (0.0, 0.0)  # radians, (right, left), 0 = straight up


# ---------------------------------------------------------------- connection

def connect(dry_run=False, plate_image=None, sim=False, webcam=None):
    """The real robot; --sim: the MuJoCo simulator on this laptop (`reachy-mini-daemon --sim --no-media`),
    which moves like the robot while frames come from --webcam or --plate-image and sound plays on the
    laptop; --dry-run: no robot at all."""
    if dry_run:
        return FakeMini(plate_image, webcam)
    from reachy_mini import ReachyMini
    if sim:
        mini = SimMini(ReachyMini(host="localhost", connection_mode="localhost_only", media_backend="no_media",
                                  automatic_body_yaw=False), plate_image, webcam)
    else:
        mini = ReachyMini(host=C.ROBOT_IP, connection_mode="network", media_backend="default",
                          automatic_body_yaw=False)  # MIMI turns its body to seats on purpose
    mini.enable_motors()
    return mini


def head_matrix(pose):
    try:
        from reachy_mini.utils import create_head_pose
    except ImportError:  # --dry-run on a laptop without the SDK
        return None
    return create_head_pose(yaw=clamp(pose.yaw, C.YAW_LIMIT), pitch=clamp(pose.pitch, C.TILT_LIMIT + 15),
                            roll=clamp(pose.roll, C.TILT_LIMIT), degrees=True)


# ---------------------------------------------------------------- sound direction

class DoaPoller(threading.Thread):
    """Polls the daemon's /api/state/doa: angle 0 = MIMI's left, pi/2 = front, pi = right."""

    def __init__(self, enabled):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.yaw = None      # degrees toward the last voice, + = left
        self.heard_at = 0.0

    def run(self):
        url = f"http://{C.ROBOT_IP}:8000/api/state/doa"
        while self.enabled:
            try:
                with urllib.request.urlopen(url, timeout=0.3) as response:
                    data = json.loads(response.read() or b"null")
                if data and data.get("speech_detected"):
                    self.yaw = math.degrees(math.pi / 2 - float(data["angle"]))
                    self.heard_at = time.time()
            except Exception:
                pass
            time.sleep(0.25)


# ---------------------------------------------------------------- arbiter + reactive layer

class Body:
    def __init__(self, mini, dry_run=False):
        self.mini = mini
        self.dry_run = dry_run
        self.lock = threading.RLock()
        self.anchor = Pose()
        self.mode = "idle"   # idle | listen | bob | still
        self.bob_start = 0.0
        self.tracking = False
        self.doa = DoaPoller(enabled=not dry_run and C.DOA_TURN_GAIN > 0)
        self.doa.start()
        self._stop = False
        self._thread = threading.Thread(target=self._reactive_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ deliberative poses (win over reactive)

    def goto(self, duration=0.6, method="minjerk", **changes):
        """Move to a new anchor pose. Fields not given keep their current value."""
        target = replace(self.anchor, **changes)
        with self.lock:
            for attempt in range(2):  # the daemon occasionally rejects a move: retry once (Lab 3)
                try:
                    self.mini.goto_target(head=head_matrix(target), antennas=list(target.antennas),
                                          body_yaw=math.radians(target.body_yaw), duration=duration, method=method)
                    break
                except Exception as error:
                    if attempt == 1:
                        print(f"  (robot skipped a move: {error})")
            self.anchor = target

    def set_mode(self, mode):
        if mode == "bob":
            self.bob_start = time.time()
        self.mode = mode

    def face_tracking(self, on):
        """Daemon-side face following, blended with our motion, while MIMI talks to the group."""
        if not C.FACE_TRACKING_WHILE_SPEAKING or self.dry_run or on == self.tracking:
            return
        try:
            if on:
                self.mini.start_head_tracking(weight=C.FACE_TRACKING_WEIGHT)
            else:
                self.mini.stop_head_tracking()
            self.tracking = on
        except Exception as error:
            print(f"  (face tracking unavailable: {error})")

    # named poses used by the FSM -------------------------------------------------

    def look_at_group(self):
        self.goto(yaw=0, pitch=C.LOOK_GROUP_PITCH, roll=0, body_yaw=0, antennas=(0, 0), duration=0.8)

    def look_at_seat(self, seat):
        self.goto(yaw=0, pitch=C.LOOK_GROUP_PITCH, roll=0, body_yaw=C.SEAT_YAW[seat], duration=0.8)

    def reading_look(self):
        self.set_mode("still")
        self.goto(yaw=0, pitch=C.READ_PITCH, roll=0, body_yaw=0, antennas=(0.3, 0.3), duration=0.9)

    def listening_tilt(self):
        self.goto(yaw=0, pitch=C.LISTEN_PITCH, roll=C.LISTEN_ROLL, body_yaw=0, antennas=(-0.2, 0.2), duration=0.9)
        self.set_mode("listen")

    def happy_wiggle(self):
        """Antenna wiggle on a match."""
        for a in [(0.7, -0.7), (-0.7, 0.7), (0.7, -0.7), (0.0, 0.0)]:
            self.goto(antennas=a, duration=0.18, method="cartoon")

    def curious_tilt(self):
        """Curious tilt on a mismatch: the build sounds different, nothing is 'wrong'."""
        self.goto(roll=-12, antennas=(0.9, -0.2), duration=0.6, method="cartoon")

    def droop(self):
        self.goto(roll=0, pitch=self.anchor.pitch + 5, antennas=(1.2, -1.2), duration=0.8)

    def bow(self):
        self.goto(pitch=25, antennas=(0.6, -0.6), duration=1.0, method="ease_in_out")
        self.goto(pitch=C.LOOK_GROUP_PITCH, antennas=(0, 0), duration=1.0, method="ease_in_out")

    def neutral(self):
        self.set_mode("still")
        self.goto(yaw=0, pitch=0, roll=0, body_yaw=0, antennas=(0, 0), duration=1.0)

    # ------------------------------------------------------------ reactive layer

    def _reactive_loop(self):
        beat_s = 60.0 / C.TEMPO_BPM
        while not self._stop:
            time.sleep(0.04)  # 25 Hz
            if self.mode == "still" or not self.lock.acquire(blocking=False):
                continue
            try:
                t = time.time()
                a = self.anchor
                pitch = a.pitch + C.BREATH_PITCH_DEG * math.sin(2 * math.pi * t / C.BREATH_PERIOD_S)
                yaw, antennas = a.yaw, a.antennas
                if self.mode == "bob":
                    phase = (t - self.bob_start) / beat_s
                    pitch = a.pitch + C.BOB_PITCH_DEG * (1 - abs(math.sin(math.pi * phase)))  # down on each beat
                    sway = 0.25 if int(phase) % 2 == 0 else -0.25
                    antennas = (sway, -sway)
                elif self.mode in ("idle", "listen") and self.doa.yaw is not None \
                        and t - self.doa.heard_at < C.DOA_HOLD_S:
                    yaw = a.yaw + C.DOA_TURN_GAIN * clamp(self.doa.yaw - a.body_yaw, C.YAW_LIMIT)
                pose = replace(a, yaw=yaw, pitch=pitch, antennas=antennas)
                self.mini.set_target(head=head_matrix(pose), antennas=list(antennas))
            except Exception:
                pass
            finally:
                self.lock.release()

    def stop(self):
        self._stop = True
        self.doa.enabled = False


# ---------------------------------------------------------------- speaker: notes and speech

class Speaker:
    def __init__(self, mini, dry_run=False):
        self.mini = mini
        self.dry_run = dry_run
        rate = -1 if dry_run else mini.media.get_output_audio_samplerate()
        self.sample_rate = rate if rate and rate > 0 else 16000
        if not dry_run:
            mini.media.start_playing()
        self._laptop_voice = None
        os.makedirs(C.TTS_CACHE_DIR, exist_ok=True)

    def play(self, audio):
        """Play float32 mono audio on the robot speaker and wait until it has finished."""
        seconds = len(audio) / self.sample_rate
        if self.dry_run:
            _laptop_play(audio, self.sample_rate)
        else:
            self.mini.media.push_audio_sample(audio)
        time.sleep(seconds)

    def say(self, text):
        print("MIMI>", text)
        path = cache_path(text)
        if not os.path.exists(path) and TTS_MODEL is not None:
            render_tts(text, path)
        if os.path.exists(path):
            if self.dry_run:
                _laptop_play_file(path)
            else:
                self.mini.media.play_sound(path)
            time.sleep(wav_seconds(path) + 0.2)  # play_sound returns when playback STARTS (Lab 3)
        else:
            voice = self.laptop_voice()
            if voice:
                voice(text)

    def laptop_voice(self):
        if self._laptop_voice is None:
            try:
                if sys.platform == "win32":
                    import comtypes.client
                    sapi = comtypes.client.CreateObject("SAPI.SpVoice")
                    self._laptop_voice = lambda text: sapi.Speak(text)
                else:
                    import pyttsx3
                    engine = pyttsx3.init()

                    def voice(text):
                        engine.say(text)
                        engine.runAndWait()
                    self._laptop_voice = voice
            except Exception as error:
                print(f"(no laptop voice: {error})")
                self._laptop_voice = False
        return self._laptop_voice


def cache_path(text):
    key = hashlib.sha1(f"{C.TTS_VOICE}|{text}".encode()).hexdigest()[:16]
    return os.path.join(C.TTS_CACHE_DIR, f"{key}.wav")


def render_tts(text, path):
    """OpenAI text-to-speech into the cache (pre-render scripted lines with prerender_tts.py)."""
    try:
        with client().audio.speech.with_streaming_response.create(
                model=TTS_MODEL, voice=C.TTS_VOICE, input=text, response_format="wav",
                instructions="Warm, playful orchestra conductor talking to 12-year-olds. Clear and not too fast.",
        ) as response:
            response.stream_to_file(path)
    except Exception as error:
        print(f"  (TTS failed, using laptop voice: {error})")
        if os.path.exists(path):
            os.remove(path)


def wav_seconds(path):
    """Length of a WAV from its file size: streamed WAVs often have a placeholder header length (Lab 3)."""
    with wave.open(path) as w:
        bytes_per_second = w.getframerate() * w.getnchannels() * w.getsampwidth()
    return max(0.0, (os.path.getsize(path) - 44) / bytes_per_second)


def write_wav(path, audio, sample_rate):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def _laptop_play(audio, sample_rate):
    path = os.path.abspath("last_notes.wav")
    write_wav(path, audio, sample_rate)
    _laptop_play_file(path)


def _laptop_play_file(path):
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)


# ---------------------------------------------------------------- dry run (no robot)

class _FakeMedia:
    """Frames for --dry-run / --sim: the laptop camera (--webcam), a photo (--plate-image), or none."""

    def __init__(self, plate_image=None, webcam=None):
        self.plate_image = plate_image
        self.webcam = webcam
        self._capture = None

    def get_frame(self):
        import cv2
        if self.webcam is not None:
            return webcam_frame(self)
        if self.plate_image:
            return cv2.imread(self.plate_image)
        return None


def open_webcam(index):
    """The laptop camera at full resolution (more pixels per stud)."""
    import cv2
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {index}. Try --webcam 1, or close other apps using the camera.")
    for _ in range(10):  # first frames are dark while the exposure settles
        capture.read()
    return capture


def webcam_frame(media):
    if media._capture is None:
        media._capture = open_webcam(media.webcam)
    media._capture.grab()  # drop the buffered frame so the image is current
    ok, frame = media._capture.read()
    return frame if ok else None


class SimMini:
    """--sim: the simulated robot's motion, with frames from --plate-image (the sim scene has no plate)."""

    def __init__(self, mini, plate_image=None, webcam=None):
        self._mini = mini
        self.media = _FakeMedia(plate_image, webcam)

    def __getattr__(self, name):  # motion and every other SDK call go to the simulator
        return getattr(self._mini, name)


class FakeMini:
    """Stands in for ReachyMini with --dry-run: no motion, frames from --plate-image, sound on the laptop."""

    def __init__(self, plate_image=None, webcam=None):
        self.media = _FakeMedia(plate_image, webcam)

    def goto_target(self, **kwargs):
        time.sleep(min(kwargs.get("duration", 0.5), 0.3))

    def set_target(self, **kwargs):
        pass

    def __getattr__(self, name):  # any other SDK call is a no-op
        return lambda *args, **kwargs: None
