"""Melodies, the comparator (deliberative layer) and the note synthesizer.

A melody is a list of Note(lane, start, beats). `start` counts beats from the start of the plate
(measure m, beat b -> start = 4*m + b), so a whole-plate read and a one-measure target use the same type.
"""

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass(frozen=True, order=True)
class Note:
    start: int  # beat index from the left edge of the grid
    lane: int   # 0 = low C ... 7 = high C
    beats: int  # 1 quarter, 2 half, 4 whole

    def __str__(self):
        return f"{C.PITCHES[self.lane]}@{self.start}x{self.beats}"


def melody(notes, measure=0):
    """Notes from config tuples (lane, beat_in_measure, beats), placed in a measure."""
    offset = measure * C.BEATS_PER_MEASURE
    return sorted(Note(offset + beat, lane, beats) for lane, beat, beats in notes)


def in_measure(notes, measure):
    lo = measure * C.BEATS_PER_MEASURE
    return sorted(n for n in notes if lo <= n.start < lo + C.BEATS_PER_MEASURE)


def to_text(notes):
    return " ".join(str(n) for n in sorted(notes)) or "(empty)"


def parse_wizard_notes(text, measure):
    """Wizard correction format, notes in order across the measure: 'E1 D1 C2', 'R' = rest beat.
    'C' = low C, 'C5' or 'HC' = high C. Returns a list of Notes, or None if it does not parse."""
    names = {"C": 0, "C4": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6, "C5": 7, "HC": 7}
    notes, beat = [], 0
    for token in text.upper().split():
        if token.startswith("R"):
            beat += int(token[1:] or 1)
            continue
        pitch, digits = token.rstrip("0123456789"), token[len(token.rstrip("0123456789")):]
        if pitch == "C" and digits == "5":  # "C5" alone = high C quarter
            pitch, digits = "C5", ""
        if pitch not in names:
            return None
        beats = int(digits or 1)
        notes.append(Note(measure * C.BEATS_PER_MEASURE + beat, names[pitch], beats))
        beat += beats
    return notes if beat <= C.BEATS_PER_MEASURE else None


# ---------------------------------------------------------------- comparator

@dataclass
class Difference:
    beat: int            # 1-4 inside the measure: the first beat that sounds different
    kind: str            # "missing", "extra", "pitch", "duration"
    target: Note | None  # the note that should sound at that beat
    built: Note | None   # what was built there

    @property
    def seat(self):
        return self.beat  # seat i owns beat i of the active measure (proposal 3.2)

    def direction(self):
        """'higher' / 'lower' / '' : how the TARGET compares with what was built."""
        if self.kind == "pitch" and self.target and self.built:
            return "higher" if self.target.lane > self.built.lane else "lower"
        return ""

    def length(self):
        """'longer' / 'shorter' / '' : how the TARGET's duration compares with what was built."""
        if self.kind == "duration" and self.target and self.built:
            return "longer" if self.target.beats > self.built.beats else "shorter"
        return ""


def _sounding(notes, beat):
    """(note that starts on this beat, note sounding on this beat) for a monophonic line."""
    onset = next((n for n in notes if n.start == beat), None)
    held = next((n for n in notes if n.start <= beat < n.start + n.beats), None)
    return onset, held


def compare(target, built, measure):
    """None if the build matches the target in this measure, else the first Difference."""
    target, built = in_measure(target, measure), in_measure(built, measure)
    if target == built:
        return None
    offset = measure * C.BEATS_PER_MEASURE
    for b in range(offset, offset + C.BEATS_PER_MEASURE):
        t_on, t_held = _sounding(target, b)
        b_on, b_held = _sounding(built, b)
        if t_on == b_on and (t_held == b_held):
            continue
        beat = b - offset + 1
        if t_held and not b_held:
            return Difference(beat, "missing", t_held, None)
        if b_held and not t_held:
            return Difference(beat, "extra", None, b_held)
        if t_held and b_held and t_held.lane != b_held.lane:
            return Difference(beat, "pitch", t_held, b_held)
        if t_held and b_held:
            return Difference(beat, "duration", t_held, b_held)
    # Same per-beat sound but different note lists (e.g. two notes stacked on one beat): first extra one
    extra = sorted(set(built) - set(target)) or sorted(set(target) - set(built))
    n = extra[0]
    return Difference(n.start - offset + 1, "extra", None, n)


# ---------------------------------------------------------------- synthesizer

def frequency(lane):
    semitones_from_c4 = [0, 2, 4, 5, 7, 9, 11, 12][lane]
    return 261.63 * 2 ** (semitones_from_c4 / 12)


def synthesize(notes, sample_rate, tempo=C.TEMPO_BPM, first_beat=None, total_beats=None):
    """Mono float32 audio for the notes: soft marimba-like tone, 90 BPM by default.
    first_beat: beat that maps to t=0 (default: the first note's measure start)."""
    if first_beat is None:
        first_beat = min((n.start for n in notes), default=0)
        first_beat -= first_beat % C.BEATS_PER_MEASURE
    if total_beats is None:
        last = max((n.start + n.beats for n in notes), default=first_beat + C.BEATS_PER_MEASURE)
        total_beats = max(C.BEATS_PER_MEASURE, last - first_beat)
    beat_s = 60.0 / tempo
    out = np.zeros(int((total_beats * beat_s + 0.4) * sample_rate), dtype=np.float32)
    for n in notes:
        dur = n.beats * beat_s * 0.92  # small gap so repeated notes are heard as separate
        t = np.arange(int(dur * sample_rate)) / sample_rate
        f = frequency(n.lane)
        tone = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
        attack = np.minimum(1.0, t / 0.015)
        release = np.minimum(1.0, (dur - t) / 0.08)
        decay = np.exp(-t * 1.2)
        wave = (0.28 * tone * attack * release * decay).astype(np.float32)
        i = int((n.start - first_beat) * beat_s * sample_rate)
        if i < 0:
            continue
        out[i:i + len(wave)] += wave[: len(out) - i]
    return np.clip(out, -1.0, 1.0)
