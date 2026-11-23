"""Dialogue manager: scripted lines + constrained LLM hints (proposal 4.2).

Every line MIMI says in the game is scripted below (pre-rendered with prerender_tts.py), EXCEPT the
one short hint after a mismatch. That hint comes from the LLM under a child-appropriate system prompt
(<= 15 words, grade-6 words, about the build never the child); if the reply is late (> 2 s), too long,
or breaks a rule, MIMI says a template hint instead. Every hint is logged with its source.

Conditions (RQ1):
    directed  MIMI turns to the seat that owns the first different beat and names that beat
    group     MIMI faces the group: "something sounds different"; the hint never names a beat or seat
"""

import concurrent.futures as futures
import re
import time

from . import config as C
from .models import LLM_MODEL, ask
from .music import Difference

# ---------------------------------------------------------------- scripted lines

SCRIPT = {
    "greet": "Hello, orchestra! I am Maestro MIMI. Welcome to my concert hall!",
    "intro_1": "This plate is our score. Each row is a note. Up is higher!",
    "intro_2": "A longer brick is a longer note. Short brick, one beat. Long brick, four beats.",
    "intro_3": "Each seat owns one beat. Seat one has beat one, and so on.",
    "tutorial_start": "Let's practice under the Tutorial sign, one brick at a time.",
    "tutorial_done": "Bravo! Listen to what you built.",
    "round_intro": "Round {n}! Listen to my melody. I will play it two times.",
    "round_again": "One more time!",
    "round_build": "Now build it under the Round {n} sign! Conductor, place the baton when ready.",
    "wrong_section": "I see bricks in another spot. Round {n} goes under the Round {n} sign.",
    "conductor": "Seat {seat}, you are the conductor this round.",
    "reading": "Let me read the score.",
    "hands": "Hands off the score, please, so I can read it.",
    "cannot_read": "I cannot read the score. Let me look again.",
    "playing_build": "Here is what you built!",
    "match": "That is my melody! You earned a star!",
    "mismatch_directed": "Seat {seat}, beat {beat} sounds different. Listen!",
    "mismatch_group": "Hmm, something sounds different. Listen!",
    "replay_target": "Here is my melody again.",
    "playbacks_left": "You can ask me {left} more time{s}.",
    "time_warning": "Twenty seconds left!",
    "lost": "Good try! Let me show you my melody.",
    "lost_answer": "Beat {beat} was a {color} {duration} note.",
    "lost_rest": "Beat {beat} should be quiet.",
    "finale_intro": "Time for the finale! Here is your symphony.",
    "win": "{stars} stars! You are Orchestra Graduates!",
    "lose": "{stars} star{s}. Great playing, orchestra! Practice makes music.",
    "goodbye": "Thank you, orchestra! Take a bow!",
    "failsafe": "Let's take a short break. My helper will check on me.",
    "resume": "I'm back! Let's keep playing.",
}

TUTORIAL_STEPS = [  # one instruction at a time (proposal 1.2, "instructions come one at a time")
    "Seat one: a short {color} brick on beat one, in the {pitch} row.",
    "Seat two: a short {color} brick on beat two, in the {pitch} row.",
    "Seats three and four: one {color} half note, in the {pitch} row.",
]


def line(key, **kwargs):
    return SCRIPT[key].format(**kwargs)


LENGTH_WORDS = {1: "", 2: "long ", 3: "longer ", 4: "very long "}  # brick length, said before the color


def color_hint(target, round_number):
    """The fading hint for a round (config COLOR_HINTS): every brick's color, the first one, or nothing."""
    level = C.COLOR_HINTS.get(round_number, "none")
    notes = sorted(target)
    if level == "all" and notes:
        bricks = [LENGTH_WORDS.get(n.beats, "") + C.LANE_COLORS[n.lane] for n in notes]
        return "The bricks are " + ", ".join(bricks[:-1]) + (" and " if len(bricks) > 1 else "") + bricks[-1] + "."
    if level == "first" and notes:
        first = notes[0]
        return f"It starts with a {LENGTH_WORDS.get(first.beats, '')}{C.LANE_COLORS[first.lane]} brick."
    return None


def all_scripted_lines():
    """Every fixed sentence MIMI can say, filled in for all seats/beats/rounds (for prerender_tts.py)."""
    from .music import melody
    lines = [v for v in SCRIPT.values() if "{" not in v]
    lines += [line(key, n=n) for key in ("round_intro", "round_build", "wrong_section") for n in (1, 2, 3)]
    lines += [line("conductor", seat=s) for s in (1, 2, 3, 4)]
    lines += [line("mismatch_directed", seat=b, beat=b) for b in (1, 2, 3, 4)]
    lines += [line("playbacks_left", left=k, s="" if k == 1 else "s") for k in (1, 2)]
    lines += [line("win", stars=s) for s in (2, 3)]
    lines += [line("lose", stars=s, s="" if s == 1 else "s") for s in (0, 1)]
    for b in (1, 2, 3, 4):
        lines.append(line("lost_rest", beat=b))
        for lane in range(len(C.PITCHES)):
            for d in C.DURATION_WORDS.values():
                lines.append(line("lost_answer", beat=b, color=C.LANE_COLORS[lane], duration=d))
    for i, note in enumerate(melody(C.TUTORIAL_MELODY)):
        lines.append(TUTORIAL_STEPS[i].format(color=C.LANE_COLORS[note.lane], pitch=C.PITCH_WORDS[note.lane]))
    lines += template_hints_all()
    for round_number, bank in C.MELODY_BANK.items():
        for tune in bank:
            hint = color_hint(melody(tune, round_number), round_number)
            if hint:
                lines.append(hint)
    return sorted(set(lines))


# ---------------------------------------------------------------- hints

SYSTEM_PROMPT = (
    "You are Maestro MIMI, a friendly robot orchestra conductor helping a group of 11 to 14 year olds "
    "rebuild a 4-beat melody with LEGO bricks. Rows are pitches (up is higher) and brick length is "
    "duration (short = 1 beat, half = 2 beats, whole = 4 beats).\n"
    "Write ONE hint sentence. Rules:\n"
    f"- at most {C.MAX_HINT_WORDS} words, simple words a 6th grader knows\n"
    "- talk about the music or the bricks, never about a person; never say wrong, mistake, bad or fail\n"
    "- help them listen and think; do not give the exact answer (no exact color, row name or brick)\n"
    "- address the group as 'orchestra' if you address anyone\n"
    "- no emojis, no quotes, no lists. Reply with the sentence only."
)
BANNED = re.compile(r"\b(wrong|mistake|bad|fail\w*|stupid|incorrect|you messed)\b", re.IGNORECASE)


def describe(diff: Difference, condition):
    """Deterministic facts for the LLM. In the GROUP condition the beat is never revealed."""
    where = f"On beat {diff.beat}, " if condition == "directed" else "Somewhere in the measure, "
    if diff.kind == "missing":
        return where + "a note is missing: the plate is silent where the melody plays a note."
    if diff.kind == "extra":
        return where + "the plate plays a note where the melody is silent."
    if diff.kind == "pitch":
        return where + f"the melody's note is {diff.direction()} than the brick on the plate."
    return where + f"the melody's note is {diff.length()} than the brick on the plate."


TEMPLATE_HINTS = {
    "missing": "Listen for a note that is missing. Is there a silent beat?",
    "extra": "Listen closely. Is there a note that should be quiet?",
    "higher": "Listen! One note should sound higher. Try a row up.",
    "lower": "Listen! One note should sound lower. Try a row down.",
    "longer": "One note should ring longer. Try a longer brick.",
    "shorter": "One note should be shorter. Try a shorter brick.",
}


def template_hint(diff: Difference):
    return TEMPLATE_HINTS.get(diff.direction() or diff.length() or diff.kind, TEMPLATE_HINTS["missing"])


def template_hints_all():
    return list(TEMPLATE_HINTS.values())


def acceptable(text):
    words = text.split()
    return 2 <= len(words) <= C.MAX_HINT_WORDS and not BANNED.search(text) and "\n" not in text.strip()


_pool = futures.ThreadPoolExecutor(max_workers=2)


class PendingHint:
    """Start the LLM call early (while the build plays back) and collect it after the playback.
    The hint is used only if the LLM answered within LLM_TIMEOUT_S of the request (proposal 4.2);
    MIMI never waits longer than that for it."""

    def __init__(self, diff: Difference, condition):
        self.diff, self.started, self.finished = diff, time.time(), None
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": describe(diff, condition) + " Write the hint."}]
        self.future = _pool.submit(self._ask, messages)

    def _ask(self, messages):
        try:
            return ask(LLM_MODEL, messages, C.LLM_TIMEOUT_S + 3)
        finally:
            self.finished = time.time()

    def result(self):
        """(hint text, source, raw LLM reply, LLM latency in s or None if it never answered)."""
        remaining = max(0.0, C.LLM_TIMEOUT_S - (time.time() - self.started))
        raw, source = "", "template:timeout"
        try:
            raw = self.future.result(timeout=remaining)
            text = re.sub(r"\s+", " ", raw.strip().strip('"').strip())
            if self.finished - self.started > C.LLM_TIMEOUT_S:
                source = "template:timeout"
            elif acceptable(text):
                return text, "llm", raw, round(self.finished - self.started, 2)
            else:
                source = "template:rejected"
        except futures.TimeoutError:
            pass
        except Exception as error:
            source = f"template:error:{type(error).__name__}"
        latency = round(self.finished - self.started, 2) if self.finished else None
        return template_hint(self.diff), source, raw, latency
