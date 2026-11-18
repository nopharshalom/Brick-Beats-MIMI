"""Wizard-of-Oz keyboard (proposal 4.3). Runs in its own thread; the FSM polls the queue.

Type a key and press Enter in the terminal:
    p   play request (same as the baton brick: use if the baton is not detected)
    c   correct a misread plate: then type the active measure's notes, e.g.  E1 D1 C2   (R = rest beat)
    x   mark the last plate read as WRONG (no correction) - for read accuracy
    r   repeat the last instruction
    n   next: skip ahead (end the tutorial step / round now, if time runs short)
    f   FAILSAFE: stop, neutral pose, wait
    g   go: leave FAILSAFE and continue
    q   quit (finale skipped, logs kept)
Every keypress is logged, so we can report how often autonomy needed help.
"""

import queue
import threading

HELP = "[wizard] p=play  c=correct  x=misread  r=repeat  n=next  f=FAILSAFE  g=go  q=quit"


class Wizard(threading.Thread):
    def __init__(self, log):
        super().__init__(daemon=True)
        self.log = log
        self.commands = queue.Queue()

    def run(self):
        print(HELP)
        while True:
            try:
                key = input().strip()
            except EOFError:
                return
            if not key:
                continue
            k, rest = key[0].lower(), key[1:].strip()
            if k == "c":
                notes = rest or input("[wizard] notes in the active measure (e.g. E1 D1 C2, R = rest): ").strip()
                self.log.event("wizard", key="c", notes=notes)
                self.commands.put(("c", notes))
            elif k == "x":  # only a log mark: the last read was wrong
                self.log.event("wizard", key="x", note="last plate read marked wrong")
                print("[wizard] last read marked wrong")
            elif k in "prnfgq":
                self.log.event("wizard", key=k)
                self.commands.put((k, ""))
            else:
                print(HELP)

    def poll(self, *wanted):
        """Return the first queued command (key, arg) whose key is in `wanted` (or any, if none given).
        'f' and 'q' are always returned so the FSM can react to them anywhere."""
        kept, found = [], None
        while True:
            try:
                item = self.commands.get_nowait()
            except queue.Empty:
                break
            if found is None and (not wanted or item[0] in wanted or item[0] in "fq"):
                found = item
            else:
                kept.append(item)
        for item in kept:
            self.commands.put(item)
        return found
