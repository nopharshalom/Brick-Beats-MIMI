"""Time-stamped CSV log (proposal 4.2, "Logging"): state transitions, plate reads, playbacks,
feedback/hints and Wizard keypresses, one row each. Rectified plate images go to captures/ so read
accuracy can be scored against Wizard ground truth afterwards (analyze.py)."""

import csv
import json
import os
import time
from datetime import datetime

import cv2

LOG_DIR = "logs"
FIELDS = ["time", "t_session_s", "session", "condition", "state", "round", "event", "detail"]


class SessionLog:
    def __init__(self, session, condition):
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs("captures", exist_ok=True)
        self.session, self.condition = session, condition
        self.path = os.path.join(LOG_DIR, f"{session}_{datetime.now():%Y%m%d-%H%M%S}.csv")
        self.t0 = time.time()
        self.state, self.round = "", 0
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def event(self, event, **detail):
        row = {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "t_session_s": round(time.time() - self.t0, 2),
            "session": self.session, "condition": self.condition,
            "state": self.state, "round": self.round, "event": event,
            "detail": json.dumps(detail, default=str),
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

    def transition(self, state):
        print(f"\n== {state} (round {self.round}) ==")
        self.state = state
        self.event("state")

    def save_image(self, image, tag):
        if image is None:
            return ""
        path = os.path.join("captures", f"{self.session}_r{self.round}_{tag}_{datetime.now():%H%M%S%f}.jpg")
        cv2.imwrite(path, image)
        return path
