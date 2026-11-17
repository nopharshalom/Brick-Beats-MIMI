"""Per-session numbers from logs/*.csv for the evaluation (proposal 6.3).

    python analyze.py                  # all sessions -> printed table + session_summary.csv

Per session: condition, stars, win, playbacks-to-match per round, Wizard interventions (by key),
plate reads (and how many came from the VLM / the Wizard), reads the Wizard marked wrong,
median read latency, hint sources (LLM vs template), and participation balance as a Gini coefficient
of notes per seat in the final read of each round (the Wizard's hand tally is the ground truth;
this is the automatic estimate).
"""

import csv
import glob
import json
import statistics
from collections import Counter


def gini(values):
    values = sorted(values)
    n, total = len(values), sum(values)
    if n == 0 or total == 0:
        return 0.0
    return sum((2 * i - n - 1) * v for i, v in enumerate(values, 1)) / (n * total)


def summarize(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        return None
    d = lambda r: json.loads(r["detail"] or "{}")  # noqa: E731
    out = {"log": path, "session": rows[0]["session"], "condition": rows[0]["condition"]}
    results = [d(r) | {"round": r["round"]} for r in rows if r["event"] == "round_result"]
    out["playbacks_by_round"] = ";".join(f"r{x['round']}:{x['playbacks']}{'' if x['won'] else 'x'}" for x in results)
    game = [d(r) for r in rows if r["event"] == "game_result"]
    out["stars"] = game[-1]["stars"] if game else sum(1 for x in results if x["won"])
    out["won_game"] = game[-1]["won"] if game else ""
    wizard = Counter(d(r)["key"] for r in rows if r["event"] == "wizard")
    out["wizard_interventions"] = sum(v for k, v in wizard.items() if k != "x")
    out["wizard_keys"] = " ".join(f"{k}={v}" for k, v in sorted(wizard.items()))
    reads = [d(r) for r in rows if r["event"] == "read"]
    out["reads"] = len(reads)
    out["reads_vlm"] = sum(1 for r in reads if r.get("source") == "vlm")
    out["reads_wizard"] = sum(1 for r in reads if r.get("source") == "wizard")
    out["reads_marked_wrong"] = wizard.get("x", 0) + wizard.get("c", 0)
    out["read_accuracy_est"] = round(1 - out["reads_marked_wrong"] / len(reads), 3) if reads else ""
    lat = [r["latency_s"] for r in reads if r.get("source") != "wizard"]
    out["read_latency_median_s"] = round(statistics.median(lat), 2) if lat else ""
    hints = Counter(d(r)["source"].split(":")[0] for r in rows if r["event"] == "hint")
    out["hints_llm"], out["hints_template"] = hints.get("llm", 0), hints.get("template", 0)
    # last read in each round -> notes per seat
    per_seat = Counter()
    last_read = {}
    for r in rows:
        if r["event"] == "read" and r["round"] in ("1", "2", "3"):
            last_read[r["round"]] = d(r).get("seat_notes", {})
    for seats in last_read.values():
        per_seat.update({int(k): v for k, v in seats.items()})
    out["notes_per_seat"] = " ".join(f"s{s}={per_seat.get(s, 0)}" for s in range(1, 5))
    out["gini_participation_est"] = round(gini([per_seat.get(s, 0) for s in range(1, 5)]), 3)
    return out


def main():
    summaries = [s for s in (summarize(p) for p in sorted(glob.glob("logs/*.csv"))) if s]
    if not summaries:
        raise SystemExit("No logs in logs/.")
    with open("session_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for s in summaries:
        print(s)
    print("saved session_summary.csv")


if __name__ == "__main__":
    main()
