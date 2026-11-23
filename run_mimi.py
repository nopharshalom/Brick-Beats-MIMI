"""Brick Beats with Maestro MIMI: run one 6-minute session.

    python run_mimi.py --session S01 --condition directed          # free Ollama models (Lab 3 setup)
    python run_mimi.py --session S02 --condition group --api       # OpenAI models + OpenAI voice
    python run_mimi.py --session T --condition group --dry-run     # no robot: type plate reads with 'c'
    python run_mimi.py --session T --condition directed --dry-run --plate-image test_plate.jpg
    python run_mimi.py --session T --condition directed --sim --plate-image test_plate.jpg   # simulator
    python run_mimi.py --session T --condition directed --sim --webcam    # simulator + laptop camera
    python run_mimi.py --session S03 --condition directed --start round2   # resume after a crash

Condition alternates by session (RQ1): odd sessions directed, even sessions group, for example.
Wizard keys are printed at start (see mimi/wizard.py).
"""

import argparse

from mimi.game import Game


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="session id for the log, e.g. S01")
    parser.add_argument("--condition", required=True, choices=["directed", "group"], help="RQ1 feedback style")
    parser.add_argument("--api", action="store_true", help="OpenAI models instead of Ollama (read by mimi/models.py)")
    parser.add_argument("--dry-run", action="store_true", help="no robot: motion is skipped, sound plays on the laptop")
    parser.add_argument("--sim", action="store_true",
                        help="MuJoCo simulator on this laptop (start `reachy-mini-daemon --sim --no-media` first)")
    parser.add_argument("--plate-image", help="with --dry-run or --sim: a photo used as every camera frame")
    parser.add_argument("--webcam", nargs="?", const=0, type=int, metavar="INDEX",
                        help="with --dry-run or --sim: use the laptop camera (default 0) as MIMI's camera")
    parser.add_argument("--seed", type=int, default=None, help="melody choice seed (default: random)")
    parser.add_argument("--start", default="greet", choices=["greet", "tutorial", "round1", "round2", "round3", "finale"])
    args = parser.parse_args()
    if (args.webcam is not None or args.plate_image) and not (args.dry_run or args.sim):
        parser.error("--webcam and --plate-image replace the robot camera: use them with --dry-run or --sim")
    Game(args).run()


if __name__ == "__main__":
    main()
