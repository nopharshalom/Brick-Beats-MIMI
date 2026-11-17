"""Pre-render every scripted line to tts_cache/ with OpenAI text-to-speech (proposal 4.2:
"Scripted lines are pre-rendered TTS clips"). Run once with an API key; afterwards the game plays the
cached clips on the robot even with the free Ollama models, and only runtime LLM hints need live TTS.

    python tools/prerender_tts.py --api
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimi.dialogue import all_scripted_lines  # noqa: E402
from mimi.models import TTS_MODEL  # noqa: E402
from mimi.robot import cache_path, render_tts  # noqa: E402


def main():
    if TTS_MODEL is None:
        raise SystemExit("Run with --api (OpenAI TTS needs OPENAI_API_KEY in .env).")
    lines = all_scripted_lines()
    for i, text in enumerate(lines, 1):
        path = cache_path(text)
        if os.path.exists(path):
            continue
        print(f"[{i}/{len(lines)}] {text}")
        render_tts(text, path)
    print(f"{len(lines)} lines in tts_cache/")


if __name__ == "__main__":
    main()
