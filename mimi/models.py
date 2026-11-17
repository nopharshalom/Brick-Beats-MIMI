"""The LLM and VLM, set up exactly like Lab 3 (Lab-3-GenAI-for-HRI/common.py):

    python run_mimi.py         free open-source models on the laptop with Ollama (llama3.2 + gemma3:4b)
    python run_mimi.py --api   OpenAI models (OPENAI_API_KEY in .env), plus OpenAI text-to-speech

In Brick Beats the models never run the game. Perception and feedback choice are deterministic;
the models only (1) give a second opinion on a low-confidence plate read (VLM) and
(2) phrase a short hint (LLM), with a template fallback.
"""

import base64
import json
import re
import sys

import cv2

USE_API = "--api" in sys.argv

_client = None


def client():
    """OpenAI-compatible client, created on first use (so --dry-run needs no model server)."""
    global _client
    if _client is None:
        from openai import OpenAI
        if USE_API:
            from dotenv import load_dotenv
            load_dotenv()  # reads OPENAI_API_KEY from .env
            _client = OpenAI(max_retries=1)
        else:
            _client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama", max_retries=1)
    return _client


if USE_API:
    LLM_MODEL = "gpt-6.1-sol"
    VLM_MODEL = "gpt-6.1-sol"
    TTS_MODEL = "gpt-4o-mini-tts"
else:
    LLM_MODEL = "llama3.2"
    VLM_MODEL = "gemma3:4b"
    TTS_MODEL = None  # the free models have no voice: laptop voice instead


def ask(model, messages, timeout=None):
    """Send a chat request and return the model's text reply."""
    kwargs = {"model": model, "messages": messages}
    if timeout is not None:
        kwargs["timeout"] = timeout
    response = client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def images_content(frames, text):
    """Message content with several images (BGR) plus a text prompt, images in order."""
    content = [{"type": "text", "text": text}]
    for frame in frames:
        h, w = frame.shape[:2]
        if w > 640:  # smaller image = faster and cheaper
            frame = cv2.resize(frame, (640, h * 640 // w))
        _, jpg = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(jpg.tobytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def parse_json(reply):
    """The first {...} object in a model reply, or {} if there is none."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}
