"""Minimal Ollama client + shared helpers for the inference-time experiments.

Ollama is an *inference-only* engine: it cannot compute gradients or fine-tune,
so it is used here for the paper's reward-based / inference-time objectives
(verifiable rewards, best-of-N, self-rewarding, CoT distillation) -- none of
which require backpropagation.

Only the Python standard library is used (urllib), so there is no extra
dependency beyond a running Ollama server (`brew services start ollama`).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: float = 300.0) -> dict:
    url = f"{OLLAMA_HOST}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:  # connection refused, etc.
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            f"Is the server running? (`brew services start ollama`). Detail: {exc}"
        ) from exc


def server_is_up() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/version")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def model_available(model: str) -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        names = {m.get("name", "") for m in tags.get("models", [])}
        # Accept exact match or the base name (model:latest vs model).
        return model in names or any(n.split(":")[0] == model.split(":")[0]
                                     for n in names)
    except Exception:
        return False


def require_ready(model: str) -> None:
    """Fail early with an actionable message if the server/model is missing."""
    if not server_is_up():
        raise OllamaError(
            f"Ollama server is not responding at {OLLAMA_HOST}.\n"
            f"Start it with:  brew services start ollama")
    if not model_available(model):
        raise OllamaError(
            f"Model '{model}' is not pulled yet.\n"
            f"Pull it with:  ollama pull {model}")


def generate(prompt: str, *, model: str = DEFAULT_MODEL, temperature: float = 0.7,
             system: str | None = None, num_predict: int = 512,
             seed: int | None = None) -> str:
    """Single completion via /api/generate (non-streaming)."""
    options: dict = {"temperature": temperature, "num_predict": num_predict}
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    if system:
        payload["system"] = system
    out = _post("/api/generate", payload)
    return out.get("response", "")


# --------------------------------------------------------------------------- #
# Answer extraction + verifiable-reward helpers (used by the math experiment). #
# --------------------------------------------------------------------------- #

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> float | None:
    """Extract a numeric final answer, preferring an explicit marker.

    Looks for '#### <n>' (GSM8K style) or 'answer is <n>' first; otherwise
    falls back to the last number in the text.
    """
    if not text:
        return None
    marker = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if marker:
        return _to_float(marker.group(1))
    phrase = re.search(r"(?:answer|result)\s*(?:is|=|:)?\s*\$?\s*"
                       r"(-?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if phrase:
        return _to_float(phrase.group(1))
    nums = _NUM_RE.findall(text)
    return _to_float(nums[-1]) if nums else None


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def is_correct(pred: float | None, gold: float, *, tol: float = 1e-4) -> bool:
    return pred is not None and abs(pred - gold) <= tol


def majority_vote(preds: list[float | None]) -> float | None:
    """Self-consistency: most common non-null numeric answer."""
    valid = [p for p in preds if p is not None]
    if not valid:
        return None
    # Round to avoid float-key fragmentation; keep integers exact.
    keyed = [round(p, 4) for p in valid]
    return Counter(keyed).most_common(1)[0][0]


@dataclass
class MathProblem:
    question: str
    answer: float
