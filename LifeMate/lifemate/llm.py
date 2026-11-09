"""Chat model behind an OpenAI-compatible endpoint (llama.cpp server, vLLM, Ollama ...). Standard library only."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request


class LLM:
    def __init__(self, base_url: str, model: str = "default", api_key: str = "", temperature: float = 0.0,
                 max_tokens: int = 512, timeout: int = 1800, extra: dict | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra = extra or {}
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        self._lock = threading.Lock()  # the counters are shared when samples run in parallel

    def chat(self, messages: list[dict]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        body.update(self.extra)
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        t0 = time.time()
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:  # a 4xx will not get better by retrying
                detail = e.read().decode("utf-8", "replace")[:500]
                if e.code < 500:
                    raise RuntimeError(f"model server refused the request ({e.code}): {detail}") from e
                last = RuntimeError(f"model server error {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"model server unreachable at {self.url}: {last}")
        usage = data.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.seconds += time.time() - t0
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"].get("content") or ""


_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_step(text: str) -> dict | None:
    """The first JSON object of a model answer that is a protocol step ({"call": ...} or {"reply": ...}); None if there is none."""
    text = _THINK.sub("", text or "")
    if "</think>" in text:  # an unclosed opening tag was cut off by the template
        text = text.split("</think>", 1)[1]
    dec = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and ("call" in obj or "reply" in obj):
            return obj
        i = text.find("{", i + 1)
    return None
