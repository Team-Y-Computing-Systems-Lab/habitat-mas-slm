"""SLM client for any OpenAI-compatible server.

Ollama (default, http://localhost:11434/v1), llama.cpp's `llama-server`, and
vLLM all speak this API, so moving from the workstation to a Jetson is a URL
change. Every call records tokens and latency for the results.
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

from openai import OpenAI

DEFAULT_URL = "http://localhost:11434/v1"


@dataclass
class SLMCall:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    output: str

    def to_dict(self):
        return asdict(self)


class SLMClient:
    def __init__(self, model: str, base_url: str = DEFAULT_URL, temperature: float = 0.0,
                 max_tokens: int = 2048, timeout: float = 600.0):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key="unused", timeout=timeout)
        self.calls: List[SLMCall] = []

    def chat(self, system: str, user: str, json_mode: bool = True) -> SLMCall:
        t0 = time.time()
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"} if json_mode else None,
        )
        call = SLMCall(
            model=self.model,
            prompt_tokens=r.usage.prompt_tokens if r.usage else 0,
            completion_tokens=r.usage.completion_tokens if r.usage else 0,
            latency_s=time.time() - t0,
            output=r.choices[0].message.content or "",
        )
        self.calls.append(call)
        return call


def parse_json(text: str) -> Optional[dict]:
    """First JSON object in `text` (tolerates <think> blocks and code fences)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None
