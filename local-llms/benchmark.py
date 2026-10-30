"""Measure the local model's time to first token and generation speed.

Time to first token matters most for RAG, because retrieved context makes prompts long.

Usage: uv run benchmark.py
"""

import time
import uuid

from openai import OpenAI

MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
PROMPT_WORDS = [50, 500, 2000, 8000]
MAX_OUTPUT_TOKENS = 200

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")


def measure(words):
    # A unique start stops the server from answering out of its prompt cache.
    filler = " ".join(["The robot checked the sensor log."] * (words // 6))
    prompt = f"{uuid.uuid4()}\n{filler}\n\nWrite a long story about a robot."

    start = time.perf_counter()
    first_token = None
    usage = None
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if first_token is None and chunk.choices and chunk.choices[0].delta.content:
            first_token = time.perf_counter()
        if chunk.usage:
            usage = chunk.usage
    end = time.perf_counter()

    return (
        usage.prompt_tokens,
        first_token - start,
        usage.completion_tokens / (end - first_token),
    )


if __name__ == "__main__":
    print("prompt tokens | first token (s) | output tokens/s")
    for words in PROMPT_WORDS:
        prompt_tokens, ttft, speed = measure(words)
        print(f"{prompt_tokens:13} | {ttft:15.2f} | {speed:15.1f}")
