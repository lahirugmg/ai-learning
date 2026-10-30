"""Call the local model through its OpenAI-compatible API.

Usage: uv run local-llms/chat.py "Why is the sky blue?"
"""

import sys

from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="mlx-community/Qwen3.6-35B-A3B-4bit",
    messages=[{"role": "user", "content": sys.argv[1]}],
    max_tokens=1000,
    stream=True,
    # Remove this line to let the model think first: slower, but better on hard questions.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
