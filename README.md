# ai-learning

My AI learning work, with one folder per topic.

| Folder | Topic |
|--------|-------|
| [local-llms](local-llms/) | Running Qwen 3.6 locally behind an OpenAI-compatible API |
| [rag](rag/) | RAG from scratch: chunk, embed, retrieve, generate |

## Running the code

All topics share one Python environment, defined in `pyproject.toml`. With [uv](https://docs.astral.sh/uv/) installed, run scripts from the repo root:

```bash
uv run local-llms/chat.py "Why is the sky blue?"
uv run local-llms/benchmark.py
uv run rag/rag.py "How many vacation days do new hires get?"
```

These need the local model server running. See [Step 3 of the setup notes](local-llms/qwen36-mac-mini-setup.md#step-3-start-the-server-and-smoke-test-it).

## Private files

This repo is public. Keep API keys in a `.env` file and private data in any folder named `private/`. Git ignores both. To search your own documents without publishing them, put them in `rag/docs/private/`, which `rag.py` still reads.
