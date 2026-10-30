# ai-learning

My AI learning work, with one folder per topic.

| Folder | Topic |
|--------|-------|
| [local-llms](local-llms/) | Running Qwen 3.6 locally behind an OpenAI-compatible API |
| [rag](rag/) | RAG from scratch: chunk, embed, retrieve, generate |

## Running the code

Each folder is self-contained, with its own dependencies and instructions, and doesn't rely on any other folder. With [uv](https://docs.astral.sh/uv/) installed, run scripts from inside a folder:

```bash
cd rag
uv run rag.py "How many vacation days do new hires get?"
```

## Private files

This repo is public. Keep API keys in a `.env` file and private data in any folder named `private/`. Git ignores both.
