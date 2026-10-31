# RAG from scratch

Ask questions about your own documents using a local LLM. This folder builds retrieval-augmented generation (RAG) in about 70 lines of Python with no framework, so you can see and change every step.

## What RAG is

An LLM only knows what it learned in training, so it can't answer questions about your documents. RAG works around that in two stages: find the passages most relevant to the question, then put them in the prompt and ask the LLM to answer from them.

## How it works

![Flow diagram in two stages. Preparing the documents: documents are split into chunks, and each chunk is embedded. Answering a question: the question is embedded, then Retrieve compares it with the chunk embeddings and keeps the best 3 chunks. Those chunks and the question go into a prompt, the LLM reads the prompt, and the answer is printed.](diagrams/how-rag-works.excalidraw.svg)

| Step | What happens | In `rag.py` |
|------|--------------|-------------|
| Chunk | Splits each document into paragraphs | `load_chunks()` |
| Embed | Turns each chunk into 384 numbers (an embedding) that represent its meaning, so similar text gets similar numbers | `embedder.passage_embed()` |
| Retrieve | Embeds the question, scores every chunk by cosine similarity (higher means closer in meaning), and keeps the best 3 | `retrieve()` |
| Generate | Sends those chunks and the question to the LLM | `generate()` |

It uses two separate models. A small embedding model, `bge-small-en-v1.5`, finds the passages, and a large LLM writes the answer.

## Architecture

The flow above shows the steps. This diagram shows what runs where:

![Architecture diagram of one Mac. The terminal sends a question to the Retriever in the rag.py process and receives the answer from the Generator. Inside rag.py, the Document loader passes chunks to the Embedding model, which stores chunk vectors in the in-memory Vector store. The Retriever uses the Embedding model and the Vector store, then passes 3 chunks to the Generator. The Generator calls the LLM in the separate mlx_lm.server process over HTTP on port 8080. On local disk are the embedding weights, the documents, and the LLM weights. Hugging Face Hub supplies both sets of weights on first use only.](diagrams/rag-architecture.excalidraw.svg)

- **Two processes:** `rag.py` starts for each question and then exits, while the LLM server stays up. They talk over HTTP using the OpenAI API format, which is why you can swap the LLM by changing `LLM_URL` and `LLM_MODEL`.
- **The embedding model runs inside `rag.py`:** it runs on the CPU through FastEmbed and ONNX Runtime, so it needs no server of its own.
- **The vector store is just a NumPy array in memory:** for the sample documents, that's 12 chunks of 384 numbers, or 18 KB. It disappears when the script exits, so every run rebuilds it.
- **Your data stays on your machine:** a normal run connects only to the LLM server at `127.0.0.1:8080`. Hugging Face is contacted only to download models the first time. If you point `LLM_URL` at a remote server, your question and the retrieved chunks go to that server.

Both diagrams are [Excalidraw](https://excalidraw.com) drawings saved as images. To change one, open the file with the [Excalidraw extension](https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor) for VS Code, which saves back to the same file. You can also drag it onto excalidraw.com, then export an SVG with **Embed scene** turned on.

## Setup

You need [uv](https://docs.astral.sh/uv/) and an LLM server. On Apple Silicon, start one in its own terminal:

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
```

The first start downloads about 19 GB. The model used 19–22 GB of memory on a 32 GB Mac. To check that it's ready, run `curl http://127.0.0.1:8080/health`, which should return `{"status": "ok"}`.

Any OpenAI-compatible chat server also works. Change `LLM_URL` and `LLM_MODEL` at the top of `rag.py`, and delete the `extra_body` line if your server rejects it.

## Run it

From this folder:

```bash
uv run rag.py "How many vacation days do new hires get?"
```

The first run installs the packages and downloads the embedding model (64 MB) to `~/.cache/fastembed`. You'll see the chunks it found, with their scores, and then the answer:

```
Retrieved:
  0.808  time-off.md: New hires receive 23 vacation days per year, increasing to 2...
  0.703  time-off.md: Unused vacation days roll over until March 31 of the followi...
  0.622  expenses.md: Employees get a one-time home office allowance of $475, whic...

Answer:
New hires receive 23 vacation days per year [time-off.md].
```

## How to test it

The files in `docs/` describe a made-up company, Acme Robotics, so correct answers have to come from those files. Try these questions:

| Question | Expected answer | What it checks |
|----------|-----------------|----------------|
| How many vacation days do new hires get? | 23, from `time-off.md` | The basic flow |
| What's the daily food budget when I'm on a work trip? | $62, from `expenses.md` | Matching by meaning: the document says "meals during business travel" |
| What extension do I call if my computer gets stolen? | 4417, from `security.md` | Matching by meaning: the document says "laptops" |
| I lost my laptop. Who do I call, and how much is the home office allowance? | 4417 and $475, from two files | Combining sources |
| What is the parental leave policy? | Says it doesn't know | Declining when the documents don't cover it |
| Do we get free lunch? | Should say it doesn't know, but says no | A known failure: reading too much into related text about travel meals |

Check the facts rather than the exact wording. The wording changes between runs, and the model sometimes leaves out the citation.

**When an answer is wrong, the `Retrieved` list shows which half failed:**
- If the right chunk isn't listed, retrieval failed. Look at chunking, the embedding model, or `TOP_K`.
- If the right chunk is listed but the answer is still wrong, generation failed. Look at the prompt or the LLM.

## Learn by experimenting

Every experiment runs through `rag.py` and changes one thing. Edit the file, run it, then undo your edit with `git restore rag.py`. The results below come from one to three runs on the sample documents, so yours may differ a little.

### 1. See what RAG adds

Compare the model with and without your documents, while pushing it to always answer. In `generate()`, change the prompt to:

```python
system = (
    "You are the HR assistant for Acme Robotics. Always give a confident, specific answer "
    "with exact numbers. Never say you do not know.\n\n"
    f"Context:\n{context}"
)
```

Then run each question below twice: once with `TOP_K = 0`, so no chunks reach the LLM and the `Retrieved` list is empty, and once with `TOP_K = 3`.

```bash
uv run rag.py "How many vacation days do new hires get?"
```

**Result:**

| Question | `TOP_K = 0` (no documents) | `TOP_K = 3` (with documents) |
|----------|----------------------------|------------------------------|
| How many vacation days do new hires get? | Invented: usually 20 days, but 15 in one run | Correct: 23 days in every run |
| What is the parental leave policy? | Invented: 16 weeks of fully paid leave | Says it has no information |
| Does the company pay for a gym membership? | Invented: yes, up to $75 per month | Wrongly says no, though the documents never mention it |

The documents are the only thing that changed, and the model sounds just as sure in both columns. A confident answer tells you nothing about whether it's true. The last row also shows that documents help but don't solve everything, which experiment 4 explores.

### 2. Change how many chunks are retrieved

Set `TOP_K = 1`, then ask "How many vacation days do new hires get, and how many can roll over?"

**Result:** it answers 23 days but says the context doesn't cover rollover, because that fact is in a different chunk.

Now ask "Do I get any extra days off each month?" The wrong paragraph ranks first, so `TOP_K = 1` answers "I don't know." With `TOP_K = 3`, the right paragraph comes second and the answer is correct.

### 3. Add a relevance cutoff

`rag.py` always sends 3 chunks, even when none of them are relevant. Top scores across 16 test questions:

| Questions | Lowest top score | Highest top score |
|-----------|------------------|-------------------|
| 8 the documents can answer | 0.626 | 0.808 |
| 8 they can't answer | 0.383 | 0.593 |

Add this after the `retrieve()` call:

```python
hits = [hit for hit in hits if hit[1] >= 0.6]
if not hits:
    sys.exit("No relevant documents found.")
```

**Result:** the parental leave question now stops without calling the LLM. The gap is narrow, though, and an unrelated chunk scoring 0.622 still gets through on the vacation question.

### 4. Change the prompt

The prompt decides what happens when the documents don't have the answer. Keep `TOP_K = 3` and compare three versions of the instructions in `generate()`: the prompt as written, the prompt without the sentence "If the context does not contain the answer, say you don't know.", and the pressure prompt from experiment 1.

**Result:**

| Question | As written | Without that sentence | Pressure prompt |
|----------|------------|-----------------------|-----------------|
| What is the company's stock price? | "I don't know." | Says the documents don't cover it | Invented: $142.50 |
| Can I bring my dog to the office? | "I don't know." | Says the documents don't cover it | Wrongly says no |
| Does the company pay for a gym membership? | "I don't know." | Says no, then says the documents don't mention it | Wrongly says no |
| What is the dress code? | "I don't know." | One run says the documents don't specify; another just gives the visitor badge rule | Says the documents don't specify, then adds the badge rule |

Retrieval and the prompt do different jobs. Retrieval supplies the facts, and the prompt decides what the model does when those facts are missing. Having the documents in the prompt didn't stop it inventing a stock price when told never to admit it didn't know.

### 5. Try a long document

Long documents show why chunking matters. An embedding model has a size limit, and `bge-small-en-v1.5` reads only about the first 512 tokens, ignoring the rest without warning.

**Result:** in a test document of 120 paragraphs with a parking fee in the last one, the question "How much does a parking permit cost?" scored 0.427 against the whole document and 0.852 against that paragraph.

The sample documents are too short to show this, as splitting them by file or by sentence still gave correct answers. Add a long document of your own to see the effect, then change `load_chunks()` to use fixed-size chunks that overlap.

### 6. Swap the embedding model

Set `EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`.

**Result:** the top chunk stayed correct on the three questions tried, but the scores dropped a lot, from 0.734 to 0.593 on the food budget question. Scores only mean something within one model, so a cutoff from experiment 3 won't carry over.

### 7. Turn on thinking

Delete the `extra_body` line and set `max_tokens=2000`.

**Result:** the same answer took about 13 seconds and 746 tokens, instead of under a second. If you leave `max_tokens` at 500, thinking uses all of it and the script crashes before there's an answer.

### 8. Use your own documents

Put `.md` files in `docs/private/`. `rag.py` reads them, and this repo's `.gitignore` keeps any `private/` folder out of git. If you copy this folder into another repo, add `private/` to that repo's `.gitignore` too.

## Limits of this version

- It embeds every document again on each run.
- It always sends 3 chunks, however weak the match.
- Chunks are paragraphs with no maximum size, so anything past the embedding model's limit is ignored.
- It searches by meaning only, so it can miss exact terms like product codes.
- The model can read too much into related text. "Do we get free lunch?" gets a "no", even though the documents only cover travel meals.
- It only reads `.md` files.
- Nothing measures quality automatically.

## What to learn next

| Topic | What it solves |
|-------|----------------|
| Vector databases | Saves embeddings so they aren't recomputed on every run |
| Hybrid search | Combines keyword and meaning search to catch exact terms |
| Reranking | Uses a second model to reorder the retrieved chunks more accurately |
| Evaluation | Turns the test table above into a script that measures how often the right chunk is retrieved |

## Further reading

- [The original RAG paper](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), on improving chunks and combining search methods
- [FastEmbed](https://github.com/qdrant/fastembed), the embedding library used here
- [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), the embedding model used here

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `openai.APIConnectionError: Connection error.` | The LLM server isn't running | Start it (see [Setup](#setup)) |
| `AttributeError: 'NoneType' object has no attribute 'strip'` | Thinking used all of `max_tokens` before any answer | Raise `max_tokens`, or keep thinking off |
| `IndexError: list index out of range` | No question was given | Pass the question in quotes |
| VS Code says `Import "fastembed" could not be resolved` | VS Code is using a different Python environment | Run **Python: Select Interpreter** and choose `rag/.venv/bin/python` |
