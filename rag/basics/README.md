# RAG from scratch

Ask questions about your own documents using a local LLM. This folder builds retrieval-augmented generation (RAG) in about 130 lines of Python with no framework. Every run prints what happens at each step, so you can follow a question all the way through and change any part of it.

## What RAG is

An LLM only knows what it learned in training, so it can't answer questions about your documents. RAG works around that in two stages: find the passages most relevant to the question, then put them in the prompt and ask the LLM to answer from them.

## How it works

![Flow diagram in two stages. Preparing the documents: documents are split into chunks, and each chunk is embedded. Answering a question: the question is embedded, then Retrieve compares it with the chunk embeddings and keeps the best 3 chunks. Those chunks and the question go into a prompt, the LLM reads the prompt, and the answer is printed.](diagrams/how-rag-works.excalidraw.svg)

| Step | What happens | In `rag.py` |
|------|--------------|-------------|
| 1. Chunk | Splits each document into paragraphs | `load_chunks()` |
| 2. Embed chunks | Turns each chunk into 384 numbers (an embedding) that represent its meaning, so similar text gets similar numbers | `embed_chunks()` |
| 3. Embed question | Turns the question into 384 numbers with the same model, so it can be compared with the chunks | `retrieve()` |
| 4. Retrieve | Scores every chunk by cosine similarity (higher means closer in meaning) and keeps the best 3 | `retrieve()` |
| 5. Build prompt | Puts those chunks and the question into one prompt | `generate()` |
| 6. Generate | Sends the prompt to the LLM and gets the answer | `generate()` |

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

The first run installs the packages and downloads the embedding model (64 MB) to `~/.cache/fastembed`. Every run prints the six steps, then the answer:

```
[1/6] CHUNK: split each document into paragraphs
  expenses.md: 4 chunk(s), longest 134 characters
  security.md: 4 chunk(s), longest 108 characters
  time-off.md: 4 chunk(s), longest 110 characters
  Total: 12 chunks

[2/6] EMBED CHUNKS: turn each chunk into numbers that represent its meaning
  Model: BAAI/bge-small-en-v1.5, running inside this script
  Result: 12 chunks x 384 numbers, 18 KB, 0.04s
  First chunk: [0.021, -0.081, 0.032, -0.043, 0.002, ...] 384 numbers, length 1.000

[3/6] EMBED QUESTION: turn the question into numbers with the same model
  Question: How many vacation days do new hires get?
  Vector: [-0.052, -0.028, -0.000, -0.064, 0.026, ...] 384 numbers, length 1.000

[4/6] RETRIEVE: score 12 chunks against the question, keep the best 3
  Score is cosine similarity: higher means closer in meaning.
  0.808  time-off.md: New hires receive 23 vacation days per year, increasing to 2...
  0.703  time-off.md: Unused vacation days roll over until March 31 of the followi...
  0.622  expenses.md: Employees get a one-time home office allowance of $475, whic...
  ---- TOP_K = 3: chunks below this line are dropped ----
  0.601  time-off.md: Every employee gets the second Friday of each month off, kno...
  0.600  time-off.md: Sick leave is not capped, but a doctor's note is required fo...
  0.570  expenses.md: Meals during business travel are reimbursed up to $62 per da...
  0.559  expenses.md: Expense reports must be submitted through the Ledgerly app w...
  0.547  expenses.md: Flights longer than 7 hours may be booked in premium economy...
  0.477  security.md: Lost or stolen laptops must be reported to the Sentinel desk...
  0.446  security.md: Passwords must be at least 16 characters long and are rotate...
  0.409  security.md: Visitors to the Assembly Hall must be escorted at all times ...
  0.406  security.md: Production robot firmware can only be deployed by engineers ...

[5/6] BUILD PROMPT: this is everything the LLM will see
  system:
    Answer using only the context below and cite the source file in brackets. If the context does not contain the answer, say you don't know.

    Context:
    [time-off.md]
    New hires receive 23 vacation days per year, increasing to 28 days after five years of service.

    [time-off.md]
    Unused vacation days roll over until March 31 of the following year, with a limit of 7 days.

    [expenses.md]
    Employees get a one-time home office allowance of $475, which can be claimed after the first 90 days.
  user:
    How many vacation days do new hires get?

[6/6] GENERATE: send the prompt to the LLM and wait for its answer
  Request: POST http://127.0.0.1:8080/v1/chat/completions
  Settings: {'model': 'mlx-community/Qwen3.6-35B-A3B-4bit', 'temperature': 0.2, 'max_tokens': 500, 'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
  Response: 0.9s, finish_reason: stop
  Tokens: 150 in the prompt, 16 generated

Answer:
New hires receive 23 vacation days per year [time-off.md].
```

### What to look for in the logs

| Log step | What it shows you |
|----------|-------------------|
| 1. Chunk | What the documents were split into. The longest chunk matters because the embedding model reads only about 512 tokens, roughly 2,500 characters of English |
| 2. Embed chunks | Each chunk is now 384 numbers. A length of 1.000 means the vectors are normalized, which is why a plain dot product gives cosine similarity |
| 3. Embed question | The question gets 384 numbers from the same model, which is what makes it comparable with the chunks |
| 4. Retrieve | Every chunk, ranked (up to 15 rows). Look around the line: here the unrelated home office chunk (0.622) is kept, because `TOP_K` keeps 3 chunks however weak the third is |
| 5. Build prompt | The exact text the LLM receives. If a fact isn't here, the LLM didn't get it from your documents |
| 6. Generate | The settings sent, the time taken, and the tokens used. `finish_reason: stop` means the model finished, and `length` means it hit `max_tokens` |

The first LLM call after a pause can be much slower. Two took 30 and 75 seconds here, and the calls right after them took under 2 seconds.

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

**When an answer is wrong, the logs show which half failed:**
- If the right chunk is below the line in step 4, retrieval failed. Look at chunking, the embedding model, or `TOP_K`.
- If the right chunk is above the line but the answer is still wrong, generation failed. Check what the LLM received in step 5, then look at the prompt or the LLM.

## Learn by experimenting

Every experiment runs through `rag.py` and changes one thing. Edit the file, run it, then undo your edit with `git restore rag.py`. Most of the results below come from one to three runs on the sample documents, so yours may differ a little.

### 1. See what RAG adds

Compare the model with and without your documents, while pushing it to always answer. In `generate()`, change the prompt to:

```python
system = (
    "You are the HR assistant for Acme Robotics. Always give a confident, specific answer "
    "with exact numbers. Never say you do not know.\n\n"
    f"Context:\n{context}"
)
```

Then run each question below twice: once with `TOP_K = 0`, so no chunks reach the LLM and `Context:` is empty in step 5, and once with `TOP_K = 3`.

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

Now ask "Do I get any extra days off each month?" Step 4 shows why `TOP_K = 1` answers "I don't know.": the wrong paragraph ranks first at 0.669, and the right one sits just below the line at 0.645. With `TOP_K = 3`, the right paragraph is kept and the answer is correct.

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

**Result:** the parental leave question now stops after step 4, without calling the LLM. The gap is narrow, though, and an unrelated chunk scoring 0.622 still reaches step 5 on the vacation question.

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

Long documents show why chunking matters. An embedding model has a size limit, and `bge-small-en-v1.5` reads only about the first 512 tokens, ignoring the rest without warning. Step 1 prints each file's longest chunk, so you can spot one past the limit.

**Result:** in a test document of 120 paragraphs with a parking fee in the last one, the question "How much does a parking permit cost?" scored 0.427 against the whole document and 0.852 against that paragraph.

The sample documents are too short to show this, as splitting them by file or by sentence still gave correct answers. Add a long document of your own to see the effect, then change `load_chunks()` to use fixed-size chunks that overlap.

### 6. Swap the embedding model

Set `EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`.

**Result:** the top chunk stayed correct on the three questions tried, but the scores dropped a lot, from 0.734 to 0.593 on the food budget question. Scores only mean something within one model, so a cutoff from experiment 3 won't carry over.

### 7. Turn on thinking

Delete the `extra_body` line and set `max_tokens=2000`.

**Result:** step 6 now prints the model's reasoning. The answer was still correct, but five runs took 4–12 seconds and 220–700 tokens, instead of under a second and 16 tokens.

Now set `max_tokens=500`. Thinking counts toward that limit, so the outcome changes between runs. In 5 of 8 runs it used all 500 tokens, step 6 showed `finish_reason: length`, and the script crashed because no answer was left. The other 3 finished in 219–338 tokens.

### 8. Use your own documents

Put `.md` files in `docs/private/`. `rag.py` reads them, and this repo's `.gitignore` keeps any `private/` folder out of git. If you copy this folder into another repo, add `private/` to that repo's `.gitignore` too. The logs print your documents' text, so don't paste them anywhere public.

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
| `AttributeError: 'NoneType' object has no attribute 'strip'` | Thinking used all of `max_tokens` before any answer, and step 6 shows `finish_reason: length` | Raise `max_tokens`, or keep thinking off |
| `IndexError: list index out of range` | No question was given | Pass the question in quotes |
| VS Code says `Import "fastembed" could not be resolved` | VS Code is using a different Python environment | Run **Python: Select Interpreter** and choose `rag/basics/.venv/bin/python` |
