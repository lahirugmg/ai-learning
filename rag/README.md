# RAG

Retrieval-augmented generation, from a first working version to the techniques used to improve it. Each folder is a separate project with its own code, dependencies and instructions.

| Folder | What it covers | Status |
|--------|----------------|--------|
| [basics](basics/) | RAG from scratch: chunk, embed, retrieve, generate | Ready to run |
| [evaluation](evaluation/) | Measuring retrieval and answer quality with a script | Planned |
| [hybrid-search](hybrid-search/) | Combining keyword and meaning search to catch exact terms | Planned |
| [reranking](reranking/) | Using a second model to reorder the retrieved chunks | Planned |
| [vector-db](vector-db/) | Saving embeddings so they aren't recomputed on every run | Planned |

## Suggested order

Start with `basics`, then do `evaluation`, because it tells you whether the later techniques actually improve anything. `hybrid-search` and `reranking` both aim to retrieve better chunks. `vector-db` is about speed and updates rather than answer quality, so it can come whenever rebuilding gets slow.

## How these projects are built

- **Self-contained:** each project has its own copy of the code and sample documents it needs, and doesn't import from another folder.
- **Local:** they use a local LLM and local embedding models, so your data stays on your machine.
- **No framework:** the code stays short enough to read in one sitting.
- **Every step is logged:** each run prints what happens at each step, so you can follow a question through the pipeline.
