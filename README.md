
# arXiv Butler

A small, interactive “agentic” CLI app built with **LangChain** that runs as a **single, continuous chat conversation** in your terminal (each prompt/response is part of the same in-memory session) and can:

1. **Search arXiv** for papers (via the public arXiv export API)
2. **Download a paper PDF** to `data/papers/`
3. **Answer questions about the paper** using **text extracted from the PDF** (no image/figure understanding)

The entrypoint is a simple terminal chat loop:
```
bash python -m src.main
``` 

---

## Requirements

- **Python 3.13+**
- macOS/Linux/Windows (examples below use macOS-style commands)
- Internet connection (for arXiv + model API calls)

---

## Project structure
```text
arxiv_butler/
  data/
    papers/                   # downloaded PDFs 
  src/
    arxiv_butler/
      main.py                  # interactive CLI entrypoint
      tools/
        arxiv_search.py         # search arXiv (Atom API)
        arxiv_download.py       # download PDFs to disk
        pdf_text.py             # extract text from local PDFs
  requirements.txt            # dependencies
  .env                        # env vars (API key, model, logging)
```

---

## Step-by-step installation

### 1) Clone / open the project

If you already have the folder locally, just `cd` into it:
```
bash cd <PATH_TO_PROJECT>/agents
``` 

### 2) Create a virtual environment

Using the standard library `venv`:
```
bash python3.13 -m venv .venv
``` 

### 3) Activate the virtual environment

**macOS / Linux:**
```bash 
source .venv/bin/activate
```

You should now see the venv reflected in your shell prompt.

### 4) Install dependencies
```bash 
pip install --upgrade pip 
pip install -r requirements.txt
```

### 5) Create a `.env` file

Create a file named `.env` in the project root (next to `requirements.txt`) and add:
```dotenv 
OPENAI_API_KEY=<YOUR_API_KEY>
OPENAI_MODEL=gpt-4o-mini
LOGGING_LEVEL=INFO
``` 

Notes:
- Use a placeholder like `<YOUR_API_KEY>` in documentation; keep real keys private.
- `OPENAI_MODEL` is optional; the code defaults to a model if not set.
- `LOGGING_LEVEL` can be `DEBUG`, `INFO`, `WARNING`, etc.

### 6) Ensure the download folder exists (optional)

The downloader will create it automatically, but you can do:
```bash 
mkdir -p data/papers
``` 

---

## Running the agent

From the project root:
```bash 
python -m arxiv_butler.main
``` 

You’ll get a prompt like:
```text
you>
``` 

Type queries and press Enter.

To exit:
```text 
exit
``` 

or
```text
quit
``` 

---

## Usage examples (recommended flow)

```text
USER: Find the paper Pretraining on the Test Set Is All You Need by Rylan Schaeffer.

AGENT: I found the paper you're looking for:
### Pretraining on the Test Set Is All You Need
- arXiv: [https://arxiv.org/abs/2309.08632v1](https://arxiv.org/abs/2309.08632v1)
- PDF: [https://arxiv.org/pdf/2309.08632v1](https://arxiv.org/pdf/2309.08632v1)

Would you like to download the PDF?

USER: Please Download the paper.

AGENT: The paper has been successfully downloaded. Path: data/papers/2309.08632v1.pdf

USER: Which model was introduced in the paper? 

USER: Which model was introduced in the paper?

AGENT: The paper introduces a model called **phi-CTNL** (pronounced "fictional"), which is a transformer-based language model with 1 million parameters. It is pretrained on a carefully curated, high-quality, non-synthetic dataset consisting of less than 100,000 tokens. The model achieves perfect results across various academic benchmarks, outperforming all known foundation models.

Key features of phi-CTNL include:
- It demonstrates faster-than-power-law scaling with compute.
- It exhibits a grokking-like ability to accurately predict downstream evaluation benchmarks.

The authors emphasize that phi-CTNL challenges the notion that model capabilities are solely determined by parameter scale, suggesting that data quality plays a crucial role.

If you have more questions or need further details, feel free to ask!
```

---

## What “text-only” means

This project **does not interpret figures/plots/images** inside PDFs.

- It can still pick up **captions** and **surrounding text**
- But it won’t “read” a graph the way a vision model would

If you want figure understanding later, you’d add a “render page to image + vision model” pipeline.

---

## Where PDFs are saved

Downloaded PDFs are stored under:
```text 
data/papers/
``` 

The download tool returns the absolute local path after saving.

---

## Configuration

### Environment variables

| Variable | Required | Description |
|---|---:|---|
| `OPENAI_API_KEY` | Yes | API key for the chat model provider |
| `OPENAI_MODEL` | No | Model name (example: `gpt-4o-mini`) |
| `LOGGING_LEVEL` | No | Logging level (`DEBUG`, `INFO`, `WARNING`, ...) |

---

## Troubleshooting

### 1) arXiv search returns no results
Try broader queries:
- remove author name or quotes
- try a keyword-only query

Example:
```text
you> Find a paper about retrieval-augmented generation evaluation
``` 

### 2) PDF downloads but `pdf_extract_text` returns empty/garbled text
Some PDFs have poor text layers (or are scanned). Text extraction quality depends on how the PDF was generated.

Workarounds:
- increase `max_pages` in the tool call (if needed)
- try a different paper
- later: add OCR for scanned PDFs

### 3) Logging is too noisy / too quiet
Change in `.env`:
```dotenv
LOGGING_LEVEL=DEBUG
``` 

or:
```dotenv
LOGGING_LEVEL=WARNING
```

---

## Safety / privacy notes

- Do not commit `.env` files or API keys.
- Downloaded PDFs are stored locally in `data/papers/`.
- The agent may send extracted text to the model provider when answering questions.

---

## Next improvements (ideas)

- Return structured JSON from `arxiv_search` (more reliable “download result #2” workflows)
- Persist “last downloaded PDF path” in state for seamless follow-up Q&A
- Add a simple “paper registry” (list downloaded papers, pick by index)
- Add OCR fallback for scanned PDFs
- Add evaluations: unit tests for parsing + golden outputs for tools

---

### License

### This project

This repository is released under the **MIT License**.

- You can use, modify, and redistribute it freely, including for commercial use.
- There is **no warranty**.

### Third‑party dependencies

This project depends on third‑party libraries (e.g. LangChain, langchain-openai, requests, pydantic, python-dotenv, pypdf).  
Those libraries are **not** covered by this repository’s MIT license; they remain licensed under their **own** licenses.

To review exact dependency licenses in your environment, you can inspect package metadata (e.g., via `pip show <package>`).