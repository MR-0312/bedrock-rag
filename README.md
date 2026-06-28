# 🧠 DocuMind — Chat with your PDF (Amazon Bedrock)

A polished RAG (retrieval-augmented generation) app: upload a PDF, ask questions,
get answers **grounded in the document with page citations**. Built to demo well.

**Chat model:** `moonshotai.kimi-k2.5` — Moonshot Kimi K2.5, a frontier
non-Anthropic model (top-tier reasoning, large context), called through Bedrock's
native **Converse API**. Switchable live from the UI.

![flow](https://img.shields.io/badge/PDF→chunks→embeddings→FAISS→Bedrock-blue)

## Features

- 🎛️ **Live model switcher** — Kimi K2.5, DeepSeek V3.2, Mistral Large 3, GLM-5,
  Qwen3, gpt-oss, Nova — swap mid-demo from the ⚙️ settings panel.
- 🔎 **Grounded answers with page citations** + per-source relevance bars.
- ⚡ **Streaming responses** with token & latency metrics.
- 🎚️ **Tunable** temperature, max tokens, and retrieval depth (`k`).
- 📊 Document stats, suggested questions, clean empty state, custom styling.
- ⚡ **Fast Bedrock embeddings** (Titan v2) — server-side, parallelized, no local model.

## Architecture

| Step | Tech | Notes |
|------|------|-------|
| Chat | **boto3 Converse API** | One uniform format for ALL Bedrock models |
| Embeddings | **Bedrock Titan v2** (1024-dim) | Server-side, parallelized — fast, no local model |
| Vector search | **FAISS** (in-memory, cosine) | Simple and fast |
| UI | **Chainlit** (primary) | Chat-native: streaming, source elements, settings panel |

> Why not the OpenAI SDK? On Bedrock the OpenAI-compatible endpoint only serves
> `gpt-oss-*` models. Converse reaches everything else, so you get real model choice.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then edit .env and paste your Bedrock API key
```

Your Bedrock **API key** is a bearer token → put it in `.env` as
`AWS_BEARER_TOKEN_BEDROCK` (boto3 reads it automatically).

## Run

```bash
# 1) Confirm your key works (chat + Bedrock embeddings smoke test)
python bedrock_client.py

# 2) Launch the app (Chainlit) — opens http://localhost:8000
chainlit run chainlit_app.py -w
```

Upload a **text-based** PDF when prompted → ask away. Use the ⚙️ panel (top-right)
to switch models and tune temperature / max tokens / retrieval depth.

## Files

| File | Purpose |
|------|---------|
| `chainlit_app.py` | **Chainlit UI** — chat, model/settings panel, source elements |
| `bedrock_client.py` | Converse chat (+ streaming/usage) and Bedrock embeddings; run for a smoke test |
| `rag.py` | PDF → chunks → embeddings → FAISS retrieval + grounded prompt |
| `discover.py` | Lists & live-tests which Bedrock models your key can access |
| `chainlit.md` | Welcome screen shown in the Chainlit app |

## Notes & gotchas

- **Text-based PDFs only.** Scanned/image PDFs yield no text; the app detects this
  and tells you (add OCR if you truly need it).
- **Model access** is per account/region. Run `python discover.py` to see what your
  key supports, then pick any ID for `CHAT_MODEL_ID` or the UI dropdown.
- **Temporary keys** are region-scoped and expire (often ~12h). On expiry, paste a
  fresh `AWS_BEARER_TOKEN_BEDROCK` into `.env`.
- **Cost**: both chat and Titan embeddings consume Bedrock tokens (embeddings are
  very cheap). Nova 2 Lite is the cheapest/fastest chat option in the dropdown.
- **Max answer tokens** in the ⚙️ panel auto-adjusts to each model's real output
  limit (e.g. Nova Pro 10k, Kimi K2.5 262k).
