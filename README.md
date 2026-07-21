# AI Chatbot with RAG

Flask + OpenAI chatbot with document retrieval, conversation memory, and tool calling.

## Setup

```bash
cd Chatbot
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI key. Then:

```bash
python ingest.py      # index anything in docs/
python app.py         # http://127.0.0.1:5000
```

Without a valid key the app still boots in **demo mode** — the UI shows an amber
status dot and echoes your input instead of calling the API.

## Layout

```
Chatbot/
├── app.py              entrypoint
├── ingest.py           document -> vector store CLI
├── docs/               drop your PDFs and text files here
├── vectorstore/        generated Chroma index (gitignored)
├── templates/
│   └── index.html      chat UI
└── chatbot/
    ├── __init__.py     app factory
    ├── config.py       env-driven settings
    ├── routes.py       HTTP endpoints
    ├── services.py     chat orchestration + tool loop
    ├── utils.py        vector store + chunking
    ├── tools.py        tool schemas + dispatch
    ├── memory.py       per-session history
    └── prompts.py      system prompt assembly
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Chat UI |
| GET | `/api/health` | Status, key validity, chunk count |
| POST | `/api/chat` | `{message, session_id}` → `{response, used_rag, demo_mode}` |
| POST | `/api/reset` | Clear one session's memory (omit `session_id` for all) |
| POST | `/api/ingest` | `{doc_id, text}` → add raw text without the CLI |
| POST | `/api/upload` | multipart `files` → save to `docs/` and index |
| GET | `/api/documents` | List indexed documents with chunk counts |
| DELETE | `/api/documents/<name>` | Remove one document's chunks and its file |

## File uploads

Drag PDF, DOCX, TXT, MD, CSV or JSON onto the panel in the UI (or click to
browse). Files are saved into `docs/`, chunked, and indexed — then you can ask
about them straight away. The panel lists what's indexed and lets you delete
individual documents.

Limit is 25 MB per file, enforced both in `documents.py` and via Flask's
`MAX_CONTENT_LENGTH` so an oversized body is rejected before it's buffered.

DOCX extraction includes table cells, not just paragraphs — a lot of real
documents keep their actual data in tables.

**Filenames from the browser are attacker-controlled.** `safe_filename()`
strips every path component under both `/` and `\` conventions, normalises
unicode (so RTL-override and look-alike characters can't smuggle separators),
and `resolve_within()` confirms the final path is genuinely inside `docs/`.
Without this, an upload named `../../.env` overwrites your config. 13 hostile
filename patterns are covered by tests, with zero escapes.

The chat response includes both `response` and `reply` with the same value, so
either key works on the front end.

## Tools

| Tool | What it does | Needs a key? |
|---|---|---|
| `calculator` | Safe arithmetic (AST-based, not `eval`) | no |
| `get_current_time` | Current UTC time | no |
| `search_knowledge_base` | Searches your ingested documents | no |
| `search_web` | Live DuckDuckGo search | no |
| `fetch_url` | Downloads a page and extracts readable text | no |

Add your own in `chatbot/tools.py`: append a schema to `TOOLS` and register a
handler in `TOOL_HANDLERS`.

### Live web access

`search_web` and `fetch_url` let the bot answer questions past its training
cutoff. Both are free — DuckDuckGo needs no API key.

```bash
pip install requests ddgs
```

Then ask things like *"search the web for the latest Flask release"* or
*"summarize https://example.com"*.

**Two safety properties worth knowing about, because removing them is easy and
costly:**

*SSRF protection.* The URL passed to `fetch_url` is chosen by the model, which
can be influenced by user input or by text on a page it already read. So
`chatbot/web.py` resolves every hostname and refuses any address that is
loopback, private, link-local, reserved, or multicast — and re-checks after
redirects. Without this, a crafted prompt could make the bot fetch
`http://127.0.0.1:5000/...` or a cloud metadata endpoint and read back the
response. 14 hostile URL patterns are covered by tests.

*Prompt-injection framing.* Web pages can contain text aimed at the model
("ignore previous instructions…"). All fetched content is wrapped in an
`[UNTRUSTED WEB CONTENT]` banner and the system prompt instructs the model to
treat it as quoted data — report what a page says, never act on what it asks.
This is mitigation, not a guarantee; don't give this bot access to anything
you'd mind a hostile web page influencing.

## Configuration

All settings read from `.env` — see `.env.example`. Notable ones:

| Variable | Default | Notes |
|---|---|---|
| `MODEL_NAME` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Used when a valid key is present |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Raise overlap if answers get cut off |
| `RAG_RESULTS` | 4 | Chunks retrieved per query |
| `MAX_HISTORY_MESSAGES` | 10 | Trimmed in whole turn pairs |
| `MAX_TOOL_ROUNDS` | 5 | Guards against tool-call loops |

## Notes and limits

- **Memory is per-process.** It resets on restart and isn't shared across
  workers. Move it to Redis before running more than one worker.
- **Embeddings need a key.** Without one, Chroma falls back to a bundled model
  that downloads ~80MB on first use.
- **Scanned PDFs extract nothing.** Run OCR first.
- **`FLASK_DEBUG=true` binds to 127.0.0.1 only.** The Werkzeug debugger is
  remote code execution to anyone who can reach it — never pair debug mode with
  `0.0.0.0`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Replies echo your input | No valid `OPENAI_API_KEY` — check `/api/health` |
| `documents_indexed: 0` | Run `python ingest.py` |
| `vector_store_available: false` | `pip install chromadb` |
| Answers ignore your docs | Chunks indexed but not matching — try raising `RAG_RESULTS` |
