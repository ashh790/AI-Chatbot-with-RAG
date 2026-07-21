<div align="center">

# AI Chatbot with RAG

**A self-hosted chat assistant that answers from your own documents, searches the live web, and runs tools — on any LLM provider, including free ones.**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

</div>

---

## What it does

Upload a PDF and ask questions about it. Ask what happened in the news today. Ask it
to do arithmetic. It figures out which capability it needs and uses it.

| | |
|---|---|
| **Retrieval (RAG)** | Upload PDF, DOCX, TXT, MD, CSV or JSON. Documents are chunked, indexed, and cited in answers. |
| **Live web** | DuckDuckGo search and page fetching, so it can answer past the model's training cutoff. |
| **Tool calling** | Calculator, document reader, knowledge-base search, web search, URL fetch, clock. |
| **Any provider** | OpenAI, Google Gemini, Groq, OpenRouter, or a local model. Two lines in `.env`. |
| **Works without paid API** | Gemini, Groq and OpenRouter all have free tiers with no credit card. |
| **No mandatory vector DB** | Uses Chroma if installed; otherwise falls back to a built-in BM25 index with zero dependencies. |
| **Conversation memory** | Per-session history, trimmed in whole turn pairs. |
| **Themed, responsive UI** | Light / dark / system, single-page, works on mobile. |

---

## Quick start

### Windows — two double-clicks

```
setup.bat      installs everything, then runs diagnostics
start.bat      kills any stale server and launches the app
```

### Any platform — manual

```bash
git clone https://github.com/ashh790/AI-Chatbot-with-RAG.git
cd AI-Chatbot-with-RAG

python -m venv venv
venv\Scripts\activate           # Windows
source venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
cp .env.example .env            # then add your API key
python app.py
```

Open **http://127.0.0.1:5000**.

Without a valid key the app still boots in **demo mode** — it starts, the UI works,
and every reply explains exactly what's missing rather than failing silently.

---

## Choosing a provider

The app speaks the OpenAI protocol, so any compatible provider works. **OpenAI itself
has no free tier** (free credits ended in 2024), so the free options below are
third-party.

| Provider | Free tier | Card? | Get a key |
|---|---|---|---|
| **Groq** | ~30 req/min, ~1,000/day | No | [console.groq.com/keys](https://console.groq.com/keys) |
| **Google Gemini** | ~250 req/day | No | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| **OpenRouter** | ~200 req/day on `:free` models | No | [openrouter.ai/keys](https://openrouter.ai/keys) |
| **OpenAI** | None — pay as you go | Yes | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

Pick one and set three lines in `.env`:

```ini
# Groq
OPENAI_API_KEY=gsk_your_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
```

```ini
# Google Gemini
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-3.1-flash-lite
```

```ini
# OpenAI (leave BASE_URL empty)
OPENAI_API_KEY=sk-your_key_here
OPENAI_BASE_URL=
MODEL_NAME=gpt-4o-mini
```

`.env.example` has all four, commented and ready to uncomment.

> **Gemini note.** If a model 404s with *"no longer available to new users"*, list what
> your key can actually reach:
> `curl -H "x-goog-api-key: $KEY" https://generativelanguage.googleapis.com/v1beta/models`

---

## How it works

```
                    ┌──────────────┐
   your question ──▶│   routes.py  │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐      ┌────────────────────┐
                    │  services.py │◀────▶│ memory.py          │  last N turns
                    └──────┬───────┘      └────────────────────┘
                           │
              ┌────────────┼─────────────┐
              ▼            ▼             ▼
      ┌──────────────┐ ┌────────┐ ┌─────────────┐
      │ utils.py     │ │prompts │ │  tools.py   │
      │ retrieval    │ │system  │ │ dispatch    │
      └──────┬───────┘ └────────┘ └──────┬──────┘
             │                            │
    ┌────────┴────────┐        ┌──────────┴──────────┐
    ▼                 ▼        ▼                     ▼
┌────────┐   ┌───────────────┐ ┌──────────┐  ┌──────────────┐
│ Chroma │   │ BM25 fallback │ │ web.py   │  │ documents.py │
│ vectors│   │ (stdlib only) │ │ search   │  │ extraction   │
└────────┘   └───────────────┘ │ + fetch  │  └──────────────┘
                               └──────────┘
```

**A request, end to end:**

1. `routes.py` validates the message and hands it to `services.py`.
2. Retrieval runs against the vector store; matching chunks become context.
3. `prompts.py` assembles the system message — instructions, the **list of indexed
   documents**, and the retrieved context.
4. The model is called with the tool schemas attached.
5. If it returns tool calls, **every** call is executed and answered, then the model
   is called again. Loops until it returns prose or hits `MAX_TOOL_ROUNDS`.
6. The turn is appended to memory and the reply returned.

**Why the document list is in the prompt.** Retrieval is similarity-based, so a
question like *"summarise the uploaded PDF"* matches no chunk and injects no context —
and the model then wrongly insists no document exists. Listing what's indexed prevents
that, and tells it which name to pass to `read_document`.

### Retrieval backends

| | Chroma | BM25 fallback |
|---|---|---|
| Install | `pip install chromadb` | none — standard library |
| Matching | Semantic (embeddings) | Lexical (keyword) |
| "car" finds "automobile" | Yes | No |
| Needs API quota | Yes, for embeddings | No |

The fallback exists because `chromadb` is a large install that fails on plenty of
machines. RAG shouldn't be gated behind it. The app picks Chroma when available and
degrades automatically; the footer shows which is live.

---

## Project structure

```
.
├── app.py                  entrypoint, banner, port handling
├── ingest.py               bulk CLI: docs/ -> vector store
├── doctor.py               diagnostics — run this when something breaks
├── setup.bat / start.bat   Windows one-click install and run
├── requirements.txt
├── .env.example            provider presets, commented
├── docs/                   your documents live here (gitignored)
├── vectorstore/            generated index (gitignored)
├── templates/
│   └── index.html          the entire UI — one file, no build step
└── chatbot/
    ├── __init__.py         app factory, logging, upload limits
    ├── config.py           env-driven settings, provider detection
    ├── routes.py           HTTP endpoints
    ├── services.py         chat orchestration + tool loop + error classification
    ├── prompts.py          system prompt assembly
    ├── memory.py           per-session conversation history
    ├── utils.py            vector store wrapper + chunking
    ├── fallback_store.py   BM25 index, stdlib only
    ├── documents.py        PDF/DOCX/text extraction, filename safety
    ├── tools.py            tool schemas + dispatch
    └── web.py              web search, page fetch, SSRF guards
```

---

## Using it

### Chat

Type and hit Enter. `Shift+Enter` for a newline. The composer grows as you type.

### Documents

Click **+** in the composer or the sidebar, or **drag files anywhere on the page**.
They're saved to `docs/`, chunked, indexed, and immediately queryable. The sidebar
lists what's indexed with chunk counts and a delete button each.

Re-uploading the same filename **replaces** it rather than creating a second copy.

For bulk loading, skip the UI:

```bash
python ingest.py                  # everything in docs/
python ingest.py --path file.pdf  # one file
python ingest.py --reset          # wipe and rebuild
python ingest.py --stats          # how many chunks are indexed
```

### Web

Just ask. *"What's the latest Python release?"* triggers a search; paste a URL and it
fetches the page. Answers cite their source.

---

## Tools

| Tool | Purpose |
|---|---|
| `calculator` | Arithmetic via AST evaluation — **not** `eval` |
| `read_document` | Full text of one indexed document, for summaries and analysis |
| `search_knowledge_base` | Chunk-level search across indexed documents |
| `search_web` | Live DuckDuckGo search |
| `fetch_url` | Downloads a page, strips scripts/styles/nav, returns readable text |
| `get_current_time` | Current UTC time |

None require an API key beyond your LLM provider.

**Adding your own:** append a schema to `TOOLS` and register a handler in
`TOOL_HANDLERS` in `chatbot/tools.py`. Handlers take a dict and return a string.

---

## API

| Method | Endpoint | Body / Params | Returns |
|---|---|---|---|
| `GET` | `/` | — | Chat UI |
| `GET` | `/api/health` | — | Provider, model, backend, chunk count, key diagnostics |
| `POST` | `/api/chat` | `{message, session_id}` | `{response, used_rag, demo_mode, session_id}` |
| `POST` | `/api/reset` | `{session_id}` (optional) | Clears memory; omit id to clear all |
| `POST` | `/api/upload` | multipart `files` | `{uploaded[], failed[], total_chunks}` |
| `POST` | `/api/ingest` | `{doc_id, text}` | Index raw text without a file |
| `GET` | `/api/documents` | — | `{documents[], total_chunks}` |
| `DELETE` | `/api/documents/<name>` | — | `{chunks_removed, file_removed}` |

`/api/health` is the fastest way to see what's wrong:

```json
{
  "status": "ok",
  "provider": "Google Gemini",
  "model": "gemini-3.1-flash-lite",
  "llm_connected": true,
  "demo_mode": false,
  "key_problem": null,
  "vector_store_backend": "bm25",
  "documents_indexed": 12,
  "active_sessions": 1
}
```

---

## Configuration

Everything lives in `.env`. The file is read at startup, so **restart after editing**.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Your provider's key |
| `OPENAI_BASE_URL` | *(empty)* | Set for non-OpenAI providers |
| `MODEL_NAME` | `gpt-4o-mini` | Chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Only used on OpenAI's own endpoint |
| `SECRET_KEY` | `dev-secret-change-me` | Flask session signing |
| `PORT` | `5000` | Change if 5000 is taken |
| `FLASK_DEBUG` | `false` | Debug error pages |
| `FLASK_RELOAD` | `false` | Auto-restart on code edits |
| `RAG_RESULTS` | `4` | Chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `MAX_HISTORY_MESSAGES` | `10` | Conversation turns kept |
| `MAX_TOOL_ROUNDS` | `5` | Guards against tool-call loops |
| `COLLECTION_NAME` | `knowledge_base` | Chroma collection |

> **Why `FLASK_RELOAD` defaults off.** The reloader watches the project tree, and
> uploads write into `docs/` inside it — so it restarts the server mid-upload and the
> browser reports a dropped connection. Only turn it on while editing code.

---

## Security

Three deliberate defences. They look like over-engineering until you consider that the
model chooses what to fetch, and users choose what to upload.

### SSRF protection — `chatbot/web.py`

`fetch_url` takes a URL chosen by the *model*, which can be influenced by user input or
by text on a page it already read. Every hostname is resolved and rejected if it maps
to a loopback, private, link-local, reserved, or multicast address — **re-checked after
redirects**, since a public URL can redirect inward.

Without this, *"fetch http://127.0.0.1:5000/api/health"* turns the bot into a proxy
into your own machine, or into cloud metadata endpoints on a server. 14 hostile URL
patterns are covered, including `localhost`, `169.254.169.254`, `[::1]`, `127.1`, and
`file://`.

### Prompt-injection framing — `chatbot/prompts.py`

Web pages can contain text aimed at the model (*"ignore previous instructions…"*). All
fetched content is wrapped in an `[UNTRUSTED WEB CONTENT]` banner, and the system
prompt instructs the model to report what a page **says**, never act on what it
**asks**.

This is mitigation, not a guarantee. No framing is fully injection-proof — bear that in
mind before giving this bot tools with real side effects.

### Upload safety — `chatbot/documents.py`

Filenames come from the browser, so they're attacker-controlled. `safe_filename()`
strips path components under both `/` and `\`, normalises unicode (so RTL-override and
look-alike characters can't smuggle separators), and the resolved path is re-checked
against `docs/`. Without it, an upload named `../../.env` overwrites your API key.

Also enforced: 25 MB per file, extension allowlist, 800-page PDF cap, and
memory/recursion guards so a malformed PDF can't take the process down.

### Secrets

`.env` is gitignored. So are `docs/` and `vectorstore/`. Never commit your key — if you
already have, rotate it, since git history is public.

---

## Troubleshooting

Run this first — it tests the whole stack in-process, no browser involved:

```bash
python doctor.py
```

| Symptom | Cause and fix |
|---|---|
| Replies say "demo mode" | No valid key. The message names the exact reason; check `/api/health`. |
| `insufficient_quota` | Provider account has no credit. Switch to a free provider above. |
| Model 404s | Model retired or unavailable to your key. Try `gemini-3.1-flash-lite` or `llama-3.3-70b-versatile`. |
| Page won't update after edits | Stale server process. `netstat -ano \| findstr :5000` then `taskkill /PID <pid> /F`. |
| Upload fails, "connection dropped" | Reloader restarting mid-upload. Set `FLASK_RELOAD=false`. |
| Bot says no document is uploaded | Fixed — the document list is in the prompt. If it recurs, check `/api/documents`. |
| Answers ignore your documents | Keyword backend missed the phrasing. Ask using words from the document, or install `chromadb`. |
| PDF indexes 0 chunks | Scanned PDF — it's images, not text. OCR it first. |
| `ModuleNotFoundError` | venv not activated, or `pip install -r requirements.txt` not run. |

---

## Limitations

Honest list — this is a working app, not a production deployment.

- **Memory is in-process.** Resets on restart, not shared across workers. Use Redis
  before running more than one worker.
- **Werkzeug's dev server** is what `app.py` runs. For real deployment use gunicorn or
  waitress behind a reverse proxy.
- **No authentication.** Anyone who can reach the port can use it and read your
  documents. Don't expose it to the internet as-is.
- **BM25 is lexical.** Without `chromadb`, conceptually-worded questions may miss.
- **Scanned PDFs need OCR** — they contain images, not extractable text.
- **No streaming.** Replies arrive complete rather than token by token.

---

## Requirements

```
flask>=3.0.0            python-dotenv>=1.0.0    openai>=1.30.0
pypdf>=4.0.0            python-docx>=1.1.0      requests>=2.31.0
ddgs>=9.0.0             chromadb>=0.5.0  (optional)
```

Python 3.10 or newer.

---

## License

MIT — do what you like with it.
