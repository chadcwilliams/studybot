# StudyBot

A free, self-hosted RAG chatbot that lets students ask questions about a
course (slides, syllabus, notes) through a plain web page. This instance is
configured for **Psyc 351D - Biopsychology**, but the code is reusable for
any course — see "Reusing this for another course" below.

Built to run entirely on free tiers:

- **LLM inference:** [Groq](https://groq.com) (free API, no credit card, currently `openai/gpt-oss-20b`)
- **Embeddings:** local `fastembed` (ONNX-based, no API calls, no cost, and light enough to run in Render's free-tier 512MB RAM — a torch-based alternative like `sentence-transformers` will OOM there)
- **Vector store:** a small local numpy-based store (`src/studybot/store.py`) — no external vector DB, no compiled dependencies to install
- **Backend hosting:** [Render](https://render.com) free web service
- **Frontend hosting:** GitHub Pages (served from `/docs`)

## How it works

1. Course materials (`.pptx`, `.pdf`, `.docx`) live in `data/materials/`.
2. `python -m studybot.ingest` extracts the text and tables, splits it into
   chunks, embeds each chunk locally, and stores the vectors in a local index
   file (`data/vector_index/`). Table-heavy documents (like a syllabus) get
   special handling — see "Ingestion details" below.
3. The FastAPI backend (`src/studybot/main.py`) exposes a `/chat` endpoint:
   a student's question (plus recent conversation history, if any) is
   embedded, the most relevant chunks are retrieved, and both are sent to
   Groq's LLM to generate a grounded, Markdown-formatted answer.
4. The static page in `docs/` is a plain HTML/JS chat UI that calls the
   backend, renders the response as Markdown (tables, bold, lists), and is
   hosted for free by GitHub Pages.

## Features

- **Conversation memory** — follow-up questions ("lay it out in steps") are
  understood in context of what was just asked. Retrieval runs twice for
  follow-ups (once on the question alone, once with recent turns folded in)
  and merges the results, so both genuine follow-ups and topic switches
  right after them are handled correctly.
- **Grounded, sourced answers** — every response cites which file and
  section/table it came from, grouped and simplified into one readable line
  per source file.
- **Rate limiting + caching** — a soft server-side rate limit and an answer
  cache protect Groq's free-tier quota from being blown through by a burst
  of simultaneous questions (e.g. the night before a deadline). Caching is
  automatically disabled for any question that has conversation history,
  since the same follow-up text can have a different correct answer
  depending on what preceded it.
- **Markdown rendering** — bot responses render as actual formatted
  Markdown (tables, bold, lists) client-side, sanitized before display.

## Ingestion details (why the docx parsing is more involved than it looks)

Getting reliable answers out of a real syllabus took more than "extract the
text." Some things worth knowing if you're adding your own materials or
debugging a bad answer:

- **Tables are chunked row-by-row**, not joined into one blob — a long table
  joined into a single string can get sliced by the character-based chunker
  at an arbitrary point, severing a fact (e.g. "Exam 1" from its date) across
  two chunks.
- **Row chunks only mention the columns that have content** — early on, every
  row repeated the full header line verbatim, which made all rows in a long
  table (even totally unrelated ones) carry faint similarity to any question
  mentioning a column name. Omitting empty cells fixed this.
- **Columns with "due"/"deadline" in the header get an auto-generated summary
  chunk** listing every entry in that column as a plain sentence (e.g.
  "Summary 1 is due on Sep 28...") — a single dense, natural-language chunk
  beats hoping semantic search finds every scattered "needle" row on its own.
- **Single-row tables** (e.g. a grading breakdown with one row per category)
  are split into one chunk per column (so categories can't bleed into each
  other), *plus* one consolidated chunk with all columns together (so a
  broad "what's the whole breakdown" question doesn't need every column
  independently retrieved to answer correctly).
- **Narrative text is split at heading boundaries**, not treated as one
  undifferentiated blob. A paragraph only counts as a real heading if it's
  short and doesn't end in sentence-final punctuation — some documents apply
  a "Heading" style to body paragraphs too, and a naive style-only check
  wipes that content out rather than just misplacing it.
- **Known gap:** sub-headers that use plain body-text styling with no
  heading marker at all (not even a mis-tagged one) won't be detected, and
  will get absorbed into whichever real heading precedes them. If a specific
  policy seems to be missing from answers, this is the first thing to check.

## Repo layout

```
.
├── src/studybot/       # the Python package (backend + ingestion)
│   ├── config.py        # settings (model names, chunk size, CORS origins...)
│   ├── ingest.py        # parses course materials -> chunks -> vector index
│   ├── rag.py            # retrieval logic + multi-query merge
│   ├── store.py           # local numpy-based vector store (no chromadb)
│   ├── llm.py              # Groq API call + system prompt
│   ├── cache.py             # rate limiter + answer cache
│   └── main.py               # FastAPI app ("/chat", "/health")
├── data/
│   └── materials/       # course materials (gitignored unless repo is private)
├── docs/                # static frontend, served by GitHub Pages
│   ├── index.html        # course name/title/greeting hardcoded here
│   ├── style.css
│   └── app.js             # API_BASE_URL + conversation history tracking
├── scripts/
│   └── run_ingest.sh
├── requirements.txt
├── .env.example
└── render.yaml           # Render Blueprint config, pins PYTHON_VERSION
```

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GROQ_API_KEY

# 1. Add your files to data/materials/, then build the vector index
python -m studybot.ingest

# 2. Run the backend locally
uvicorn studybot.main:app --reload --app-dir src
```

Open `docs/index.html` locally (or with `python -m http.server` inside
`docs/`) and point `API_BASE_URL` in `docs/app.js` at `http://localhost:8000`.

## Getting a free Groq API key

1. Sign up at [console.groq.com](https://console.groq.com) — no card required.
2. Create an API key and put it in `.env` as `GROQ_API_KEY`.
3. The free tier is rate-limited (currently ~30 requests/min, ~1,000/day),
   shared across *all* users of your key. The rate limiter/cache in this repo
   help stretch that budget — see "Free-tier limits" below.
4. **Groq's model lineup changes.** This repo was originally built against
   `llama-3.3-70b-versatile`, which Groq deprecated entirely mid-project —
   it's now Enterprise-only. The current default, `groq_model` in
   `config.py`, is `openai/gpt-oss-20b`. If Groq returns a `model_not_found`
   error, check [console.groq.com/docs/models](https://console.groq.com/docs/models)
   for the current lineup and update that one setting.

## Deploying

**Backend (Render):**
1. Push this repo to GitHub.
2. On Render, create a new **Blueprint** from the repo — it reads
   `render.yaml` automatically, which sets the build/start commands and pins
   `PYTHON_VERSION` (Render's default Python version can be too new to have
   prebuilt wheels for every dependency; 3.12 is a safe, tested choice here).
3. Fill in the env vars marked `sync: false`: `GROQ_API_KEY` and
   `ALLOWED_ORIGINS` (your GitHub Pages origin, e.g.
   `https://yourname.github.io` — no path, no trailing slash).
4. **Getting your materials onto the server.** `data/materials/` is
   gitignored by default. Render's free tier has no shell access and no
   persistent disk, so the only path is: **make the GitHub repo private**,
   remove the `data/materials/*` line from `.gitignore`, and commit your real
   files. `render.yaml`'s build command runs `python -m studybot.ingest` on
   every deploy, so Render rebuilds the index from whatever's in the repo —
   this is required, not optional, on the free tier.
5. Render's free tier sleeps after inactivity; the first request after idle
   time (and the very first request ever, which also loads the embedding
   model) will be noticeably slower. The frontend's greeting message already
   warns students about this.

**Frontend (GitHub Pages):**
1. In the repo's GitHub settings → Pages, set source to the `docs/` folder on
   your main branch.
2. Edit `docs/app.js` → set `API_BASE_URL` to your Render backend URL
   (no trailing slash).
3. Your chatbot is live at `https://yourname.github.io/studybot`.

## Free-tier limits — what to expect at 100 students

Groq's free tier applies rate limits per API key/org, not per student. At
current published limits (~30 requests/min, ~1,000/day), a burst of
simultaneous questions (e.g. the night before a deadline) can hit the
ceiling. This repo mitigates that with:
- An **answer cache** (`cache.py`) — identical/near-identical questions with
  no conversation history are served from cache instead of hitting Groq again.
- A **soft rate limit** that returns a friendly "we're at capacity, try again
  in a moment" message instead of a raw error.

Render's free tier also shares 750 instance-hours/month across your whole
workspace, not per-service — worth knowing if you ever run more than one
class's bot at once (see below).

## Reusing this for another course

The cleanest approach: keep `main` as the clean package (no materials, no
course-specific branding baked in) and create one git branch per class with
that class's `data/materials/`. Each class gets its own Render service
(pointed at its branch) but can share one GitHub Pages deployment — put each
class's frontend in its own `docs/<class>/` subfolder on `main`, each
pointing `API_BASE_URL` at its own Render backend. Two places currently need
manual updates per course, since there's no shared config between them:
`src/studybot/config.py`'s `course_name` (used in the system prompt) and the
hardcoded title/greeting text in `docs/index.html` (a static frontend can't
read Python config at runtime).

## License

MIT — see `LICENSE`.
