# StudyBot

A free, self-hosted RAG chatbot that lets students ask questions about a course
(slides, syllabus, notes) through a plain web page. Built to run entirely on
free tiers:

- **LLM inference:** [Groq](https://groq.com) (free API, no credit card, Llama 3.3 70B)
- **Embeddings:** local `sentence-transformers` model (no API calls, no cost)
- **Vector store:** a small local numpy-based store (`src/studybot/store.py`) — no external vector DB, no compiled dependencies to install
- **Backend hosting:** [Render](https://render.com) free web service
- **Frontend hosting:** GitHub Pages (served from `/docs`)

## How it works

1. You drop your course materials (`.pptx`, `.pdf`, `.docx`) into `data/materials/`.
2. `python -m studybot.ingest` extracts the text, splits it into chunks, embeds
   each chunk locally, and stores the vectors in a local index file.
3. The FastAPI backend (`src/studybot/main.py`) exposes a `/chat` endpoint:
   a student's question is embedded, the most relevant chunks are retrieved,
   and both are sent to Groq's LLM to generate a grounded answer.
4. The static page in `docs/` is a plain HTML/JS chat UI that calls the backend
   and is hosted for free by GitHub Pages.

## Repo layout

```
.
├── src/studybot/       # the Python package (backend + ingestion)
│   ├── config.py        # settings (model names, chunk size, CORS origins...)
│   ├── ingest.py        # parses course materials -> chunks -> vector index
│   ├── rag.py            # retrieval logic
│   ├── store.py           # local numpy-based vector store (no chromadb)
│   ├── llm.py             # Groq API call
│   ├── cache.py            # simple in-memory rate limiter + answer cache
│   └── main.py              # FastAPI app ("/chat", "/health")
├── data/
│   └── materials/       # put your slides/syllabus/notes here (gitignored)
├── docs/                # static frontend, served by GitHub Pages
│   ├── index.html
│   ├── style.css
│   └── app.js
├── scripts/
│   └── run_ingest.sh
├── requirements.txt
├── .env.example
└── render.yaml           # optional one-click Render config
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

Open `docs/index.html` locally (or with `python -m http.server` inside `docs/`)
and point `API_BASE_URL` in `docs/app.js` at `http://localhost:8000`.

## Getting a free Groq API key

1. Sign up at [console.groq.com](https://console.groq.com) — no card required.
2. Create an API key and put it in `.env` as `GROQ_API_KEY`.
3. The free tier is rate-limited (currently ~30 requests/min, ~1,000/day on
   Llama 3.3 70B, shared across *all* users of your key). This app includes a
   basic server-side queue/cache to make that budget go further — see
   "Free-tier limits" below.

## Deploying

**Backend (Render):**
1. Push this repo to GitHub.
2. On Render, create a new **Web Service** from the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn studybot.main:app --host 0.0.0.0 --port $PORT --app-dir src`
5. Add environment variables: `GROQ_API_KEY`, `ALLOWED_ORIGINS` (your GitHub
   Pages URL, e.g. `https://yourname.github.io`).
6. **Getting your materials onto the server.** `data/materials/` is
   gitignored by default (so you don't accidentally publish course content to
   a public repo). You have two options:
   - **Make the GitHub repo private** and remove the `data/materials/*` line
     from `.gitignore`, then commit your files. `render.yaml`'s build command
     already runs `python -m studybot.ingest` on every deploy, so this is the
     simplest path for a 100-student class.
   - **Keep materials out of git entirely**: use Render's Shell tab (or a
     persistent disk) to upload files directly to `data/materials/` on the
     server, then run `python -m studybot.ingest` manually from that shell.
     Note Render's free-tier disk is ephemeral on redeploy, so you'd redo this
     after any redeploy — a persistent disk (small paid add-on) avoids that
     if it becomes a hassle.

**Frontend (GitHub Pages):**
1. In the repo's GitHub settings → Pages, set source to the `docs/` folder on
   your main branch.
2. Edit `docs/app.js` → set `API_BASE_URL` to your Render backend URL.
3. Your chatbot is live at `https://yourname.github.io/studybot`.

## Free-tier limits — what to expect at 100 students

Groq's free tier applies rate limits per API key/org, not per student. At
current published limits (~30 requests/min, ~1,000/day on Llama 3.3 70B),
a burst of simultaneous questions (e.g. the night before a deadline) can hit
the ceiling. This repo mitigates that with:
- An **answer cache** (`cache.py`) — identical/near-identical questions are
  served from cache instead of hitting Groq again.
- A **request queue** that smooths bursts instead of throwing raw 429 errors
  at students.
- A friendly "we're at capacity, try again in a moment" fallback message.

If usage regularly exceeds the free tier, Groq's paid Developer tier removes
the card requirement barrier only (still pay-as-you-go, no subscription) and
gives ~10x the limits — worth knowing about even though this build defaults
to fully free.

## License

MIT — see `LICENSE`.
