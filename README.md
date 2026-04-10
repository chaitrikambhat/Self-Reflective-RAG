# Self-Reflective RAG

A local RAG system that critiques its own answers and regenerates them when quality is low.

Built with Python (Flask) + Groq.

---

## Setup

### 1. Install dependencies

```bash
pip install flask flask-cors groq
```

### 2. Get your free Groq API key

1. Go to https://console.groq.com and sign up
2. Click **API Keys** in the sidebar
3. Click **Create API Key** and copy it

### 3. Set your API key

**Mac / Linux:**
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

**Windows:**
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
```

### 4. Run the server

```bash
python app.py
```

### 5. Open in browser

Go to: **http://127.0.0.1:5000**

> Note: Use `127.0.0.1:5000` not `localhost:5000` — some browsers block localhost by default.

---

## How it works

```
User Query
    ↓
Retriever  ──  keyword-based chunk scoring
    ↓
Answer Generator  ──  LLM drafts answer from context
    ↓
Critic Model  ──  scores 0-100, flags grounding / hallucination
    ↓
Score >= threshold?
    YES  →  return final answer
    NO   →  inject feedback  →  regenerate (max 3 attempts)
```

---

## API endpoint

`POST /api/run`

**Request:**
```json
{
  "query":          "Your question",
  "knowledge_base": "Your documents",
  "threshold":      70
}
```

**Response:**
```json
{
  "chunks_retrieved": 3,
  "context": "...",
  "attempts": [
    {
      "attempt": 1,
      "answer": "...",
      "score": 62,
      "grounded": true,
      "complete": false,
      "hallucination": false,
      "feedback": "Answer is missing key information about X."
    },
    {
      "attempt": 2,
      "answer": "...",
      "score": 88,
      "grounded": true,
      "complete": true,
      "hallucination": false,
      "feedback": "Answer is accurate and well-supported."
    }
  ],
  "final_answer": "...",
  "final_score": 88,
  "passed": true,
  "total_attempts": 2
}
```

---

## Deploying to the web (Render)

1. Push these files to a GitHub repository
2. Add a `requirements.txt` file:
   ```
   flask
   flask-cors
   groq
   gunicorn
   ```
3. Sign up at https://render.com with your GitHub account
4. New → Web Service → connect your repo
5. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Instance type: Free
6. Add environment variable: `GROQ_API_KEY` = your key
7. Deploy — your public URL will be `https://your-app.onrender.com`

---

## Extending this project

- **Better retrieval** — replace `retrieve_chunks()` with vector embeddings using `sentence-transformers` + FAISS
- **Real documents** — use `pypdf` or `docx2txt` to load actual files into the knowledge base
- **Persistent storage** — add SQLite to store questions, answers, and critique history
- **Streaming** — use `client.chat.completions.create(stream=True)` for real-time token streaming
