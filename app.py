"""
Self-Reflective RAG - Backend Server (Groq Edition - FREE)
===========================================================
Run:  python app.py
Then: open http://localhost:5000 in your browser

Requirements:
    pip install flask flask-cors groq

Get your FREE Groq API key at: https://console.groq.com
No credit card needed. ~14,400 free requests/day.

Set your key:
    Mac/Linux:  export GROQ_API_KEY="gsk_your_key_here"
    Windows:    set GROQ_API_KEY=gsk_your_key_here
"""

import os
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from groq import Groq

app = Flask(__name__, static_folder=".")
CORS(app)

# ── Groq client (FREE) ────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

# Available free Groq models (pick one):
#   "llama-3.3-70b-versatile"   <- best quality (recommended)
#   "llama-3.1-8b-instant"      <- fastest, lower quality
#   "mixtral-8x7b-32768"        <- good for long contexts
MODEL = "llama-3.3-70b-versatile"

MAX_ATTEMPTS = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_llm(system: str, user: str) -> str:
    """Single call to the Groq API, returns response text."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def retrieve_chunks(knowledge_base: str, query: str):
    """
    Simple keyword-based retriever.
    In production, replace with vector similarity search
    (e.g. sentence-transformers + FAISS).
    """
    sentences = re.split(r"(?<=[.!?])\s+", knowledge_base.strip())
    query_words = set(w.lower() for w in re.split(r"\W+", query) if len(w) > 3)

    scored = []
    for sent in sentences:
        sent_words = set(w.lower() for w in re.split(r"\W+", sent))
        overlap = len(query_words & sent_words)
        if overlap > 0:
            scored.append((overlap, sent))

    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:5]] if scored else sentences
    return " ".join(top), len(top)


def generate_answer(context: str, query: str, previous_feedback: str = "") -> str:
    """LLM call to generate (or regenerate) an answer from retrieved context."""
    feedback_block = ""
    if previous_feedback:
        feedback_block = (
            f"\n\nIMPORTANT - The previous answer was critiqued. "
            f"Address this feedback in your new answer:\n{previous_feedback}"
        )

    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context. Be concise and factual. Do not invent or assume any "
        "information that is not explicitly present in the context."
        + feedback_block
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return call_llm(system, user)


def critique_answer(context: str, query: str, answer: str) -> dict:
    """
    LLM call to score and critique the generated answer.
    Returns a dict with: score, grounded, complete, hallucination, feedback.
    """
    system = """You are a strict answer quality critic for a RAG system.

Evaluate whether the answer is well-supported by the provided context.

Respond ONLY with a valid JSON object - no markdown fences, no extra text.
Use exactly this schema:
{
  "score": <integer 0-100>,
  "grounded": <true if every claim traces to the context, else false>,
  "complete": <true if the answer covers all key info relevant to the question, else false>,
  "hallucination": <true if the answer contains info NOT in the context, else false>,
  "feedback": "<1-2 sentences: what is wrong and how to fix it, or 'Answer is accurate and well-supported.' if score >= 85>"
}

Score rubric:
  85-100 : Fully grounded, complete, no hallucination
  65-84  : Minor omission or slight imprecision
  40-64  : Significant missing info or minor hallucination
  0-39   : Major hallucination or completely off-topic"""

    user = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer to evaluate:\n{answer}"
    )

    raw = call_llm(system, user)

    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        score_match    = re.search(r'"score"\s*:\s*(\d+)', raw)
        feedback_match = re.search(r'"feedback"\s*:\s*"([^"]+)"', raw)
        return {
            "score":         int(score_match.group(1)) if score_match else 50,
            "grounded":      True,
            "complete":      True,
            "hallucination": False,
            "feedback":      feedback_match.group(1) if feedback_match else "Could not parse critic feedback.",
        }


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    data      = request.get_json()
    query     = data.get("query", "").strip()
    kb        = data.get("knowledge_base", "").strip()
    threshold = int(data.get("threshold", 70))

    if not query or not kb:
        return jsonify({"error": "Both 'query' and 'knowledge_base' are required."}), 400

    context, chunks_found = retrieve_chunks(kb, query)

    attempts          = []
    previous_feedback = ""
    final_answer      = ""
    final_score       = 0

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        answer   = generate_answer(context, query, previous_feedback)
        critique = critique_answer(context, query, answer)
        score    = max(0, min(100, int(critique.get("score", 0))))

        attempts.append({
            "attempt":       attempt_num,
            "answer":        answer,
            "score":         score,
            "grounded":      critique.get("grounded", True),
            "complete":      critique.get("complete", True),
            "hallucination": critique.get("hallucination", False),
            "feedback":      critique.get("feedback", ""),
        })

        final_answer = answer
        final_score  = score

        if score >= threshold:
            break

        previous_feedback = critique.get("feedback", "")

    return jsonify({
        "chunks_retrieved": chunks_found,
        "context":          context,
        "attempts":         attempts,
        "final_answer":     final_answer,
        "final_score":      final_score,
        "passed":           final_score >= threshold,
        "total_attempts":   len(attempts),
    })


@app.route("/api/health", methods=["GET"])
def health():
    key_set = bool(os.environ.get("GROQ_API_KEY"))
    return jsonify({"status": "ok", "api_key_set": key_set, "model": MODEL})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("\n  GROQ_API_KEY environment variable not set.")
        print("   Get your free key at: https://console.groq.com")
        print("   Then run:  export GROQ_API_KEY='gsk_your_key_here'\n")
    print(f"Starting Self-Reflective RAG server at http://127.0.0.1.5000  (model: {MODEL})")
    app.run(debug=True, port=5000, host="127.0.0.1")
