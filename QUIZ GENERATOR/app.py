"""
Quiz Paper Generator - Groq Powered
====================================
A web app that generates a 10-question quiz on any topic using the Groq API,
shows questions with answers, and lets you download a clean PDF question paper
(questions only, formatted like a real exam paper).

Run:
  pip install flask groq python-dotenv reportlab
  python app.py
  then open http://127.0.0.1:5000
"""

import io
import os
import re
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from flask import Flask, request, jsonify, send_file
    from groq import Groq
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    )
except ImportError as e:
    sys.exit(f"Missing dependency: {e.name}\nRun:  pip install flask groq python-dotenv reportlab")


app = Flask(__name__)

MODEL = "openai/gpt-oss-20b"

# --------------------------------------------------------------------------- #
# Groq question generation.
# --------------------------------------------------------------------------- #
GEN_PROMPT = """You are a quiz generator. Generate exactly 10 questions on the topic: "{topic}".
Difficulty: {difficulty}.
Return ONLY a JSON array (no markdown, no commentary) of objects with this shape:
{{"question": "...", "answer": "..."}}
Questions should be clear, factual, and varied (mix of definition, application, and reasoning).
Answers should be concise (1-3 sentences)."""


def parse_questions(raw: str) -> list[dict]:
    """Extract the JSON array from the model output, tolerating markdown fences."""
    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Find the first JSON array in the text.
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        if isinstance(data, list):
            return [
                {"question": str(q.get("question", "")),
                 "answer": str(q.get("answer", ""))}
                for q in data
            ]
    except json.JSONDecodeError:
        return []
    return []


def generate_quiz(topic: str, difficulty: str) -> list[dict]:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are a helpful quiz generator that always returns valid JSON."},
            {"role": "user",
             "content": GEN_PROMPT.format(topic=topic, difficulty=difficulty)},
        ],
        temperature=0.7,
    )
    raw = resp.choices[0].message.content
    questions = parse_questions(raw)
    if not questions:
        raise ValueError("Could not parse questions from the model response.")
    return questions[:10]


# --------------------------------------------------------------------------- #
# PDF question paper builder (questions only, exam-style).
# --------------------------------------------------------------------------- #
def build_pdf(topic: str, difficulty: str, questions: list[dict]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"], fontSize=18, spaceAfter=6, alignment=TA_CENTER
    )
    subtitle_style = ParagraphStyle(
        "Sub", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
        textColor="#555555", spaceAfter=4,
    )
    info_style = ParagraphStyle(
        "Info", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER,
        spaceAfter=2,
    )
    q_style = ParagraphStyle(
        "Q", parent=styles["Normal"], fontSize=11, leading=18, spaceBefore=10
    )
    answer_lines_style = ParagraphStyle(
        "Lines", parent=styles["Normal"], fontSize=10, textColor="#888888",
        spaceBefore=4, leading=20,
    )

    story = []
    story.append(Paragraph("Quiz Question Paper", title_style))
    story.append(Paragraph(f"Topic: {topic}", subtitle_style))
    story.append(Paragraph(f"Difficulty: {difficulty.capitalize()}  |  "
                           f"Total Questions: {len(questions)}", info_style))
    story.append(Paragraph(f"Time: 30 minutes   Max Marks: {len(questions) * 2}",
                           info_style))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1, color="#333333"))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Instructions: Answer all questions in the space "
                           "provided. Each question carries equal marks.",
                           ParagraphStyle("Instr", parent=styles["Normal"],
                                          fontSize=10, textColor="#444444",
                                          spaceAfter=10)))

    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"<b>Q{i}.</b> {q['question']}", q_style))
        # Three blank ruled lines for the student to write on.
        story.append(Paragraph("_" * 90, answer_lines_style))
        story.append(Paragraph("_" * 90, answer_lines_style))
        story.append(Paragraph("_" * 90, answer_lines_style))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="60%", thickness=0.5, color="#999999"))
    story.append(Paragraph("End of Question Paper", subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# --------------------------------------------------------------------------- #
# Routes.
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return HTML_PAGE


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    difficulty = (data.get("difficulty") or "medium").strip()
    if not topic:
        return jsonify({"error": "Please provide a topic."}), 400
    try:
        questions = generate_quiz(topic, difficulty)
        # Cache the last generated quiz in memory for the download endpoint.
        app.config["LAST_QUIZ"] = {
            "topic": topic, "difficulty": difficulty, "questions": questions
        }
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download")
def download():
    quiz = app.config.get("LAST_QUIZ")
    if not quiz:
        return jsonify({"error": "No quiz generated yet."}), 400
    pdf_bytes = build_pdf(quiz["topic"], quiz["difficulty"], quiz["questions"])
    safe_topic = re.sub(r"[^a-zA-Z0-9]+", "_", quiz["topic"])[:30].strip("_")
    filename = f"quiz_{safe_topic}.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


# --------------------------------------------------------------------------- #
# Single-file HTML page (no separate templates folder needed).
# --------------------------------------------------------------------------- #
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quiz Paper Generator</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --accent:#6366f1; --accent2:#22d3ee;
          --text:#e2e8f0; --muted:#94a3b8; --ok:#22c55e; --err:#ef4444; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
         color: var(--text); min-height: 100vh; padding: 24px; }
  .wrap { max-width: 880px; margin: 0 auto; }
  header { text-align: center; margin-bottom: 28px; }
  header h1 { font-size: 2rem; background: linear-gradient(90deg,var(--accent),var(--accent2));
              -webkit-background-clip: text; background-clip: text; color: transparent; }
  header p { color: var(--muted); margin-top: 6px; }
  .card { background: var(--card); border-radius: 14px; padding: 22px;
          box-shadow: 0 8px 24px rgba(0,0,0,.25); margin-bottom: 22px; }
  .form-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .form-row input[type=text] { flex: 1 1 320px; }
  input, select { background: #0f172a; border: 1px solid #334155; color: var(--text);
          padding: 12px 14px; border-radius: 8px; font-size: 15px; outline: none; }
  input:focus, select:focus { border-color: var(--accent); }
  button { background: var(--accent); color: #fff; border: none; padding: 12px 22px;
           border-radius: 8px; font-size: 15px; cursor: pointer; font-weight: 600;
           transition: transform .08s, background .2s; }
  button:hover { background: #4f46e5; }
  button:active { transform: scale(.97); }
  button:disabled { opacity: .55; cursor: not-allowed; }
  button.secondary { background: #334155; }
  button.secondary:hover { background: #475569; }
  .status { margin-top: 14px; color: var(--muted); font-size: 14px; }
  .error { color: var(--err); }
  .qa { border-left: 3px solid var(--accent); padding: 12px 16px; margin: 12px 0;
        background: #0f172a; border-radius: 0 8px 8px 0; }
  .qa .q { font-weight: 600; margin-bottom: 6px; }
  .qa .a { color: var(--muted); font-size: 14px; }
  .qa .a b { color: var(--ok); }
  .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
  .hidden { display: none; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--muted);
             border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  footer { text-align: center; color: var(--muted); font-size: 13px; margin-top: 30px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Quiz Paper Generator</h1>
    <p>Enter a topic - get 10 questions with answers + a downloadable PDF question paper.</p>
  </header>

  <div class="card">
    <div class="form-row">
      <input type="text" id="topic" placeholder="e.g. World War II, Python loops, Photosynthesis..." autofocus>
      <select id="difficulty">
        <option value="easy">Easy</option>
        <option value="medium" selected>Medium</option>
        <option value="hard">Hard</option>
      </select>
      <button id="genBtn" onclick="generate()">Generate</button>
    </div>
    <div class="status" id="status"></div>
  </div>

  <div id="results" class="hidden">
    <div class="card">
      <div class="toolbar">
        <button class="secondary" onclick="toggleAnswers()" id="ansBtn">Hide Answers</button>
        <button class="secondary" onclick="download()">Download Question Paper (PDF)</button>
      </div>
      <div id="qaList"></div>
    </div>
  </div>

  <footer>Powered by Groq API &middot; Task 3 - Quiz Generator</footer>
</div>

<script>
let showAnswers = true;

async function generate() {
  const topic = document.getElementById('topic').value.trim();
  const difficulty = document.getElementById('difficulty').value;
  const btn = document.getElementById('genBtn');
  const status = document.getElementById('status');
  const results = document.getElementById('results');

  if (!topic) { status.innerHTML = '<span class="error">Please enter a topic.</span>'; return; }

  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span> Generating 10 questions... please wait';
  results.classList.add('hidden');

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({topic, difficulty})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Request failed');

    const list = document.getElementById('qaList');
    list.innerHTML = data.questions.map((q, i) => `
      <div class="qa">
        <div class="q">Q${i+1}. ${escapeHtml(q.question)}</div>
        <div class="a"><b>Answer:</b> <span class="ans">${escapeHtml(q.answer)}</span></div>
      </div>`).join('');
    showAnswers = true;
    document.getElementById('ansBtn').textContent = 'Hide Answers';
    results.classList.remove('hidden');
    status.innerHTML = `<span style="color:var(--ok)">Done! ${data.questions.length} questions generated.</span>`;
  } catch (e) {
    status.innerHTML = '<span class="error">Error: ' + escapeHtml(e.message) + '</span>';
  } finally {
    btn.disabled = false;
  }
}

function toggleAnswers() {
  showAnswers = !showAnswers;
  document.querySelectorAll('.ans').forEach(el => {
    el.style.display = showAnswers ? '' : 'none';
  });
  document.getElementById('ansBtn').textContent = showAnswers ? 'Hide Answers' : 'Show Answers';
}

function download() {
  window.location.href = '/download';
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

document.getElementById('topic').addEventListener('keydown', e => {
  if (e.key === 'Enter') generate();
});
</script>
</body>
</html>"""


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        # Try reading .env directly as a fallback.
        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("GROQ_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if k and k != "your-key-here":
                        os.environ["GROQ_API_KEY"] = k
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("No GROQ_API_KEY found. Put it in a .env file next to this script.")
    print("\n  Quiz Paper Generator running at http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
