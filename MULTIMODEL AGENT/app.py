"""
Task 5 - Flask UI: Multimodal Agent Frontend
=============================================
A web UI that connects to the FastAPI backend.
Users can upload an image + ask a question, or just ask a text question.
The multi-agent team's responses are displayed step by step.

Run:
  1. Start the backend first:  python api.py   (port 8000)
  2. Then start this UI:        python app.py   (port 5000)
  3. Open http://127.0.0.1:5000
"""

import io
import os
import sys
import requests
from pathlib import Path

from flask import Flask, request, render_template_string

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000"

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multimodal AI Agent</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --accent:#6366f1; --accent2:#22d3ee;
          --text:#e2e8f0; --muted:#94a3b8; --ok:#22c55e; --err:#ef4444; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
         color: var(--text); min-height: 100vh; padding: 24px; }
  .wrap { max-width: 900px; margin: 0 auto; }
  header { text-align: center; margin-bottom: 28px; }
  header h1 { font-size: 2rem; background: linear-gradient(90deg,var(--accent),var(--accent2));
              -webkit-background-clip: text; background-clip: text; color: transparent; }
  header p { color: var(--muted); margin-top: 6px; }
  .card { background: var(--card); border-radius: 14px; padding: 22px;
          box-shadow: 0 8px 24px rgba(0,0,0,.25); margin-bottom: 22px; }
  .form-group { margin-bottom: 16px; }
  label { display: block; margin-bottom: 6px; color: var(--muted); font-size: 14px; }
  input[type=text], textarea { width: 100%; background: #0f172a; border: 1px solid #334155;
        color: var(--text); padding: 12px 14px; border-radius: 8px; font-size: 15px; outline: none; }
  input[type=text]:focus, textarea:focus { border-color: var(--accent); }
  input[type=file] { color: var(--muted); }
  .preview { margin-top: 10px; max-width: 300px; border-radius: 8px; display: none; }
  button { background: var(--accent); color: #fff; border: none; padding: 12px 28px;
           border-radius: 8px; font-size: 16px; cursor: pointer; font-weight: 600;
           transition: transform .08s, background .2s; width: 100%; }
  button:hover { background: #4f46e5; }
  button:active { transform: scale(.98); }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .status { margin-top: 14px; color: var(--muted); text-align: center; }
  .error { color: var(--err); }
  .agent-msg { border-left: 3px solid var(--accent); padding: 14px 16px; margin: 12px 0;
               background: #0f172a; border-radius: 0 8px 8px 0; }
  .agent-name { font-weight: 700; color: var(--accent2); margin-bottom: 6px; font-size: 14px; }
  .agent-content { white-space: pre-wrap; font-size: 14px; line-height: 1.6; }
  .img-desc { background: #1a2e1a; border-left: 3px solid var(--ok); padding: 12px 16px;
              margin: 12px 0; border-radius: 0 8px 8px 0; font-size: 14px; }
  .img-desc b { color: var(--ok); }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid var(--muted);
             border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite;
             vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .hidden { display: none; }
  footer { text-align: center; color: var(--muted); font-size: 13px; margin-top: 30px; }
  .agents-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .agent-tag { background: #334155; padding: 4px 12px; border-radius: 20px; font-size: 12px;
               color: var(--muted); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Multimodal AI Agent</h1>
    <p>Upload an image and ask a question - multiple AI agents will collaborate to answer</p>
  </header>

  <div class="card">
    <div class="agents-bar">
      <span class="agent-tag">VisionAgent</span>
      <span class="agent-tag">ResearchAgent</span>
      <span class="agent-tag">AnalysisAgent</span>
      <span class="agent-tag">FinalAgent</span>
    </div>

    <form id="form" enctype="multipart/form-data">
      <div class="form-group">
        <label>Upload an image (optional)</label>
        <input type="file" id="image" name="image" accept="image/*"
               onchange="previewImage(this)">
        <img id="preview" class="preview" />
      </div>
      <div class="form-group">
        <label>Your question</label>
        <textarea id="query" name="query" rows="2"
                  placeholder="Ask anything about the image, or any question..."></textarea>
      </div>
      <button type="submit" id="btn">Ask Agents</button>
    </form>
    <div class="status" id="status"></div>
  </div>

  <div id="results" class="hidden">
    <div class="card" id="resultCard"></div>
  </div>

  <footer>Powered by AutoGen + Groq + HuggingFace &middot; FastAPI backend + Flask UI</footer>
</div>

<script>
function previewImage(input) {
  const preview = document.getElementById('preview');
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = e => { preview.src = e.target.result; preview.style.display = 'block'; };
    reader.readAsDataURL(input.files[0]);
  } else {
    preview.style.display = 'none';
  }
}

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  const resultCard = document.getElementById('resultCard');

  const query = document.getElementById('query').value.trim();
  const imageFile = document.getElementById('image').files[0];

  if (!query) { status.innerHTML = '<span class="error">Please enter a question.</span>'; return; }

  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span> Agents are collaborating... this may take 20-40s';
  results.classList.add('hidden');

  const formData = new FormData();
  formData.append('query', query);
  if (imageFile) formData.append('image', imageFile);

  try {
    const res = await fetch('http://127.0.0.1:8000/api/multimodal', {
      method: 'POST', body: formData
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Request failed');

    let html = '';
    if (data.image_description) {
      html += '<div class="img-desc"><b>Image Description (BLIP):</b> ' +
              escapeHtml(data.image_description) + '</div>';
    }
    data.messages.forEach(m => {
      html += '<div class="agent-msg"><div class="agent-name">' + m.agent +
              '</div><div class="agent-content">' + escapeHtml(m.content) + '</div></div>';
    });
    resultCard.innerHTML = html;
    results.classList.remove('hidden');
    status.innerHTML = '<span style="color:var(--ok)">Done! ' + data.messages.length +
                       ' agent responses.</span>';
  } catch (err) {
    status.innerHTML = '<span class="error">Error: ' + escapeHtml(err.message) +
                       '<br>Make sure the FastAPI backend is running (python api.py)</span>';
  } finally {
    btn.disabled = false;
  }
});

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/health")
def health():
    try:
        r = requests.get(f"{FASTAPI_URL}/", timeout=5)
        return {"backend": "up", "detail": r.json()}
    except Exception:
        return {"backend": "down", "detail": "FastAPI backend not running"}


if __name__ == "__main__":
    print("\n  Flask UI running at http://127.0.0.1:5000")
    print("  Make sure FastAPI backend is also running: python api.py\n")
    app.run(debug=False, port=5000)
