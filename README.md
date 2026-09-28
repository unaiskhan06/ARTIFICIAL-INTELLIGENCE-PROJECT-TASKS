# EMATIX INTERNSHIP - AI/ML Project Portfolio

This repository contains 6 tasks completed during the internship, covering
email classification, chatbot development, quiz generation, Hugging Face
models, AutoGen multi-agent systems, and CNN-based face recognition.

---

## Task 1 - Email Classification (Spam vs Ham)
**Folder:** `task 1 EMAIL CLASSIFICATION/`
**File:** `Task 1 email_classifier.py`

A machine learning model that classifies emails as Spam or Not Spam (Ham)
using TF-IDF text features and a Multinomial Naive Bayes classifier.

**Features:**
- Spam / Not-Spam classification with confidence score
- Built-in labeled training dataset (runs out of the box)
- Heuristic signals blended into features (URL detection, currency symbols,
  uppercase ratio, exclamation marks)
- Per-email explanation of why it was flagged
- Three modes: demo, single email (`--email`), batch file (`--batch`)
- Optional custom CSV training data (`--train`)

**How to run:**
```powershell
python "task 1 EMAIL CLASSIFICATION/Task 1 email_classifier.py"
python "task 1 EMAIL CLASSIFICATION/Task 1 email_classifier.py" --email "Win free money now!"
```

**Tech:** Python, scikit-learn, TF-IDF, Naive Bayes

---

## Task 2 - Groq Chatbot (Terminal)
**Folder:** `task 2 groq chatbot/`
**File:** `task 2 chatbot.py`

An interactive terminal chatbot powered by the Groq API with streaming
responses (tokens appear as they generate).

**Features:**
- Streaming responses (live token generation)
- Switch between Groq models on the fly (`/model` command)
- Customizable system prompt (`/system`)
- Conversation history with token usage tracking
- Save transcript to JSON (`/save`)
- Slash commands: `/help`, `/model`, `/system`, `/clear`, `/history`, `/save`, `/exit`
- Colorized terminal output (ANSI, Windows-compatible)

**Setup:**
1. Get a free API key at https://console.groq.com/keys
2. Paste it in the `.env` file: `GROQ_API_KEY=your-key`

**How to run:**
```powershell
python "task 2 groq chatbot/task 2 chatbot.py"
```

**Tech:** Python, Groq API, python-dotenv

---

## Task 3 - Quiz Paper Generator (Web UI)
**Folder:** `task 3 quiz generator/`
**File:** `app.py`

A web app that generates 10 questions on any topic using the Groq API,
shows questions with answers, and lets you download a clean PDF question
paper (questions only, formatted like a real exam).

**Features:**
- Enter any topic + difficulty level -> generates 10 Q&A pairs
- Hide/Show answers toggle
- Download Question Paper as PDF (exam-style with header, instructions,
  numbered questions, ruled answer lines)
- Dark, modern web UI with streaming generation status

**Setup:**
1. Groq API key in `.env` file
2. Install: `pip install flask groq python-dotenv reportlab`

**How to run:**
```powershell
python "task 3 quiz generator/app.py"
```
Then open http://127.0.0.1:5000 in your browser.

**Tech:** Flask, Groq API, ReportLab (PDF generation), HTML/CSS/JS

---

## Task 4 - Hugging Face Models + AutoGen Multi-Agent System
**Folder:** `task 4 hf autogen/`
**Files:**
- `text_gen.py` - Part 1: Text generation with Hugging Face
- `image_gen.py` - Part 2: Image generation via HF Inference API
- `autogen_study_notes.py` - Part 3: AutoGen basics study notes
- `autogen_research.py` - Part 4: Multi-agent research system

### Part 1: Text Generation
Uses the open-source `distilgpt2` model from Hugging Face. Runs locally
on CPU (no GPU needed).
```powershell
python "task 4 hf autogen/text_gen.py" --prompt "The future of AI is"
```

### Part 2: Image Generation
Generates images from text prompts using FLUX.1-schnell via HuggingFace
Inference API (runs on HF servers).
```powershell
python "task 4 hf autogen/image_gen.py" "a cat astronaut on Mars"
```
Requires a free HuggingFace token in `.env`: `HF_TOKEN=hf_xxxxx`

### Part 3: AutoGen Study Notes
Documents AutoGen's architecture, key components (AssistantAgent, Teams,
Termination conditions, Model clients), and the reflection pattern.

### Part 4: Multi-Agent Research System
A 4-agent pipeline powered by Groq + AutoGen:
```
ResearchAgent -> AnalysisAgent -> ReviewAgent -> FinalAgent
   gathers facts    synthesizes     reviews quality   writes final answer
```
- RoundRobinGroupChat (agents take turns)
- ReviewAgent says "APPROVE" when satisfied -> system stops
- Retry logic for Groq free-tier rate limits
```powershell
python "task 4 hf autogen/autogen_research.py" --topic "Benefits of renewable energy"
```

**Tech:** PyTorch, Transformers, HuggingFace Hub, AutoGen, Groq API

---

## Task 5 - Multimodal AutoGen Agent (FastAPI + Flask)
**Folder:** `task 5 multimodal agent/`
**Files:**
- `api.py` - FastAPI backend (BLIP image captioning + AutoGen agent team)
- `app.py` - Flask UI (upload image + ask question)

A multimodal AI agent system that processes both text and images. The
image is analyzed by a BLIP vision model, then a team of 4 AutoGen agents
collaborate to answer the user's question.

**Architecture:**
```
User (browser) -> Flask UI (port 5000) -> FastAPI backend (port 8000)
                                              |
                                    Image uploaded? -> BLIP model -> text caption
                                              |
                                    AutoGen agent team (Groq-powered):
                                      VisionAgent -> ResearchAgent -> AnalysisAgent -> FinalAgent
                                              |
                                    JSON response back to UI
```

**Features:**
- Upload an image + ask a question (or text-only)
- BLIP model (runs locally) generates image description
- 4 AutoGen agents collaborate in round-robin
- Each agent's response displayed in the UI
- Image preview before submission

**Setup:**
1. Groq API key + HuggingFace token in `.env`
2. Install: `pip install fastapi uvicorn flask python-multipart autogen-agentchat autogen-ext[openai]`

**How to run (two terminals):**
```powershell
# Terminal 1 - backend
python "task 5 multimodal agent/api.py"

# Terminal 2 - UI
python "task 5 multimodal agent/app.py"
```
Then open http://127.0.0.1:5000

**Tech:** FastAPI, Flask, AutoGen, HuggingFace BLIP, Groq API, PyTorch

---

## Task 6 - CNN-Based Face Recognition System
**Folder:** `task 6 cnn face recognition/`
**Files:**
- `cnn_face_recognition.py` - Main CNN model + training + visualization
- `cnn_explanation.txt` - Layer-by-layer explanation for review
- `output_plots/` - Generated visualization images

A Convolutional Neural Network that recognizes faces using the Olivetti
Faces dataset (400 images, 40 people). Achieves 92.5% test accuracy.

**CNN Architecture (6 layers):**
1. **Input Layer** - receives 64x64 grayscale face image
2. **Convolution Layer** - extracts features (edges, textures, facial parts)
3. **Max Pooling Layer** - downsamples, keeps important features
4. **Flatten Layer** - converts 2D feature maps to 1D vector
5. **Dense Layer** - fully-connected reasoning (combines features)
6. **Output Layer** - 40 neurons, one per person (classification)

**Visualizations generated:**
- `01_sample_faces.png` - sample faces from the dataset
- `02_training_progress.png` - loss & accuracy curves over 15 epochs
- `03_single_prediction.png` - a face with prediction + confidence
- `04_conv_filters.png` - the learned 3x3 convolution filters
- `05_feature_maps.png` - how the image transforms through each layer

**How to run:**
```powershell
python "task 6 cnn face recognition/cnn_face_recognition.py"
```

**Results:**
- Test accuracy: 92.5%
- Total parameters: 1,063,432
- Training: 15 epochs, batch size 32, Adam optimizer

**Tech:** PyTorch, scikit-learn (Olivetti dataset), Matplotlib

---

## Dependencies

Install all dependencies at once:
```powershell
pip install scikit-learn pandas groq python-dotenv flask reportlab fastapi uvicorn python-multipart torch transformers huggingface_hub pillow matplotlib "autogen-agentchat" "autogen-ext[openai]"
```

**Python version:** 3.14

## API Keys Required

| Key | Where to get it | Used in |
|-----|----------------|---------|
| Groq API Key | https://console.groq.com/keys | Tasks 2, 3, 4, 5 |
| HuggingFace Token | https://huggingface.co/settings/tokens | Tasks 4, 5 |

Store keys in a `.env` file in each task folder:
```
GROQ_API_KEY=gsk_your_key_here
HF_TOKEN=hf_your_token_here
```

## Project Structure
```
EMATIX INTERNSHIP/
|-- task 1 EMAIL CLASSIFICATION/
|   `-- Task 1 email_classifier.py
|-- task 2 groq chatbot/
|   |-- task 2 chatbot.py
|   |-- .env
|   `-- .env.example
|-- task 3 quiz generator/
|   |-- app.py
|   `-- .env
|-- task 4 hf autogen/
|   |-- text_gen.py
|   |-- image_gen.py
|   |-- autogen_study_notes.py
|   |-- autogen_research.py
|   |-- .env
|   `-- generated_images/
|-- task 5 multimodal agent/
|   |-- api.py          (FastAPI backend)
|   |-- app.py          (Flask UI)
|   `-- .env
`-- task 6 cnn face recognition/
    |-- cnn_face_recognition.py
    |-- cnn_explanation.txt
    `-- output_plots/
        |-- 01_sample_faces.png
        |-- 02_training_progress.png
        |-- 03_single_prediction.png
        |-- 04_conv_filters.png
        `-- 05_feature_maps.png
```
