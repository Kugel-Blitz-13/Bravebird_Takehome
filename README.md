Bravebird Assignment – Multi-Agent PDF Download & Query System
Overview

This project demonstrates a two-agent system that cooperates to:

Download a PDF from Google Drive using a browser flow (no API usage)

Index the downloaded PDF and make it queryable via natural-language questions

The system focuses on:

Clean agent handoff

Explicit context management

Strong guardrails

Clear logging and reproducibility

Architecture
Agent A – Download Agent

Role: Downloads a Google Drive PDF using a real browser interaction

Executor: Playwright (Chromium)

Reasoner: gpt-5-mini

Capabilities:

Uses browser UI controls (no Drive API)

Retries on failure

Enforces a 10-minute execution limit

Logs every step

Stores all artifacts in a sandbox folder

Writes a clean handoff artifact with metadata

Agent B – Query Agent

Role: Indexes the downloaded PDF and answers questions about it

Capabilities:

Loads PDF from Agent A’s handoff

Builds embeddings and a FAISS vector index

Answers questions via a CLI

Includes guardrails to avoid hallucination

Project Structure
bravebird_demo/
├── agent_a.py              # Agent A: browser-based download agent
├── agent_b.py              # Agent B: PDF indexing + query agent
├── main.py                 # Orchestrates Agent A → Agent B
├── requirements.txt
└── sandbox/
    ├── downloads/          # Downloaded PDFs
    ├── logs/               # Agent logs
    ├── screenshots/        # Error screenshots
    └── handoff.json        # Agent A → Agent B context

    (OPEN AI KEY directly declared in terminal)

Requirements Mapping
Assignment Requirement	How It’s Met
Browser-based download (no API)	Playwright simulates real Drive UI interactions
Uses GPT-5-mini	Agent A planner explicitly calls gpt-5-mini
Completes within 10 minutes	asyncio.wait_for(..., 600) + internal guard
Retries on failure	Tenacity retry with exponential backoff
Stores file in sandbox	sandbox/downloads/
Logs each step	sandbox/logs/agent_a.log
Clean agent handoff	sandbox/handoff.json with metadata
Indexes PDF	PyPDFLoader + embeddings + FAISS
Queryable interface	Interactive CLI
Context management & guardrails	Strict action planner + retrieval-only answering
Reproducible	Single command run + pinned dependencies
Setup Instructions
1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

2. Install dependencies
pip install -r requirements.txt
python -m playwright install chromium

3. Set OpenAI API key

OPENAI_API_KEY=your_api_key_here

Running the System
Step 1: Obtain a valid Google Drive PDF link

The file must be a PDF

Sharing must be set to Anyone with the link → Viewer

Link format:

https://drive.google.com/file/d/FILE_ID/view?usp=sharing

Step 2: Run the system
python main.py --url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"

Expected Runtime Behavior

Chromium browser opens Google Drive

Agent A:

Uses GPT-5-mini to decide which UI action to take

Clicks Download

Saves PDF to sandbox/downloads/

Writes sandbox/handoff.json

Agent B:

Reads handoff.json

Indexes PDF

Starts interactive CLI

User can ask questions about the PDF

Example Output
Planner(gpt-5-mini): {'action': 'CLICK_DOWNLOAD_BUTTON', ...}
Saved to: sandbox/downloads/Bravebird Assignment.pdf
Handoff written to: sandbox/handoff.json
Agent B: Document is ready. Ask me anything about it.

Notes

No Google Drive APIs are used

All state is explicit and stored locally

Agents communicate only via a structured handoff artifact
