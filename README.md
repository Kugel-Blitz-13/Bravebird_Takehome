# Bravebird Assignment – Multi-Agent PDF Download & Query System

## Overview

This project demonstrates a **two-agent system** that cooperates to:

1. Download a PDF from Google Drive using a **browser-based flow (no API usage)**
2. Index the downloaded PDF and make it **queryable using natural language**

The system emphasizes:
- Clean **agent handoff**
- Explicit **context management**
- Strong **guardrails**
- Clear **logging and reproducibility**

---

## Architecture

### Agent A – Download Agent
- **Role**: Downloads a Google Drive PDF using real browser interactions
- **Executor**: Playwright (Chromium)
- **Reasoner**: `gpt-5-mini`
- **Key Features**:
  - Uses Drive UI controls (no Google Drive API)
  - Retries on failure
  - Enforces a strict 10-minute execution limit
  - Logs every step
  - Stores all artifacts in a sandbox directory
  - Writes a structured handoff artifact with metadata

### Agent B – Query Agent
- **Role**: Indexes the downloaded PDF and answers questions about its content
- **Key Features**:
  - Reads the file path and metadata from Agent A’s handoff
  - Builds embeddings and a FAISS vector index
  - Answers questions via a CLI interface
  - Uses retrieval-only guardrails to avoid hallucinations

---

### Sample Output:

Planner(gpt-5-mini): {'action': 'CLICK_DOWNLOAD_BUTTON', ...}
Saved to: sandbox/downloads/Bravebird Assignment.pdf
Handoff written to: sandbox/handoff.json
Agent B: Document is ready. Ask me anything about it.
