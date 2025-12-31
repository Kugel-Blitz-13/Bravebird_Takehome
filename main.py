import argparse
import asyncio
import sys
import json
import os

from agent_a import DownloadAgent, HANDOFF_PATH
from agent_b import QueryAgent

async def main(url: str):
    print("--- Bravebird System Initialized ---")

    # --- PHASE 1: Agent A (Download) ---
    downloader = DownloadAgent()

    try:
        print("Agent A: Launching browser download flow...")
        # Hard 10-min cap (requirement)
        await asyncio.wait_for(downloader.run(url, max_seconds=600), timeout=600)
    except asyncio.TimeoutError:
        print("System Error: Agent A timed out (>10 mins).")
        sys.exit(1)
    except Exception as e:
        print(f"System Error: Agent A failed. {e}")
        sys.exit(1)

    # --- PHASE 2: Agent B (Query) ---
    analyst = QueryAgent()

    try:
        if not os.path.exists(HANDOFF_PATH):
            raise FileNotFoundError(f"Expected handoff.json at {HANDOFF_PATH} but not found.")

        with open(HANDOFF_PATH, "r", encoding="utf-8") as f:
            handoff = json.load(f)

        print("\n--- Handover Successful ---")
        print(f"Handoff file: {HANDOFF_PATH}")
        print(f"File path: {handoff.get('file_path')}")
        print(f"File name: {handoff.get('file_name')}")
        print(f"SHA256: {handoff.get('sha256')}\n")

        analyst.index_document_from_handoff(handoff)

    except Exception as e:
        print(f"Agent B Error: Failed to index document. {e}")
        sys.exit(1)

    # --- INTERACTIVE LOOP (CLI) ---
    print("Agent B: The document is ready. Ask me anything about it.")
    print("(Type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Shutting down.")
            break

        out = analyst.query(user_input)
        print(f"\nAgent B: {out['answer']}")
        if out["sources"]:
            pages = [s["page"] for s in out["sources"] if s.get("page") is not None]
            if pages:
                pages = sorted(set(pages))
                print(f"Sources (pages): {pages}")
        print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Public Google Drive PDF link (browser download flow, no API).")
    args = ap.parse_args()
    asyncio.run(main(args.url))
