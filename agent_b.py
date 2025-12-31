import os
import json
import logging
from typing import Dict, Any, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

SANDBOX_DIR = "sandbox"
LOGS_DIR = os.path.join(SANDBOX_DIR, "logs")
HANDOFF_PATH = os.path.join(SANDBOX_DIR, "handoff.json")

os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger("agent_b")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOGS_DIR, "agent_b.log"), encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)


GUARDED_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Agent B. Answer the question using ONLY the context provided.\n"
        "If the context does not contain the answer, say: \"I don't know based on the document.\"\n"
        "Keep the answer concise.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


class QueryAgent:
    def __init__(self):
        # You can keep gpt-4o-mini here; requirement only mandates GPT-5-mini for Agent A.
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = None
        self.qa_chain = None

    def load_handoff(self, handoff_path: str = HANDOFF_PATH) -> Dict[str, Any]:
        if not os.path.exists(handoff_path):
            raise FileNotFoundError(f"handoff.json not found at: {handoff_path}")
        with open(handoff_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Basic validation
        for k in ["file_path", "file_name", "source_url", "sha256", "bytes", "downloaded_at_iso"]:
            if k not in data:
                raise ValueError(f"handoff.json missing required field: {k}")

        file_path = data["file_path"]
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"Invalid file_path in handoff.json: {file_path}")

        return data

    def index_document_from_handoff(self, handoff: Dict[str, Any]) -> None:
        file_path = handoff["file_path"]
        logger.info(f"Agent B: Loading PDF: {file_path}")

        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()

        logger.info(f"Agent B: Loaded {len(docs)} chunks. Building embeddings + FAISS index...")
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": GUARDED_QA_PROMPT},
        )

        logger.info("Agent B: Indexing complete. Ready for queries.")

    def query(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {"answer": "System: No document indexed.", "sources": []}

        result = self.qa_chain.invoke({"query": question})
        answer = result.get("result", "")

        # Collect simple citations (page numbers) from source docs
        sources = []
        for d in result.get("source_documents", []) or []:
            meta = d.metadata or {}
            page = meta.get("page", None)
            src = meta.get("source", None)
            sources.append({"page": page, "source": src})

        # De-dup
        seen = set()
        uniq = []
        for s in sources:
            key = (s.get("page"), s.get("source"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)

        return {"answer": answer, "sources": uniq}


def run_cli():
    agent = QueryAgent()
    handoff = agent.load_handoff()
    agent.index_document_from_handoff(handoff)

    print("\nAgent B: Document is queryable. Ask questions. Type 'exit' to quit.\n")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        out = agent.query(q)
        print("\nAnswer:\n" + out["answer"])
        if out["sources"]:
            pages = [s["page"] for s in out["sources"] if s.get("page") is not None]
            if pages:
                pages = sorted(set(pages))
                print(f"\nSources (pages): {pages}")
        print()

if __name__ == "__main__":
    run_cli()
