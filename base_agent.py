import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "robochat"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=_get_embeddings(),
        )
    return _vectorstore


def build_llm(groq_api_key: str | None = None):
    key = groq_api_key if (groq_api_key and groq_api_key.strip()) else os.getenv("GROQ_API_KEY", "")

    primary = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=key,
        temperature=0.2,
        max_tokens=1024,
    )
    fallback = ChatOllama(
        model="llama3.2",
        temperature=0.2,
    )
    return primary.with_fallbacks([fallback])


def build_retriever(topics: list[str], k: int = 4):
    vs = get_vectorstore()
    if len(topics) == 1:
        search_kwargs = {"k": k, "filter": {"topic": topics[0]}}
    else:
        search_kwargs = {
            "k": k,
            "filter": {"topic": {"$in": topics}},
        }
    return vs.as_retriever(search_kwargs=search_kwargs)


def format_docs(docs: list[Document]) -> tuple[str, list[dict]]:
    parts: list[str] = []
    sources: list[dict] = []
    seen: set[str] = set()

    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] {doc.page_content}")
        if src not in seen:
            sources.append({
                "source": src,
                "topic": doc.metadata.get("topic", ""),
                "snippet": doc.page_content[:120].replace("\n", " ") + "...",
            })
            seen.add(src)

    return "\n\n".join(parts), sources
