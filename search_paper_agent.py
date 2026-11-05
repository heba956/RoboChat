import urllib.parse
import xml.etree.ElementTree as ET
import requests
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from .base_agent import build_llm


def fetch_arxiv_papers(query: str, max_results: int = 5) -> tuple[str, list[dict]]:
    clean_query = query.lower()
    for phrase in [
        "search for papers on",
        "search for papers about",
        "find papers on",
        "find papers about",
        "search papers on",
        "search arxiv for",
        "looking for publications on",
        "looking for papers on",
        "give me a list of papers about",
        "give me papers on",
        "papers about",
        "papers on",
        "search for",
        "search",
    ]:
        clean_query = clean_query.replace(phrase, " ")
    clean_query = " ".join(clean_query.split())
    clean_query = clean_query.strip(" ?.,!:\n\t")
    if not clean_query:
        clean_query = query

    encoded_query = urllib.parse.quote_plus(clean_query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

    try:
        resp = requests.get(url, timeout=12)
        if resp.status_code != 200:
            return "", []
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "", []

        context_parts = []
        sources = []
        for i, entry in enumerate(entries, 1):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)

            title = title_el.text.strip().replace("\n", " ") if title_el is not None else "Untitled"
            summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
            link = id_el.text.strip() if id_el is not None else ""
            pub_date = published_el.text.strip()[:10] if published_el is not None else ""

            authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
            authors_str = ", ".join(authors[:4])
            if len(authors) > 4:
                authors_str += " et al."

            context_parts.append(
                f"Paper [{i}]:\nTitle: {title}\nAuthors: {authors_str}\nDate: {pub_date}\nLink: {link}\nAbstract: {summary}\n"
            )
            sources.append({
                "source": title,
                "topic": "arXiv",
                "snippet": f"{authors_str} ({pub_date}) - {link}",
            })

        return "\n\n".join(context_parts), sources
    except Exception:
        return "", []


SYSTEM_PROMPT = """You are a robotics research assistant.
The user is searching for academic papers.
Below are live search results fetched directly from arXiv:

{context}

Present the relevant papers clearly with:
- Title and clickable markdown link to the arXiv page
- Authors and publication date
- A concise summary of their contribution / method
If the user asks specific questions, answer them using the retrieved papers.
"""


def run(question: str, chat_history: list = None, groq_api_key: str | None = None) -> dict:
    history = chat_history or []
    context, sources = fetch_arxiv_papers(question, max_results=5)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    llm = build_llm(groq_api_key)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "chat_history": history,
        "question": question,
    })
    return {"answer": answer, "sources": sources, "intent": "Search_Paper"}
