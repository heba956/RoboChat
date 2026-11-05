from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from .base_agent import build_llm, build_retriever, format_docs

TOPICS = ["ros_docs", "book"]

SYSTEM_PROMPT = """You are an expert ROS2 debugger and robotics software engineer.
Analyze the user's code, error description, and prior conversation to provide a precise root-cause analysis
and a corrected code snippet where applicable.
Use the retrieved context for ROS2-specific details.

Retrieved context:
{context}
"""


def run(question: str, chat_history: list = None, groq_api_key: str | None = None) -> dict:
    history = chat_history or []
    retriever = build_retriever(TOPICS, k=5)
    docs = retriever.invoke(question)
    context, sources = format_docs(docs)

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
    return {"answer": answer, "sources": sources, "intent": "Debug_Code"}
