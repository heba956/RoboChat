from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from .base_agent import build_llm, build_retriever, format_docs

TOPICS = ["ros_docs", "book", "websites"]

SYSTEM_PROMPT = """You are a friendly robotics and ROS2 teaching assistant.
Use the retrieved context and prior conversation to answer clearly and accurately.
If the context does not cover the question, say so and give a brief general answer.

Retrieved context:
{context}
"""


def run(question: str, chat_history: list = None, groq_api_key: str | None = None) -> dict:
    history = chat_history or []
    retriever = build_retriever(TOPICS, k=4)
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
    return {"answer": answer, "sources": sources, "intent": "Explain_Concept"}
