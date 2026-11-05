from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from .base_agent import build_llm

SYSTEM_PROMPT = """You are RoboChat, a helpful and friendly robotics teaching assistant.
The user is making small talk or asking something outside robotics and ROS2.
Keep your reply brief, warm, and redirect gently toward robotics topics if appropriate."""


def run(question: str, chat_history: list = None, groq_api_key: str | None = None) -> dict:
    history = chat_history or []
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    llm = build_llm(groq_api_key)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "chat_history": history,
        "question": question,
    })
    return {"answer": answer, "sources": [], "intent": "Chitchat_Other"}
