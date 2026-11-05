import os
from langchain_core.runnables import RunnableLambda, RunnableBranch

from intent_inference import IntentPredictor
from agents import explain_agent, debug_agent, search_paper_agent, chitchat_agent

_predictor: IntentPredictor | None = None

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "intent_classifier.pt")


def _get_predictor() -> IntentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = IntentPredictor(CHECKPOINT, device="cpu")
    return _predictor


def _make_agent_runnable(agent_module):
    return RunnableLambda(
        lambda payload, mod=agent_module: mod.run(
            question=payload["question"],
            chat_history=payload.get("chat_history"),
            groq_api_key=payload.get("groq_api_key"),
        )
    )


_explain_runnable = _make_agent_runnable(explain_agent)
_debug_runnable = _make_agent_runnable(debug_agent)
_search_runnable = _make_agent_runnable(search_paper_agent)
_chitchat_runnable = _make_agent_runnable(chitchat_agent)

_branch = RunnableBranch(
    (lambda p: p["intent"] == "Explain_Concept", _explain_runnable),
    (lambda p: p["intent"] == "Debug_Code",      _debug_runnable),
    (lambda p: p["intent"] == "Search_Paper",    _search_runnable),
    _chitchat_runnable,
)


def route(question: str, chat_history: list = None, groq_api_key: str | None = None) -> dict:
    predictor = _get_predictor()
    classification = predictor.predict(question)

    payload = {
        "question": question,
        "chat_history": chat_history or [],
        "intent": classification["label"],
        "groq_api_key": groq_api_key,
    }

    result = _branch.invoke(payload)

    result["intent"] = classification["label"]
    result["confidence"] = classification["confidence"]
    result["probabilities"] = classification["probabilities"]
    return result
