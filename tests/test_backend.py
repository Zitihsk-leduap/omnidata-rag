from app.backend.main import chatbot_answer
import app.backend.main as backend_main


def test_greeting_short_circuits_without_calling_rag(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("query_rag should not be called for a greeting")

    monkeypatch.setattr(backend_main, "query_rag", fail_if_called)

    result = chatbot_answer("Hi")
    assert result["answer"] == "Hello! How can I assist you today?"
    assert result["sources"] == []


def test_canned_response_matches_case_insensitively(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("query_rag should not be called for a canned phrase")

    monkeypatch.setattr(backend_main, "query_rag", fail_if_called)

    result = chatbot_answer("So, Who Are You exactly?")
    assert "AI-powered chatbot" in result["answer"]


def test_real_question_falls_through_to_rag(monkeypatch):
    calls = []

    def fake_query_rag(user_input):
        calls.append(user_input)
        return {"answer": "7 founders are required.", "mode": "fact", "language": "english", "sources": [{"id": "x"}]}

    monkeypatch.setattr(backend_main, "query_rag", fake_query_rag)

    result = chatbot_answer("How many founders does a public company need?")
    assert calls == ["How many founders does a public company need?"]
    assert result["answer"] == "7 founders are required."
    assert result["sources"] == [{"id": "x"}]
