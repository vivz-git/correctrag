import pytest
from app.evaluation.action_router import ActionRouter, RoutingDecision
from app.retrieval.retriever import RetrievedChunk

class MockLLM:
    def __init__(self, response_text='{"decision": "CORRECT", "reason": "test"}', raise_exc=False):
        self.response_text = response_text
        self.raise_exc = raise_exc
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        if self.raise_exc:
            raise Exception("API failure")
        return self.response_text


def test_clearly_relevant():
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=MockLLM())
    chunks = [RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=0.8)]
    decision = router.route("query", chunks, [0.8])
    assert decision.action == "CORRECT"
    assert decision.similarity_pre_filter_decision == "CORRECT"
    assert decision.judge_called is False

def test_clearly_irrelevant():
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=MockLLM())
    chunks = [RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=-0.5)]
    decision = router.route("query", chunks, [-0.5])
    assert decision.action == "INCORRECT"
    assert decision.similarity_pre_filter_decision == "INCORRECT"
    assert decision.judge_called is False

def test_borderline_judge_called_once():
    llm = MockLLM(response_text='{"decision": "CORRECT", "reason": "test"}')
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=llm)
    chunks = [
        RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=0.2),
        RetrievedChunk(chunk_id="2", source="doc", text="y", page_number=1, score=0.3)
    ]
    decision = router.route("query", chunks, [0.2, 0.3])
    assert decision.judge_called is True
    assert llm.call_count == 1
    assert decision.action == "CORRECT"

def test_judge_returns_ambiguous():
    llm = MockLLM(response_text='{"decision": "AMBIGUOUS", "reason": "test"}')
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=llm)
    chunks = [RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=0.2)]
    decision = router.route("query", chunks, [0.2])
    assert decision.action == "AMBIGUOUS"
    assert decision.judge_reason == "test"

def test_judge_returns_incorrect():
    llm = MockLLM(response_text='{"decision": "INCORRECT", "reason": "test"}')
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=llm)
    chunks = [RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=0.2)]
    decision = router.route("query", chunks, [0.2])
    assert decision.action == "INCORRECT"

def test_malformed_judge_response():
    llm = MockLLM(response_text="some random string CORRECT blah")
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=llm)
    chunks = [RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=0.2)]
    decision = router.route("query", chunks, [0.2])
    assert decision.action == "CORRECT"
    assert decision.judge_reason == "Failed to parse structured output."

def test_judge_api_failure():
    llm = MockLLM(raise_exc=True)
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=llm)
    chunks = [RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=0.2)]
    decision = router.route("query", chunks, [0.2])
    assert decision.action == "AMBIGUOUS"
    assert "Judge API failure" in decision.judge_reason

def test_provider_configuration_works():
    # If no LLM client is provided, defaults to AMBIGUOUS safely
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2)
    chunks = [RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=0.2)]
    decision = router.route("query", chunks, [0.2])
    assert decision.action == "AMBIGUOUS"
    assert decision.judge_called is False

def test_alpha_validation():
    with pytest.raises(ValueError):
        ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=0.5)

def test_length_validation():
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2)
    with pytest.raises(ValueError):
        router.route("q", [], [])
    with pytest.raises(ValueError):
        router.route("q", [RetrievedChunk(chunk_id="1", source="doc", text="x", page_number=1, score=0.2)], [0.2, 0.3])
