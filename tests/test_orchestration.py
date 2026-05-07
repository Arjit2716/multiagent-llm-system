"""Tests for the dynamic multi-agent orchestration pipeline."""
import asyncio
import json
# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── SharedContext Tests ───────────────────────────────────────────────────────

class TestSharedContext:
    def test_create_context(self):
        from backend.agents.shared_context import SharedContext
        ctx = SharedContext(query="What is transformer attention?")
        assert ctx.query == "What is transformer attention?"
        assert ctx.session_id
        assert ctx.completed_agents == []

    def test_write_agent_output(self):
        from backend.agents.shared_context import SharedContext, AgentOutput, AgentRole
        ctx = SharedContext(query="test")
        out = AgentOutput(agent_role=AgentRole.RAG, raw_output="answer")
        ctx.write_agent_output(out)
        assert AgentRole.RAG.value in ctx.completed_agents
        assert ctx.get_output(AgentRole.RAG) is not None

    def test_dependency_graph_immutable_after_set(self):
        from backend.agents.shared_context import SharedContext, DependencyGraph
        ctx = SharedContext(query="test")
        ctx.set_dependency_graph(DependencyGraph())
        with pytest.raises(ValueError):
            ctx.set_dependency_graph(DependencyGraph())

    def test_append_routing_decision(self):
        from backend.agents.shared_context import SharedContext, RoutingDecision, AgentRole
        ctx = SharedContext(query="test")
        d = RoutingDecision(selected_agent=AgentRole.RAG, context_budget=1000,
                            reasoning="RAG needed for retrieval")
        ctx.append_routing_decision(d)
        assert len(ctx.routing_decisions) == 1
        assert ctx.routing_decisions[0].reasoning == "RAG needed for retrieval"

    def test_add_critique_flag(self):
        from backend.agents.shared_context import SharedContext, CritiqueFlag, AgentRole, TextSpan
        ctx = SharedContext(query="test")
        flag = CritiqueFlag(
            target_agent=AgentRole.RAG,
            flagged_span=TextSpan(start_char=0, end_char=10, text_snippet="some text"),
            issue_type="factual_error",
            critique_text="This is wrong",
            confidence_score=0.3,
        )
        ctx.add_critique_flag(flag)
        assert len(ctx.critique_flags) == 1
        assert ctx.get_flags_for_agent(AgentRole.RAG)[0].issue_type == "factual_error"

    def test_agent_has_completed(self):
        from backend.agents.shared_context import SharedContext, AgentOutput, AgentRole
        ctx = SharedContext(query="test")
        assert not ctx.agent_has_completed(AgentRole.DECOMPOSITION)
        ctx.write_agent_output(AgentOutput(agent_role=AgentRole.DECOMPOSITION, raw_output="done"))
        assert ctx.agent_has_completed(AgentRole.DECOMPOSITION)

    def test_snapshot(self):
        from backend.agents.shared_context import SharedContext
        ctx = SharedContext(query="test query")
        snap = ctx.snapshot()
        assert "session_id" in snap
        assert "query" in snap
        assert snap["has_final_answer"] is False


# ── DependencyGraph Tests ─────────────────────────────────────────────────────

class TestDependencyGraph:
    def _make_graph(self):
        from backend.agents.shared_context import DependencyGraph, SubTask, SubTaskType, SubTaskStatus
        g = DependencyGraph()
        t1 = SubTask(task_id="t1", task_type=SubTaskType.FACTUAL,
                     description="retrieve", depends_on=[], status=SubTaskStatus.PENDING)
        t2 = SubTask(task_id="t2", task_type=SubTaskType.REASONING,
                     description="reason", depends_on=["t1"], status=SubTaskStatus.PENDING)
        t3 = SubTask(task_id="t3", task_type=SubTaskType.SUMMARIZATION,
                     description="summarize", depends_on=["t1", "t2"], status=SubTaskStatus.PENDING)
        g.tasks = {"t1": t1, "t2": t2, "t3": t3}
        return g

    def test_ready_tasks_initially_only_root(self):
        g = self._make_graph()
        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t1"

    def test_ready_after_t1_done(self):
        g = self._make_graph()
        g.mark_done("t1", "retrieved data")
        ready = g.get_ready_tasks()
        assert any(t.task_id == "t2" for t in ready)

    def test_all_done(self):
        g = self._make_graph()
        assert not g.all_done()
        g.mark_done("t1", "x")
        g.mark_done("t2", "y")
        g.mark_done("t3", "z")
        assert g.all_done()

    def test_topological_order(self):
        g = self._make_graph()
        order = g.topological_order()
        assert order.index("t1") < order.index("t2")
        assert order.index("t2") < order.index("t3")

    def test_no_cycle_detection(self):
        g = self._make_graph()
        assert not g.has_cycle()

    def test_cycle_detection(self):
        from backend.agents.shared_context import DependencyGraph, SubTask, SubTaskType, SubTaskStatus
        g = DependencyGraph()
        t1 = SubTask(task_id="t1", task_type=SubTaskType.FACTUAL,
                     description="a", depends_on=["t2"])
        t2 = SubTask(task_id="t2", task_type=SubTaskType.FACTUAL,
                     description="b", depends_on=["t1"])
        g.tasks = {"t1": t1, "t2": t2}
        assert g.has_cycle()

    def test_mark_failed(self):
        from backend.agents.shared_context import SubTaskStatus
        g = self._make_graph()
        g.mark_failed("t1")
        assert g.tasks["t1"].status == SubTaskStatus.FAILED


# ── DocStore Tests ────────────────────────────────────────────────────────────

class TestDocStore:
    def test_search_returns_results(self):
        from backend.agents.doc_store import InMemoryDocStore
        store = InMemoryDocStore()
        results = store.search("transformer attention mechanism", top_k=3)
        assert len(results) > 0
        assert all(isinstance(score, float) for _, score in results)

    def test_search_returns_relevant_chunks(self):
        from backend.agents.doc_store import InMemoryDocStore
        store = InMemoryDocStore()
        results = store.search("multi-head attention", top_k=3)
        texts = [c.text.lower() for c, _ in results]
        assert any("attention" in t for t in texts)

    def test_ingest_custom_document(self):
        from backend.agents.doc_store import InMemoryDocStore, Document
        store = InMemoryDocStore()
        before = len(store)
        doc = Document(doc_id="test_doc", title="Test", text="a" * 1000)
        n = store.ingest(doc)
        assert n > 0
        assert len(store) > before

    def test_query_expansion(self):
        from backend.agents.doc_store import InMemoryDocStore
        store = InMemoryDocStore()
        expanded = store.expand_query("transformer", "multi-head attention residual connections", max_terms=3)
        assert "transformer" in expanded
        assert len(expanded) > len("transformer")

    def test_hop_index_in_results(self):
        from backend.agents.shared_context import RetrievedChunk
        chunk = RetrievedChunk(chunk_id="c1", source="test", text="test", hop_index=2)
        assert chunk.hop_index == 2

    def test_chunk_relevance_sorted(self):
        from backend.agents.doc_store import InMemoryDocStore
        store = InMemoryDocStore()
        results = store.search("LLM parameter count 7B", top_k=5)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]


# ── DecompositionAgent Tests ──────────────────────────────────────────────────

class TestDecompositionAgent:
    def test_parse_valid_plan(self):
        from backend.agents.decomposition import DecompositionAgent
        agent = DecompositionAgent.__new__(DecompositionAgent)
        plan_json = json.dumps({
            "query_analysis": "Multi-part query",
            "subtasks": [
                {"task_id": "t1", "task_type": "factual", "description": "Find X", "depends_on": [], "assigned_to": "rag"},
                {"task_id": "t2", "task_type": "reasoning", "description": "Reason about X", "depends_on": ["t1"], "assigned_to": "rag"},
            ],
            "dependency_rationale": "t2 needs t1's facts"
        })
        result = agent._parse_plan(plan_json)
        assert result is not None
        assert len(result["subtasks"]) == 2

    def test_parse_json_in_codeblock(self):
        from backend.agents.decomposition import DecompositionAgent
        agent = DecompositionAgent.__new__(DecompositionAgent)
        text = '```json\n{"query_analysis": "test", "subtasks": [{"task_id":"t1","task_type":"factual","description":"x","depends_on":[],"assigned_to":"rag"}]}\n```'
        result = agent._parse_plan(text)
        assert result is not None

    def test_fallback_plan_has_two_tasks(self):
        from backend.agents.decomposition import DecompositionAgent
        agent = DecompositionAgent.__new__(DecompositionAgent)
        plan = agent._fallback_plan("What is quantum computing?")
        assert len(plan["subtasks"]) == 2
        assert plan["subtasks"][1]["depends_on"] == ["t1"]

    def test_fallback_plan_has_dependency(self):
        from backend.agents.decomposition import DecompositionAgent
        agent = DecompositionAgent.__new__(DecompositionAgent)
        plan = agent._fallback_plan("test query")
        assert len(plan["subtasks"][1]["depends_on"]) > 0


# ── RAGAgent Tests ────────────────────────────────────────────────────────────

class TestRAGAgent:
    def test_build_claims_with_valid_chunks(self):
        from backend.agents.rag_agent import RAGAgent
        from backend.agents.shared_context import RetrievedChunk
        from backend.agents.doc_store import InMemoryDocStore
        agent = RAGAgent(doc_store=InMemoryDocStore())

        chunks = [
            RetrievedChunk(chunk_id="c1", source="doc1", text="Transformers use attention", hop_index=1),
            RetrievedChunk(chunk_id="c2", source="doc2", text="Multi-hop enables better reasoning", hop_index=2),
        ]
        parsed = {
            "answer": "Transformers use attention for context. Multi-hop enables better reasoning.",
            "claims": [
                {"text": "Transformers use attention", "contributing_chunks": ["c1"], "confidence": 0.9},
                {"text": "Multi-hop enables better reasoning", "contributing_chunks": ["c2"], "confidence": 0.8},
            ]
        }
        claims = agent._build_claims(parsed, chunks)
        assert len(claims) == 2
        assert claims[0].confidence_score == 0.9
        assert "c1" in claims[0].supporting_chunks

    def test_parse_response_json(self):
        from backend.agents.rag_agent import RAGAgent
        from backend.agents.doc_store import InMemoryDocStore
        agent = RAGAgent(doc_store=InMemoryDocStore())
        response = '{"answer": "Test answer", "claims": [], "hop_reasoning": {"hop1_insight": "x"}}'
        parsed = agent._parse_response(response)
        assert parsed["answer"] == "Test answer"

    def test_min_hops_constant(self):
        from backend.agents.rag_agent import RAGAgent
        assert RAGAgent.MIN_HOPS == 2

    def test_hop_index_increments(self):
        from backend.agents.shared_context import RetrievedChunk
        c1 = RetrievedChunk(chunk_id="a", source="s", text="t", hop_index=1)
        c2 = RetrievedChunk(chunk_id="b", source="s", text="t", hop_index=2)
        assert c2.hop_index > c1.hop_index


# ── CritiqueAgent Tests ───────────────────────────────────────────────────────

class TestCritiqueAgent:
    def test_parse_review_valid_json(self):
        from backend.agents.critique_agent import CritiqueAgent
        agent = CritiqueAgent.__new__(CritiqueAgent)
        review_json = json.dumps({
            "agent_reviewed": "rag",
            "overall_assessment": "Good but one unsupported claim",
            "flags": [{"flagged_text": "some claim", "start_char": 0, "end_char": 10,
                       "issue_type": "unsupported_claim", "confidence_score": 0.4,
                       "critique_text": "Not supported by chunks"}],
            "per_claim_scores": [],
            "average_confidence": 0.65
        })
        result = agent._parse_review(review_json)
        assert result is not None
        assert len(result["flags"]) == 1

    def test_build_flags_from_review(self):
        from backend.agents.critique_agent import CritiqueAgent
        from backend.agents.shared_context import AgentRole
        agent = CritiqueAgent.__new__(CritiqueAgent)
        raw_output = "Transformers use attention for context modeling."
        review = {
            "flags": [{
                "flagged_text": "Transformers use attention",
                "start_char": 0, "end_char": 25,
                "issue_type": "unsupported_claim",
                "confidence_score": 0.4,
                "critique_text": "No chunk supports this exactly"
            }]
        }
        flags = agent._build_flags(review, AgentRole.RAG, raw_output)
        assert len(flags) == 1
        assert flags[0].issue_type == "unsupported_claim"
        assert flags[0].confidence_score == 0.4

    def test_build_flags_empty_when_no_flags(self):
        from backend.agents.critique_agent import CritiqueAgent
        from backend.agents.shared_context import AgentRole
        agent = CritiqueAgent.__new__(CritiqueAgent)
        flags = agent._build_flags({"flags": []}, AgentRole.RAG, "clean output")
        assert flags == []


# ── SynthesisAgent Tests ──────────────────────────────────────────────────────

class TestSynthesisAgent:
    def test_parse_response_extracts_final_answer(self):
        from backend.agents.synthesis_agent import SynthesisAgent
        agent = SynthesisAgent.__new__(SynthesisAgent)
        response = json.dumps({
            "final_answer": "The transformer uses multi-head attention.",
            "sentences": [{"index": 0, "text": "The transformer uses multi-head attention.",
                           "source_agent": "rag", "source_chunks": ["c1"], "confidence": 0.9}],
            "contradiction_resolutions": []
        })
        parsed = agent._parse_response(response)
        assert parsed["final_answer"] == "The transformer uses multi-head attention."

    def test_build_provenance_from_parsed(self):
        from backend.agents.synthesis_agent import SynthesisAgent
        from backend.agents.shared_context import SharedContext, RetrievedChunk
        agent = SynthesisAgent.__new__(SynthesisAgent)
        ctx = SharedContext(query="test")
        ctx.add_retrieved_chunk(RetrievedChunk(chunk_id="c1", source="doc", text="attention text", hop_index=1))
        parsed = {
            "final_answer": "Attention is key to transformers.",
            "sentences": [{"index": 0, "text": "Attention is key.", "source_agent": "rag",
                           "source_chunks": ["c1"], "confidence": 0.9}],
            "contradiction_resolutions": []
        }
        answer, prov = agent._build_provenance(parsed, ctx)
        assert answer == "Attention is key to transformers."
        assert len(prov.sentences) == 1
        assert prov.sentences[0].source_chunks == ["c1"]

    def test_auto_provenance_splits_sentences(self):
        from backend.agents.synthesis_agent import SynthesisAgent
        from backend.agents.shared_context import SharedContext
        agent = SynthesisAgent.__new__(SynthesisAgent)
        ctx = SharedContext(query="test")
        text = "First sentence. Second sentence. Third sentence."
        sentences = agent._auto_provenance(text, ctx)
        assert len(sentences) == 3

    def test_provenance_map_to_readable(self):
        from backend.agents.shared_context import ProvenanceMap, ProvenanceSentence, AgentRole
        pm = ProvenanceMap(sentences=[
            ProvenanceSentence(sentence_index=0, text="Sentence one.", source_agent=AgentRole.RAG,
                               source_chunks=["c1"], confidence=0.9)
        ])
        readable = pm.to_readable()
        assert "Agent=rag" in readable
        assert "c1" in readable


# ── DynamicOrchestrator Tests ─────────────────────────────────────────────────

class TestDynamicOrchestrator:
    def test_extract_json_from_codeblock(self):
        from backend.agents.dynamic_orchestrator import DynamicOrchestrator
        text = '```json\n{"selected_agent": "rag", "context_budget": 2000}\n```'
        result = DynamicOrchestrator._extract_json(text)
        assert result["selected_agent"] == "rag"

    def test_extract_json_bare(self):
        from backend.agents.dynamic_orchestrator import DynamicOrchestrator
        text = '{"selected_agent": "critique", "context_budget": 1500, "stop_pipeline": false}'
        result = DynamicOrchestrator._extract_json(text)
        assert result["selected_agent"] == "critique"

    def test_extract_json_returns_none_on_garbage(self):
        from backend.agents.dynamic_orchestrator import DynamicOrchestrator
        result = DynamicOrchestrator._extract_json("not json at all!!!")
        assert result is None

    def test_register_agent(self):
        from backend.agents.dynamic_orchestrator import DynamicOrchestrator
        from backend.agents.shared_context import AgentRole
        orch = DynamicOrchestrator()
        fn = AsyncMock()
        orch.register(AgentRole.RAG, fn)
        assert AgentRole.RAG in orch._agent_registry

    def test_build_state_json_contains_keys(self):
        from backend.agents.dynamic_orchestrator import DynamicOrchestrator
        from backend.agents.shared_context import SharedContext
        orch = DynamicOrchestrator()
        ctx = SharedContext(query="test query")
        state = orch._build_state_json(ctx)
        data = json.loads(state)
        assert "completed_agents" in data
        assert "has_final_answer" in data
        assert "tokens_remaining" in data

    @pytest.mark.asyncio
    async def test_stop_when_synthesis_complete(self):
        from backend.agents.dynamic_orchestrator import DynamicOrchestrator
        from backend.agents.shared_context import SharedContext, AgentOutput, AgentRole, ProvenanceMap
        orch = DynamicOrchestrator(max_steps=10)
        ctx = SharedContext(query="test")
        # Pre-fill synthesis as done
        ctx.write_agent_output(AgentOutput(agent_role=AgentRole.SYNTHESIS, raw_output="done"))
        ctx.final_answer = "The final answer."
        ctx.provenance_map = ProvenanceMap()
        # Should return immediately without calling any agents
        result = await orch.run(ctx)
        assert result.final_answer == "The final answer."


# ── Integration: Full Pipeline with Mocked LLM ───────────────────────────────

class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_decomposition_writes_to_context(self):
        from backend.agents.decomposition import DecompositionAgent
        from backend.agents.shared_context import SharedContext

        decomp_response = json.dumps({
            "query_analysis": "Two-part question",
            "subtasks": [
                {"task_id": "t1", "task_type": "factual", "description": "Retrieve facts", "depends_on": [], "assigned_to": "rag"},
                {"task_id": "t2", "task_type": "reasoning", "description": "Reason over facts", "depends_on": ["t1"], "assigned_to": "rag"},
            ],
            "dependency_rationale": "Retrieval before reasoning"
        })

        with patch("backend.core.llm_client.acompletion") as mock:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content=decomp_response))]
            mock_resp.usage = MagicMock(prompt_tokens=50, completion_tokens=100)
            mock.return_value = mock_resp

            ctx = SharedContext(query="What is transformer attention and how many parameters?")
            agent = DecompositionAgent()
            await agent(ctx, token_budget=1000)

        assert ctx.dependency_graph is not None
        assert len(ctx.dependency_graph.tasks) == 2
        assert ctx.agent_has_completed(from_import := __import__('backend.agents.shared_context', fromlist=['AgentRole']).AgentRole.DECOMPOSITION)

    @pytest.mark.asyncio
    async def test_rag_writes_chunks_to_context(self):
        from backend.agents.rag_agent import RAGAgent
        from backend.agents.shared_context import SharedContext
        from backend.agents.doc_store import InMemoryDocStore

        rag_response = json.dumps({
            "answer": "Transformers use scaled dot-product attention.",
            "claims": [{"text": "Transformers use scaled dot-product attention.",
                        "contributing_chunks": [], "confidence": 0.85}],
            "hop_reasoning": {"hop1_insight": "attention details",
                              "hop2_insight": "parameter counts",
                              "multi_hop_chain": "linked via model architecture"},
            "unanswered_aspects": []
        })

        with patch("backend.core.llm_client.acompletion") as mock:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content=rag_response))]
            mock_resp.usage = MagicMock(prompt_tokens=200, completion_tokens=100)
            mock.return_value = mock_resp

            ctx = SharedContext(query="Explain transformer attention mechanism")
            store = InMemoryDocStore()
            agent = RAGAgent(doc_store=store)
            await agent(ctx, token_budget=2000)

        # Must have at least 2 hops worth of chunks
        assert len(ctx.retrieved_chunks) >= 2
        hops = {c.hop_index for c in ctx.retrieved_chunks.values()}
        assert 1 in hops
        assert 2 in hops
        assert len(ctx.rag_hop_trace) == 2

    @pytest.mark.asyncio
    async def test_context_no_direct_agent_calls(self):
        """Verify agents only write to context, not call each other."""
        from backend.agents.shared_context import SharedContext, AgentOutput, AgentRole
        ctx = SharedContext(query="test")
        # Simulate two agents writing outputs independently
        ctx.write_agent_output(AgentOutput(agent_role=AgentRole.DECOMPOSITION, raw_output="decomp done"))
        ctx.write_agent_output(AgentOutput(agent_role=AgentRole.RAG, raw_output="rag done"))
        # Each agent has its own slot
        assert ctx.get_output(AgentRole.DECOMPOSITION).raw_output == "decomp done"
        assert ctx.get_output(AgentRole.RAG).raw_output == "rag done"
        # Neither called the other
        assert len(ctx.error_log) == 0
