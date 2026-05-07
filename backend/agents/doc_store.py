"""
In-memory document store for the RAG agent.
Provides BM25-style keyword retrieval without requiring a vector database.
Easily replaceable with a real embeddings backend (Chroma, Qdrant, etc.).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Document:
    doc_id:   str
    title:    str
    text:     str
    metadata: Dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id:  str
    doc_id:    str
    title:     str
    text:      str
    start_idx: int
    metadata:  Dict = field(default_factory=dict)


class InMemoryDocStore:
    """
    Lightweight in-process document store with BM25 retrieval.

    Supports:
    - Document ingestion with automatic chunking
    - BM25-ranked keyword retrieval
    - Multi-hop query expansion (second hop uses enriched query from first hop)
    """

    K1 = 1.5
    B  = 0.75
    CHUNK_SIZE = 400      # characters per chunk
    CHUNK_OVERLAP = 80    # character overlap between chunks

    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}
        self._inverted: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # term → [(chunk_id, freq)]
        self._avg_dl: float = 0.0
        self._doc_count: int = 0

        # Seed with built-in knowledge for demo purposes
        self._seed_knowledge()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest(self, doc: Document) -> int:
        """Chunk a document and index it. Returns number of chunks created."""
        chunks = self._chunk_text(doc)
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._rebuild_index()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """BM25 retrieval. Returns list of (chunk, score) sorted by relevance."""
        terms = self._tokenize(query)
        scores: Dict[str, float] = defaultdict(float)
        N = len(self._chunks)
        if N == 0:
            return []

        dl_map = {cid: len(self._tokenize(c.text)) for cid, c in self._chunks.items()}
        avg_dl = sum(dl_map.values()) / N if N > 0 else 1.0

        for term in terms:
            postings = dict(self._inverted.get(term, []))
            df = len(postings)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for cid, freq in postings.items():
                dl = dl_map.get(cid, avg_dl)
                tf_norm = (freq * (self.K1 + 1)) / (freq + self.K1 * (1 - self.B + self.B * dl / avg_dl))
                scores[cid] += idf * tf_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._chunks[cid], score) for cid, score in ranked if cid in self._chunks]

    def expand_query(self, original_query: str, first_hop_text: str, max_terms: int = 6) -> str:
        """
        Expand query using content from the first retrieval hop.
        Extracts high-frequency non-stopword terms to form the second-hop query.
        """
        combined = f"{original_query} {first_hop_text}"
        tokens = self._tokenize(combined)
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "could",
                     "should", "may", "might", "shall", "can", "of", "to", "in", "for",
                     "on", "with", "at", "by", "from", "as", "into", "through", "about",
                     "and", "or", "but", "not", "this", "that", "these", "those", "it",
                     "its", "they", "them", "their", "what", "which", "who", "whom"}
        freq: Dict[str, int] = defaultdict(int)
        for t in tokens:
            if t not in stopwords and len(t) > 3:
                freq[t] += 1
        top_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_terms]
        expansion = " ".join(t for t, _ in top_terms)
        return f"{original_query} {expansion}"

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def _chunk_text(self, doc: Document) -> List[Chunk]:
        text = doc.text
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.CHUNK_SIZE, len(text))
            chunk_text = text[start:end]
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_c{idx}",
                doc_id=doc.doc_id,
                title=doc.title,
                text=chunk_text,
                start_idx=start,
                metadata=doc.metadata,
            ))
            if end >= len(text):
                break
            start = end - self.CHUNK_OVERLAP
            idx += 1
        return chunks

    def _rebuild_index(self) -> None:
        self._inverted.clear()
        for cid, chunk in self._chunks.items():
            tokens = self._tokenize(chunk.text)
            freq: Dict[str, int] = defaultdict(int)
            for t in tokens:
                freq[t] += 1
            for t, f in freq.items():
                self._inverted[t].append((cid, f))

    def _seed_knowledge(self) -> None:
        """Pre-load a small built-in corpus for demo/testing."""
        docs = [
            Document(
                doc_id="doc_transformer",
                title="Transformer Architecture",
                text=(
                    "The Transformer architecture was introduced by Vaswani et al. in 2017 in the paper "
                    "'Attention Is All You Need'. It relies entirely on self-attention mechanisms and "
                    "eliminates recurrence and convolutions. The core innovation is the multi-head "
                    "attention mechanism, which allows the model to jointly attend to information from "
                    "different representation subspaces at different positions. "
                    "Each encoder layer consists of two sub-layers: a multi-head self-attention mechanism "
                    "and a position-wise fully connected feed-forward network. Residual connections and "
                    "layer normalization are applied around each sub-layer. "
                    "The decoder has an additional cross-attention layer that attends to the encoder output. "
                    "Positional encodings are added to input embeddings to inject sequence order information. "
                    "The model scales to billions of parameters, enabling modern large language models (LLMs). "
                    "GPT, BERT, T5, and LLaMA are all based on the Transformer architecture."
                ),
            ),
            Document(
                doc_id="doc_attention",
                title="Attention Mechanism Details",
                text=(
                    "Scaled dot-product attention computes attention scores as: "
                    "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. "
                    "Here Q (queries), K (keys), and V (values) are linear projections of the input. "
                    "d_k is the dimensionality of queries and keys; the division by sqrt(d_k) prevents "
                    "the dot products from growing too large in magnitude, stabilising gradients. "
                    "Multi-head attention runs h parallel attention heads, each with separate learned "
                    "projection matrices W^Q, W^K, W^V. The h outputs are concatenated and projected again. "
                    "This allows the model to capture different types of relationships simultaneously. "
                    "For a GPT-3 scale model with 96 attention heads and d_model=12288, the attention "
                    "computation accounts for roughly 2/3 of the total FLOPs per forward pass."
                ),
            ),
            Document(
                doc_id="doc_llm_params",
                title="LLM Parameter Counting",
                text=(
                    "For a decoder-only Transformer with L layers, d_model hidden dimension, d_ff "
                    "feed-forward dimension, h attention heads, and vocabulary size V: "
                    "Embedding parameters: V * d_model (input) + context_length * d_model (positional). "
                    "Per attention layer: 4 * d_model^2 (Q, K, V, O projections assuming d_k = d_model/h). "
                    "Per FFN layer: 2 * d_model * d_ff (two linear layers; d_ff typically = 4*d_model). "
                    "Total non-embedding parameters ≈ L * (4*d_model^2 + 2*d_model*d_ff). "
                    "For a 7B model (e.g., LLaMA-7B): L=32, d_model=4096, d_ff=11008, h=32. "
                    "Attention params per layer: 4 * 4096^2 ≈ 67M. FFN params per layer: 2*4096*11008 ≈ 90M. "
                    "Total per layer ≈ 157M. 32 layers ≈ 5B. Add embeddings (32000 * 4096 ≈ 131M) "
                    "and output projection ≈ 6.7B total, approximately 7B."
                ),
            ),
            Document(
                doc_id="doc_react",
                title="ReAct and Chain-of-Thought Prompting",
                text=(
                    "Chain-of-Thought (CoT) prompting, introduced by Wei et al. 2022, elicits multi-step "
                    "reasoning by including intermediate reasoning steps in the prompt examples. "
                    "It significantly improves performance on arithmetic, commonsense, and symbolic reasoning. "
                    "ReAct (Reasoning + Acting), introduced by Yao et al. 2022, interleaves reasoning traces "
                    "with action steps that interact with external tools (search, code execution). "
                    "The key difference: CoT is purely internal reasoning, while ReAct grounds reasoning in "
                    "external observations. ReAct is better for knowledge-intensive tasks that require "
                    "up-to-date or external information. CoT excels at self-contained mathematical reasoning. "
                    "Tree-of-Thought (ToT) extends CoT by exploring multiple reasoning paths in parallel "
                    "and using a verifier to select the best path, improving success rates on hard tasks."
                ),
            ),
            Document(
                doc_id="doc_rag",
                title="Retrieval-Augmented Generation",
                text=(
                    "Retrieval-Augmented Generation (RAG), introduced by Lewis et al. 2020, combines "
                    "a parametric memory (the LLM) with a non-parametric memory (a retrieval corpus). "
                    "At inference time, the system retrieves relevant document chunks using a dense "
                    "retriever (e.g., DPR, bi-encoder) or sparse retriever (BM25), then conditions "
                    "the generator on both the query and the retrieved context. "
                    "RAG reduces hallucinations by grounding the model in retrieved evidence. "
                    "Multi-hop RAG extends this by performing multiple retrieval rounds: the first hop "
                    "retrieves seed documents, and the second hop uses entities/concepts from hop-1 "
                    "results to retrieve additional evidence for complex reasoning chains. "
                    "HotpotQA and MuSiQue are standard benchmarks for multi-hop retrieval QA tasks."
                ),
            ),
        ]
        for doc in docs:
            self.ingest(doc)
        logger.info = lambda *a, **k: None  # suppress during init

    def __len__(self) -> int:
        return len(self._chunks)


# Module-level singleton
_store: Optional[InMemoryDocStore] = None


def get_doc_store() -> InMemoryDocStore:
    global _store
    if _store is None:
        _store = InMemoryDocStore()
    return _store


import logging
logger = logging.getLogger(__name__)
