"""
Hybrid search combining Pinecone semantic search with BM25 keyword search,
fused using Reciprocal Rank Fusion (RRF).
"""

import math
import re
from collections import defaultdict
from typing import List, Optional

from langchain_core.documents import Document


class HybridRetriever:
    """
    Hybrid retriever combining Pinecone semantic search with BM25 keyword search,
    fused using Reciprocal Rank Fusion (RRF).

    Strategy:
      - Semantic : query Pinecone with dense embeddings
      - Keyword  : BM25 over all document chunks pre-loaded at startup
      - Fusion   : RRF merges both ranked lists; docs found by only one method
                   still contribute via their single rank score
    """

    def __init__(
        self,
        embeddings,
        pinecone_index,
        doc_chunks: List[Document],
        k: int = 5,
        beta: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize the hybrid retriever and pre-compute BM25 corpus statistics.

        Args:
            embeddings    : HuggingFace embeddings object (must support embed_query)
            pinecone_index: Raw Pinecone Index object (from PineconeStore.index)
            doc_chunks    : All document chunks loaded from PDFs, used as BM25 corpus
            k             : Number of final results to return
            beta          : Semantic weight in RRF (0 = pure keyword, 1 = pure semantic)
            verbose       : Print retrieval progress
        """
        self.embeddings = embeddings
        self.pinecone_index = pinecone_index
        self.doc_chunks = doc_chunks
        self.k = k
        self.beta = beta
        self.verbose = verbose

        # Pre-computed BM25 corpus statistics (built once at startup)
        self._doc_tokens_list: List[List[str]] = []
        self._avg_doc_length: float = 1.0
        self._doc_frequencies: dict = {}

        self._build_bm25_cache()

    # ------------------------------------------------------------------
    # BM25 helpers
    # ------------------------------------------------------------------

    def _build_bm25_cache(self):
        """Pre-tokenize all chunks and compute BM25 corpus statistics."""
        if not self.doc_chunks:
            if self.verbose:
                print("Warning: no document chunks provided — BM25 disabled.")
            return

        if self.verbose:
            print(f"Building BM25 cache from {len(self.doc_chunks)} chunks...")

        self._doc_tokens_list = [
            self._tokenize(doc.page_content) for doc in self.doc_chunks
        ]

        total_tokens = sum(len(t) for t in self._doc_tokens_list)
        self._avg_doc_length = total_tokens / len(self._doc_tokens_list)

        freq: dict = defaultdict(int)
        for tokens in self._doc_tokens_list:
            for token in set(tokens):
                freq[token] += 1
        self._doc_frequencies = freq

        if self.verbose:
            print(f"BM25 cache ready — avg doc length: {self._avg_doc_length:.1f} tokens")

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase word tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def _bm25_score(
        self,
        query_tokens: List[str],
        doc_idx: int,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """BM25 score for a single document (by corpus index)."""
        doc_tokens = self._doc_tokens_list[doc_idx]
        doc_length = len(doc_tokens)
        total_docs = len(self._doc_tokens_list)

        counts: dict = defaultdict(int)
        for token in doc_tokens:
            counts[token] += 1

        score = 0.0
        for token in query_tokens:
            if token not in counts:
                continue
            tf = counts[token]
            df = self._doc_frequencies.get(token, 0)
            idf = math.log(1 + max(0.0, (total_docs - df + 0.5) / (df + 0.5)))
            tf_norm = (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * (doc_length / self._avg_doc_length))
            )
            score += idf * tf_norm

        return score

    # ------------------------------------------------------------------
    # Individual search methods
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, top_k: int) -> List[dict]:
        """
        BM25 search over all document chunks.

        Returns list of {doc, score, rank} sorted by score desc.
        """
        if not self._doc_tokens_list:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for i, doc in enumerate(self.doc_chunks):
            score = self._bm25_score(query_tokens, i)
            if score > 0:
                scored.append({'doc': doc, 'score': score})

        scored.sort(key=lambda x: x['score'], reverse=True)

        for rank, item in enumerate(scored[:top_k], 1):
            item['rank'] = rank

        return scored[:top_k]

    def _semantic_search(self, query: str, top_k: int) -> List[dict]:
        """
        Dense vector search via Pinecone.

        Returns list of {text, metadata, score, rank} sorted by score desc.
        """
        query_vector = self.embeddings.embed_query(query)

        results = self.pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        semantic = []
        for rank, match in enumerate(results.matches, 1):
            # LangChain-Pinecone stores chunk text under the 'text' key
            text = match.metadata.get('text', match.metadata.get('page_content', ''))
            metadata = {
                k: v for k, v in match.metadata.items()
                if k not in ('text', 'page_content')
            }
            semantic.append({
                'text': text,
                'metadata': metadata,
                'score': match.score,
                'rank': rank,
                'key': text[:200].strip()
            })

        return semantic

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        semantic_results: List[dict],
        keyword_results: List[dict],
        k_rrf: int = 60
    ) -> List[dict]:
        """
        Reciprocal Rank Fusion.

        RRF(d) = beta/(k_rrf + sem_rank) + (1-beta)/(k_rrf + kw_rank)

        Documents found by only one method contribute only that term;
        the missing term contributes 0 (not a penalizing fallback rank).
        """
        results_map: dict = {}

        # Seed with semantic results
        for r in semantic_results:
            key = r['key']
            results_map[key] = {
                'text': r['text'],
                'metadata': r['metadata'],
                'semantic_rank': r['rank'],
                'keyword_rank': None,
            }

        # Merge keyword results — match by first 200 chars of text
        for r in keyword_results:
            key = r['doc'].page_content[:200].strip()
            if key in results_map:
                results_map[key]['keyword_rank'] = r['rank']
            else:
                # BM25 surfaced a chunk that semantic search missed
                results_map[key] = {
                    'text': r['doc'].page_content,
                    'metadata': r['doc'].metadata,
                    'semantic_rank': None,
                    'keyword_rank': r['rank'],
                }

        # Compute RRF scores
        fused = []
        for r in results_map.values():
            sem_rank = r['semantic_rank']
            kw_rank = r['keyword_rank']
            sem_rrf = self.beta / (k_rrf + sem_rank) if sem_rank is not None else 0.0
            kw_rrf = (1 - self.beta) / (k_rrf + kw_rank) if kw_rank is not None else 0.0
            fused.append({**r, 'rrf_score': sem_rrf + kw_rrf})

        fused.sort(key=lambda x: x['rrf_score'], reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve top-k documents using hybrid search.

        Args:
            query: Natural-language query string

        Returns:
            List of LangChain Documents ranked by RRF score
        """
        # Fetch a larger candidate pool before fusion for better coverage
        fetch_k = min(self.k * 3, 50)

        if self.verbose:
            print(f"\nHybrid search | k={self.k} beta={self.beta} fetch_k={fetch_k}")

        semantic_results = self._semantic_search(query, fetch_k)
        keyword_results = self._keyword_search(query, fetch_k)

        if self.verbose:
            print(
                f"  Semantic: {len(semantic_results)} results  "
                f"| Keyword: {len(keyword_results)} results"
            )

        fused = self._rrf_fuse(semantic_results, keyword_results)

        if self.verbose:
            print(f"  After RRF: {len(fused)} unique docs — returning top {self.k}")

        return [
            Document(page_content=r['text'], metadata=r['metadata'])
            for r in fused[:self.k]
        ]
