"""
retrievers.py — BM25, DPR, and Hybrid (RRF / Alpha) retrievers
used throughout the RAG retrieval comparison study.
"""

import time
import string
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()


class BM25Retriever:
    """Sparse retrieval using BM25Okapi (k1=1.5, b=0.75 default)."""

    def __init__(self, corpus):
        """corpus: list of {'id': str, 'text': str}"""
        self.ids = [p['id'] for p in corpus]
        self.texts = [p['text'] for p in corpus]
        self.bm25 = None

    def build_index(self):
        t0 = time.perf_counter()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        ms = (time.perf_counter() - t0) * 1000
        print(f'✅ BM25 — {len(self.ids)} passages | {ms:.0f}ms')

    def retrieve(self, query, top_k=5):
        t0 = time.perf_counter()
        scores = self.bm25.get_scores(tokenize(query))
        ms = (time.perf_counter() - t0) * 1000
        ranked = sorted(zip(self.ids, self.texts, scores),
                         key=lambda x: x[2], reverse=True)[:top_k]
        return [{'id': pid, 'text': txt, 'score': float(sc), 'rank': i + 1, 'latency_ms': ms}
                for i, (pid, txt, sc) in enumerate(ranked)]


class DPRRetriever:
    """Dense retrieval using sentence-transformers + FAISS IndexFlatIP."""

    def __init__(self, corpus, model_name='multi-qa-MiniLM-L6-cos-v1'):
        self.ids = [p['id'] for p in corpus]
        self.texts = [p['text'] for p in corpus]
        self.model_name = model_name
        self.model = None
        self.index = None

    def build_index(self):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Loading DPR on {device}...')
        self.model = SentenceTransformer(self.model_name, device=device)

        t0 = time.perf_counter()
        embs = self.model.encode(
            self.texts, batch_size=64, show_progress_bar=True,
            normalize_embeddings=True
        ).astype('float32')
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        ms = (time.perf_counter() - t0) * 1000
        print(f'✅ DPR — {self.index.ntotal} vectors | {ms:.0f}ms')

    def retrieve(self, query, top_k=5):
        t0 = time.perf_counter()
        q_vec = self.model.encode([query], normalize_embeddings=True).astype('float32')
        scores, idxs = self.index.search(q_vec, top_k)
        ms = (time.perf_counter() - t0) * 1000
        return [{'id': self.ids[i], 'text': self.texts[i], 'score': float(s),
                 'rank': r + 1, 'latency_ms': ms}
                for r, (i, s) in enumerate(zip(idxs[0], scores[0])) if i >= 0]


class HybridRetriever:
    """Hybrid retrieval combining BM25 + DPR via RRF or alpha-weighted fusion."""

    def __init__(self, bm25, dpr, fusion='rrf', rrf_k=60, alpha=0.3, factor=3):
        """
        fusion: 'rrf' (Reciprocal Rank Fusion) or 'alpha' (linear score fusion)
        rrf_k: RRF smoothing constant (default 60)
        alpha: weight on BM25 in alpha fusion (0=pure DPR, 1=pure BM25)
        factor: candidate_factor — retrieve top_k*factor from each retriever before fusion
        """
        self.bm25 = bm25
        self.dpr = dpr
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.factor = factor

    def retrieve(self, query, top_k=5):
        k = top_k * self.factor
        t0 = time.perf_counter()
        b = self.bm25.retrieve(query, k)
        d = self.dpr.retrieve(query, k)
        ms = (time.perf_counter() - t0) * 1000

        if self.fusion == 'rrf':
            scores, texts = {}, {}
            for r in b:
                scores[r['id']] = scores.get(r['id'], 0) + 1 / (self.rrf_k + r['rank'])
                texts[r['id']] = r['text']
            for r in d:
                scores[r['id']] = scores.get(r['id'], 0) + 1 / (self.rrf_k + r['rank'])
                texts[r['id']] = r['text']
            merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [{'id': pid, 'text': texts[pid], 'score': sc, 'rank': i + 1, 'latency_ms': ms}
                    for i, (pid, sc) in enumerate(merged)]

        else:  # alpha fusion
            def norm(res):
                s = [r['score'] for r in res]
                lo, hi = min(s), max(s)
                denom = hi - lo if hi != lo else 1
                return {r['id']: (r['score'] - lo) / denom for r in res}

            bn, dn = norm(b), norm(d)
            texts = {r['id']: r['text'] for r in b + d}
            comb = {pid: self.alpha * bn.get(pid, 0) + (1 - self.alpha) * dn.get(pid, 0)
                    for pid in set(bn) | set(dn)}
            merged = sorted(comb.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [{'id': pid, 'text': texts[pid], 'score': sc, 'rank': i + 1, 'latency_ms': ms}
                    for i, (pid, sc) in enumerate(merged)]
