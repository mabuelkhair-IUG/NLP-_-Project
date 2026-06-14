"""
metrics.py — Retrieval and generation evaluation metrics
used throughout the RAG retrieval comparison study.
"""

import numpy as np
from collections import Counter
from scipy import stats


# ── Retrieval-level metrics ─────────────────────────────────────────────────

def precision_at_k(retrieved, relevant, k):
    """Fraction of top-k retrieved passages that are relevant."""
    return len(set(retrieved[:k]) & set(relevant)) / k if relevant else 0.0


def recall_at_k(retrieved, relevant, k):
    """Fraction of relevant passages found in top-k."""
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def mrr(retrieved, relevant, k=10):
    """Mean Reciprocal Rank: 1/rank of first relevant passage in top-k."""
    for i, pid in enumerate(retrieved[:k], 1):
        if pid in set(relevant):
            return 1.0 / i
    return 0.0


# ── Generation-level metrics ────────────────────────────────────────────────

def token_f1(answer, context):
    """Token-level F1 overlap between answer and retrieved context."""
    a = Counter(answer.lower().split())
    c = Counter(context.lower().split())
    common = sum((a & c).values())
    if not common:
        return 0.0
    p = common / sum(c.values())
    r = common / sum(a.values())
    return 2 * p * r / (p + r)


def exact_match(answer, context):
    """1 if normalized answer string appears verbatim in context."""
    return float(answer.lower() in context.lower())


def faithfulness(answer, context):
    """Proportion of answer tokens covered by context tokens."""
    a = set(answer.lower().split())
    c = set(context.lower().split())
    return len(a & c) / len(a) if a else 0.0


def answer_relevance(answer, question):
    """Token overlap between answer and question (topicality)."""
    a = set(answer.lower().split())
    q = set(question.lower().split())
    return len(a & q) / max(len(a), len(q)) if a and q else 0.0


# ── Evaluation harness ──────────────────────────────────────────────────────

def evaluate(name, retriever, queries, top_k=5, with_per_query=False):
    """
    Run a retriever over a list of queries and compute aggregate metrics.

    queries: list of dicts with keys:
        'question', 'relevant_ids', 'answer'

    Returns a dict of mean metrics. If with_per_query=True, also includes
    '_mrr_list' (per-query MRR values) for statistical significance testing.
    """
    P, R, M, F, E, Faith, Rel, Lat = [], [], [], [], [], [], [], []
    for q in queries:
        res = retriever.retrieve(q['question'], top_k)
        ids = [r['id'] for r in res]
        ctx = res[0]['text'] if res else ''
        ans = q.get('answer', '')

        P.append(precision_at_k(ids, q['relevant_ids'], top_k))
        R.append(recall_at_k(ids, q['relevant_ids'], top_k))
        M.append(mrr(ids, q['relevant_ids']))
        F.append(token_f1(ans, ctx))
        E.append(exact_match(ans, ctx))
        Faith.append(faithfulness(ans, ctx))
        Rel.append(answer_relevance(ans, q['question']))
        Lat.append(res[0]['latency_ms'] if res else 0)

    result = {
        'Method': name,
        'P@5': round(np.mean(P), 4),
        'R@5': round(np.mean(R), 4),
        'MRR': round(np.mean(M), 4),
        'F1': round(np.mean(F), 4),
        'EM': round(np.mean(E), 4),
        'Faithfulness': round(np.mean(Faith), 4),
        'Ans.Relevance': round(np.mean(Rel), 4),
        'Latency(ms)': round(np.mean(Lat), 3),
    }
    if with_per_query:
        result['_mrr_list'] = M
    return result


# ── Statistical testing ─────────────────────────────────────────────────────

def paired_significance(mrr_list, baseline_mrr_list):
    """Paired t-test on per-query MRR vs a baseline (e.g., BM25)."""
    t_stat, p_value = stats.ttest_rel(mrr_list, baseline_mrr_list)
    return t_stat, p_value


def bootstrap_ci(data, n_boot=1000, ci=95):
    """Bootstrap confidence interval for the mean of `data`."""
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True))
                   for _ in range(n_boot)]
    lo = (100 - ci) / 2
    hi = 100 - lo
    return np.percentile(boot_means, lo), np.percentile(boot_means, hi)


def highlight_best(s):
    """Pandas Styler helper: highlight best value in a column (min for latency)."""
    best = s.min() if s.name == 'Latency(ms)' else s.max()
    return ['background-color:#d4efdf;font-weight:bold' if v == best else '' for v in s]
