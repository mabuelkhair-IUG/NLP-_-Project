"""
data_loaders.py — Build independent open-retrieval corpora and query sets
for TriviaQA, HotpotQA, MS MARCO, and Natural Questions.

Each loader returns (passages, queries):
  passages: list of {'id': str, 'text': str}  — the shared corpus
  queries:  list of {'id', 'type', 'question', 'relevant_ids', 'answer'}

Each query's relevant_ids reference passages in the SAME shared corpus,
enabling genuine open-retrieval evaluation (no closed/query-specific pools).
"""

from datasets import load_dataset


def load_triviaqa(n_source=600, n_queries=500, max_len=400):
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia",
                            split=f"validation[:{n_source}]", trust_remote_code=True)

    passages, queries, seen = [], [], {}

    for ex in dataset:
        for ctx in ex['entity_pages']['wiki_context']:
            if not ctx or len(ctx.strip()) < 50:
                continue
            text = ctx.strip()[:max_len]
            if text not in seen:
                pid = f'tqa_p{len(passages)}'
                passages.append({'id': pid, 'text': text})
                seen[text] = pid

    for ex in dataset:
        relevant_ids = []
        for ctx in ex['entity_pages']['wiki_context']:
            if not ctx or len(ctx.strip()) < 50:
                continue
            text = ctx.strip()[:max_len]
            if text in seen:
                relevant_ids.append(seen[text])
        if not relevant_ids:
            continue
        queries.append({
            'id': f'q{len(queries):04d}', 'type': 'factoid',
            'question': ex['question'], 'relevant_ids': relevant_ids,
            'answer': ex['answer']['value'],
        })
        if len(queries) >= n_queries:
            break

    return passages, queries


def load_hotpotqa(n_source=1500, n_queries=500, max_len=400):
    dataset = load_dataset("sentence-transformers/hotpotqa", "triplet",
                            split=f"train[:{n_source}]", trust_remote_code=True)

    passages, queries, seen = [], [], {}

    for ex in dataset:
        for txt in [ex['positive'], ex['negative']]:
            if not txt or len(txt.strip()) < 50:
                continue
            text = txt.strip()[:max_len]
            if text not in seen:
                pid = f'hp_p{len(passages)}'
                passages.append({'id': pid, 'text': text})
                seen[text] = pid

    for ex in dataset:
        positive = ex['positive']
        if not positive or len(positive.strip()) < 50:
            continue
        text = positive.strip()[:max_len]
        if text not in seen:
            continue
        queries.append({
            'id': f'q{len(queries):04d}', 'type': 'multi-hop',
            'question': ex['anchor'], 'relevant_ids': [seen[text]],
            'answer': positive[:100],
        })
        if len(queries) >= n_queries:
            break

    return passages, queries


def load_msmarco(n_source=1400, n_queries=500, max_len=400, batch=600):
    """Loads MS MARCO in batches until n_queries with is_selected=1 are found."""
    passages, queries, seen = [], [], {}

    start = 0
    while len(queries) < n_queries and start < n_source:
        end = min(start + batch, n_source)
        dataset = load_dataset("microsoft/ms_marco", "v2.1",
                                split=f"validation[{start}:{end}]", trust_remote_code=True)

        for ex in dataset:
            for txt in ex['passages']['passage_text']:
                if not txt or len(txt.strip()) < 30:
                    continue
                text = txt.strip()[:max_len]
                if text not in seen:
                    pid = f'ms_p{len(passages)}'
                    passages.append({'id': pid, 'text': text})
                    seen[text] = pid

        for ex in dataset:
            if len(queries) >= n_queries:
                break
            answers = ex['answers']
            answer = answers[0] if answers and answers[0] != 'No Answer Present.' else ''
            if not answer:
                continue

            relevant_ids = []
            for txt, sel in zip(ex['passages']['passage_text'], ex['passages']['is_selected']):
                if not txt or len(txt.strip()) < 30:
                    continue
                text = txt.strip()[:max_len]
                if sel == 1 and text in seen:
                    relevant_ids.append(seen[text])
            if not relevant_ids:
                continue

            queries.append({
                'id': f'q{len(queries):04d}', 'type': ex['query_type'].lower(),
                'question': ex['query'].strip(), 'relevant_ids': relevant_ids,
                'answer': answer,
            })

        start = end

    return passages, queries


def load_natural_questions(n_source=600, n_queries=500, max_len=400):
    dataset = load_dataset("sentence-transformers/natural-questions",
                            split=f"train[:{n_source}]", trust_remote_code=True)

    passages, queries, seen = [], [], {}

    for ex in dataset:
        answer = ex['answer'].strip()
        if not answer or len(answer) < 50:
            continue
        text = answer[:max_len]
        if text not in seen:
            pid = f'nq_p{len(passages)}'
            passages.append({'id': pid, 'text': text})
            seen[text] = pid

    for ex in dataset:
        question = ex['query'].strip()
        answer = ex['answer'].strip()
        if not question or not answer or len(answer) < 50:
            continue
        text = answer[:max_len]
        if text not in seen:
            continue
        queries.append({
            'id': f'q{len(queries):04d}', 'type': 'factoid',
            'question': question, 'relevant_ids': [seen[text]],
            'answer': answer[:100],
        })
        if len(queries) >= n_queries:
            break

    return passages, queries


LOADERS = {
    'triviaqa': load_triviaqa,
    'hotpotqa': load_hotpotqa,
    'msmarco': load_msmarco,
    'nq': load_natural_questions,
}
