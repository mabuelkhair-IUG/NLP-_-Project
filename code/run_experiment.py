"""
run_experiment.py — Run the full BM25/DPR/Hybrid-RRF/Hybrid-alpha comparison
on a single dataset with statistical significance testing.

Usage:
    python run_experiment.py --dataset triviaqa --n_queries 500
    python run_experiment.py --dataset hotpotqa --n_queries 500
    python run_experiment.py --dataset msmarco  --n_queries 500
    python run_experiment.py --dataset nq       --n_queries 500

Outputs:
    results/<dataset>_results.csv       — main MRR/P@5/R@5/Latency table
    results/<dataset>_significance.json — p-values and 95% CIs
"""

import argparse
import json
import os

import pandas as pd

from data_loaders import LOADERS
from retrievers import BM25Retriever, DPRRetriever, HybridRetriever
from metrics import evaluate, paired_significance, bootstrap_ci, highlight_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=LOADERS.keys())
    parser.add_argument('--n_queries', type=int, default=500)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--rrf_k', type=int, default=60)
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f'Loading {args.dataset}...')
    passages, queries = LOADERS[args.dataset](n_queries=args.n_queries)
    print(f'✅ Corpus: {len(passages)} passages | Queries: {len(queries)}')

    print('\nBuilding indexes...')
    bm25 = BM25Retriever(passages)
    bm25.build_index()

    dpr = DPRRetriever(passages)
    dpr.build_index()

    hybrid_rrf = HybridRetriever(bm25, dpr, fusion='rrf', rrf_k=args.rrf_k)
    hybrid_alpha = HybridRetriever(bm25, dpr, fusion='alpha', alpha=args.alpha)

    print(f'\n🔬 Running experiments on {len(queries)} queries...\n')
    results = [
        evaluate('BM25', bm25, queries, args.top_k, with_per_query=True),
        evaluate('DPR', dpr, queries, args.top_k, with_per_query=True),
        evaluate('Hybrid-RRF', hybrid_rrf, queries, args.top_k, with_per_query=True),
        evaluate(f'Hybrid-a{args.alpha}', hybrid_alpha, queries, args.top_k, with_per_query=True),
    ]

    df = pd.DataFrame([{k: v for k, v in r.items() if k != '_mrr_list'} for r in results]).set_index('Method')
    print(df)

    csv_path = os.path.join(args.outdir, f'{args.dataset}_results.csv')
    df.to_csv(csv_path)
    print(f'\n✅ Saved {csv_path}')

    # Statistical significance vs BM25
    print('\nStatistical Significance (paired t-test on MRR, vs BM25)')
    baseline = results[0]['_mrr_list']
    sig_results = {'corpus_size': len(passages), 'n_queries': len(queries)}

    for r in results:
        lo, hi = bootstrap_ci(r['_mrr_list'])
        entry = {'MRR': r['MRR'], 'CI': [round(lo, 4), round(hi, 4)]}
        if r['Method'] != 'BM25':
            t_stat, p_value = paired_significance(r['_mrr_list'], baseline)
            entry['p_value'] = round(p_value, 6)
            sig = '✅' if p_value < 0.05 else '❌'
            print(f"  {r['Method']:15s} MRR={r['MRR']:.4f} [{lo:.4f},{hi:.4f}]  p={p_value:.6f} {sig}")
        else:
            print(f"  {r['Method']:15s} MRR={r['MRR']:.4f} [{lo:.4f},{hi:.4f}]  (baseline)")
        sig_results[r['Method']] = entry

    json_path = os.path.join(args.outdir, f'{args.dataset}_significance.json')
    with open(json_path, 'w') as f:
        json.dump(sig_results, f, indent=2)
    print(f'\n✅ Saved {json_path}')


if __name__ == '__main__':
    main()
