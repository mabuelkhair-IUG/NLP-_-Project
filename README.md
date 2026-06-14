# RAG Retrieval Strategy Comparison

A reproducible, statistically-validated comparison of **BM25**, **Dense Passage Retrieval (DPR)**, **Hybrid-RRF**, and **Hybrid-α** retrieval strategies for Retrieval-Augmented Generation (RAG), evaluated on **7 open-retrieval benchmarks** totaling **2,961 queries**.

📄 Full paper: [`paper/RAG_Final_v7.docx`](paper/RAG_Final_v7.docx)

---

## Key Finding

**No single retrieval strategy is universally optimal** — the best method depends on corpus size and query type:

| Dataset | Corpus | Best Method | MRR | p-value vs BM25 |
|---|---|---|---|---|
| TriviaQA | 947 | Hybrid-α (α=0.3) | 0.926 | p < 0.001 *** |
| HotpotQA | 2,986 | Hybrid-RRF | 0.908 | p = 0.001 ** |
| MS MARCO | 13,868 | DPR (pure) | 0.589 | p < 0.001 *** |
| Natural Questions | 596 | Hybrid-α (α=0.3) | 0.962 | p < 0.001 *** |
| SQuAD 2.0 | 500 | Hybrid-RRF | 0.769 | p < 0.001 *** |
| FEVER | 542 | Hybrid-α (α=0.3) | 0.956 | p < 0.001 *** |
| Wikipedia 2023 | 13,000 | Hybrid-RRF | 0.937 | p = 0.005 ** |

All results use **independent open-retrieval corpora** (not closed/query-specific pools), with **95% bootstrap confidence intervals** and **paired t-tests** vs. BM25.

---

## Repository Structure

```
rag_repo/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── retrievers.py                  # BM25Retriever, DPRRetriever, HybridRetriever (RRF + alpha)
│   ├── metrics.py                     # P@k, R@k, MRR, significance tests, bootstrap CI
│   ├── data_loaders.py                # Open-retrieval corpus builders for all 7 datasets
│   ├── run_experiment.py              # CLI runner — one dataset end to end
│   └── combined_results_and_charts.py # Generates all figures from final results (7 datasets)
├── notebooks/
│   └── RAG_Comparison_Colab.ipynb    # Step-by-step Colab walkthrough
├── figures/
│   ├── master_7ds.png                 # Combined: Heatmap + 7 bar charts + line trend
│   ├── fig1_heatmap.png               # MRR heatmap (7 datasets × 4 methods)
│   ├── fig2_squad_fever.png           # SQuAD 2.0 & FEVER detailed
│   ├── fig3_line_trend.png            # MRR trend lines with 95% CI bands
│   ├── fig4_squad_detail.png          # SQuAD 2.0 analysis (MRR, p-value, delta)
│   ├── fig5_fever_detail.png          # FEVER analysis (MRR, p-value, delta)
│   └── pilot_phase/                   # Phase 2 closed-corpus figures (superseded)
├── paper/
│   └── RAG_Final_v7.docx              # Full paper (7 datasets, 25 references, appendix)
└── results/                           # Populated when run_experiment.py is run
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run any dataset (500 queries, open-retrieval corpus)
python code/run_experiment.py --dataset triviaqa        --n_queries 500
python code/run_experiment.py --dataset hotpotqa        --n_queries 500
python code/run_experiment.py --dataset msmarco         --n_queries 500
python code/run_experiment.py --dataset nq              --n_queries 500
python code/run_experiment.py --dataset squad           --n_queries 500
python code/run_experiment.py --dataset fever           --n_queries 500
python code/run_experiment.py --dataset wikipedia2023   --n_queries 500
```

Each run saves:
- `results/<dataset>_results.csv` — MRR, P@5, R@5, Latency per method
- `results/<dataset>_significance.json` — p-values and 95% bootstrap CIs

To regenerate all figures from final results (no re-running experiments):
```bash
python code/combined_results_and_charts.py
```

---

## Methods

| Method | Type | Description | Key Params |
|---|---|---|---|
| **BM25** | Sparse | Lexical retrieval via `rank_bm25` | k1=1.5, b=0.75 |
| **DPR** | Dense | `sentence-transformers` + FAISS IndexFlatIP | `multi-qa-MiniLM-L6-cos-v1`, 384-dim |
| **Hybrid-RRF** | Hybrid | Reciprocal Rank Fusion of BM25 + DPR | k=60, candidate_factor=3 |
| **Hybrid-α** | Hybrid | Alpha-weighted score fusion of BM25 + DPR | α=0.3 (DPR-dominant, optimal) |

---

## Datasets — All 7 Open-Retrieval Corpora

| Dataset | Queries | Corpus | Query Type | HuggingFace Source |
|---|---|---|---|---|
| TriviaQA | 500 | 947 | Factoid | `mandarjoshi/trivia_qa` (rc.wikipedia) |
| HotpotQA | 500 | 2,986 | Multi-hop | `sentence-transformers/hotpotqa` (triplet) |
| MS MARCO | 500 | 13,868 | Heterogeneous | `microsoft/ms_marco` (v2.1) |
| Natural Questions | 500 | 596 | Open-domain factoid | `sentence-transformers/natural-questions` |
| SQuAD 2.0 | 500 | 500 | Reading comprehension | `rajpurkar/squad_v2` |
| FEVER | 461 | 542 | Fact verification | `copenlu/fever_gold_evidence` |
| Wikipedia 2023 | 500 | 13,000 | Encyclopedic | `wikimedia/wikipedia` (20231101.en) |

Each query is evaluated against the **full shared corpus** of its dataset — genuine open retrieval, not a query-specific subset.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| DPR model | multi-qa-MiniLM-L6-cos-v1 (384-dim) |
| DPR pre-training | 10 epochs, batch_size=64, in-batch negatives |
| Fine-tuning | None (zero-shot evaluation) |
| GPU | NVIDIA T4 (Google Colab) |
| Max passage length | 400 characters |
| Bootstrap resamples | 1,000 |
| Significance threshold | p < 0.05 (paired t-test) |

---

## Ablation Study

α was swept over {0.1, ..., 0.9} on a 100-query pilot (see `figures/pilot_phase/`).
**α=0.3** (DPR-dominant, 70% DPR weight) achieved the best average MRR (0.831) and was
carried forward to all Phase 3 experiments.

---

## References

This project is based on the following key works:

1. Lewis et al. (2020) — RAG [NeurIPS]
2. Karpukhin et al. (2020) — DPR [EMNLP]
3. Robertson & Zaragoza (2009) — BM25 [Foundations & Trends]
4. Cormack et al. (2009) — RRF [SIGIR]
5. Devlin et al. (2019) — BERT [NAACL]
6. Thakur et al. (2021) — BEIR [NeurIPS]
7. Yang et al. (2018) — HotpotQA [EMNLP]
8. Joshi et al. (2017) — TriviaQA [ACL]
9. Rajpurkar et al. (2018) — SQuAD 2.0 [ACL]
10. Thorne et al. (2018) — FEVER [NAACL]
11. Formal et al. (2021) — SPLADE [SIGIR]
12. Yu et al. (2024) — RankRAG [arXiv]

Full 25-reference list available in the paper.

---

## License

MIT
