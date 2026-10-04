# tcm-stroke-llm-eval

Evaluation and analysis code for:

> **Framework-migration evaluation of large language models on Traditional Chinese Medicine
> stroke records: task constructs, reference labels, and evidential boundaries**
> Jian Wang, Rende Huang, Feng Gao*

This repository contains the scripts that build the evaluation sets, run the prompting
matrix, compute metrics, and render the manuscript figures. It accompanies a
descriptive framework-migration study; the study does not distribute the source
clinical text (see [Data](#data)).

## What this code does

| Stage | Script |
|---|---|
| Build evaluation sets (T1 / T2 / T3) | `analysis/build_eval_set.py`, `analysis/build_tasks23.py`, `analysis/build_task3_stage.py` |
| Prompt templates (P1–P4, zh/en) | `analysis/prompts.py` |
| Run the model matrix | `analysis/runner.py` |
| Metrics (macro-F1, bootstrap 95% CI, McNemar) | `analysis/metrics.py` |
| Output-protocol audit | `analysis/audit_output_protocol.py` |
| Equivalence / ablation | `analysis/equivalence.py`, `analysis/ablation.py` |
| Final matrix and reporting | `analysis/final_matrix_v2.py`, `analysis/final_report*.py` |
| Figures (main + supplementary) | `analysis/figure_pack_v3.py`, `analysis/figure_pack_fendo_supp.py` |
| Peak CIs | `analysis/peak_ci_v1.py` |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The source corpus is the publicly available **TCM-SD** dataset:

- Ren C, Huang H, Zhou Y, Cao Y, et al. *TCM-SD: A Benchmark for Probing Syndrome
  Differentiation via Natural Language Processing.* CCL 2022, LNCS.
  doi:10.1007/978-3-031-18315-7_16
- Release used: `TCM_SD_with_knowledge`
- Obtain from: https://tianchi.aliyun.com/dataset/dataDetail?dataId=139034
- License: **CC BY-NC-SA 4.0**

The original clinical text is **not** redistributed here and is not included in this
repository. Place the dataset files under `data/` (see `data/README.md`) before running
the build scripts.

## License

Code in this repository is released under the **MIT License** (see `LICENSE`).

Derived artifacts generated from TCM-SD (task sets, predictions, metric tables) are
derived works of a CC BY-NC-SA 4.0 dataset and must therefore be shared under the same
**CC BY-NC-SA 4.0** terms.

## Citation

See `CITATION.cff`.
