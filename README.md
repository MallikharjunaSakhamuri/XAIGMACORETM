# XAIGMACORETM

Code and results for **"Explainable Analysis of Synthetic-Data Bias in Molecular Graph Prediction for Immunology"** (submitted to *Neurocomputing*).

The paper audits how subgroup disparities present in molecular benchmark data propagate through self-supervised graph contrastive pretraining, and what explainability reveals about the resulting models. Two encoders are compared: **GMACORE-TM**, which uses generative augmentation with a momentum encoder and age-weighted negatives, and **ManualAug-CL**, which uses handcrafted augmentations and no generative component.

The encoders themselves are introduced in a companion study; their implementation lives at [GMACORETM](https://github.com/MallikharjunaSakhamuri/GMACORETM). This repository contains the explainability audit built on top of them.

## What is here

| Path | Contents |
| --- | --- |
| `xai_audit/` | The audit library: motif library, domain rule sets, disparity measures, KernelSHAP runner, generator gate profiling, leakage checks, encoder and fine-tuning code |
| `scripts/` | One script per analysis stage, runnable from the command line |
| `notebooks/` | Data preparation, pretraining of both encoders, and downstream evaluation |
| `results/` | Every table and figure reported in the paper, as CSV and PDF |
| `docs/reproducing.md` | Step-by-step reproduction, including expected runtimes |

## Quick start

```bash
git clone https://github.com/MallikharjunaSakhamuri/XAIGMACORETM.git
cd XAIGMACORETM

# Install torch and torch-geometric for your CUDA version first, then:
pip install -r requirements.txt
python scripts/check_environment.py
```

The analysis stages read a MoleculeNet-style data directory and the two pretrained checkpoints. Neither is vendored here; see `docs/reproducing.md` for how to obtain both.

```bash
python scripts/run_01_data_profile.py  --data-root <DATA> --output-dir results/table_a --qm9-target gap
python scripts/run_01b_corpus_profile.py --corpus <DATA>/pubchem-10m-clean.txt --data-root <DATA> --output-dir results/corpus_profile
python scripts/run_02_predict.py       --data-root <DATA> --runs-root <RUNS> --output-dir results/predictions
python scripts/run_03_bias.py          --predictions results/predictions --output-dir results/table_b --split test
python scripts/run_04_attribution.py   --data-root <DATA> --runs-root <RUNS> --output-dir results/table_b_attribution
python scripts/run_05_gates.py         --data-root <DATA> --runs-root <RUNS> --output-dir results/generator_gates
python scripts/run_06_figures.py       --table-b results/table_b/table_b_aggregated.csv --output-dir results/figures
python scripts/run_07_leakage.py       --data-root <DATA> --corpus <DATA>/pubchem-10m-clean.txt --output-dir results/leakage
```

## Analysis stages

| Stage | Script | Produces | Needs a checkpoint |
| --- | --- | --- | --- |
| 1 | `run_01_data_profile.py` | Table A: benchmark subgroup prevalence, label rates, rule partitions | No |
| 1b | `run_01b_corpus_profile.py` | Table A: pretraining corpus composition and its distance from each benchmark | No |
| 2 | `run_02_predict.py` | Per-molecule test predictions for both encoders, across seeds | Yes |
| 3 | `run_03_bias.py` | Table B: group disparity, amplification, Benjamini-Hochberg correction | No |
| 4 | `run_04_attribution.py` | Table B: KernelSHAP attribution concentration and explanation stability | Yes |
| 5 | `run_05_gates.py` | Figure 10: generator feature-gate profile per motif | Yes |
| 6 | `run_06_figures.py` | Figure 9: label gap against each model's predicted gap, and amplification | No |
| 7 | `run_07_leakage.py` | Pretraining corpus against held-out benchmark overlap | No |

Stage 4 is the only expensive one. Stages 1b and 7 stream the pretraining corpus and take minutes.

## Methodological notes

These are deliberate choices that differ from a naive implementation, and they affect the numbers.

**KernelSHAP settings.** Attribution uses `l1_reg = 0.0` and `nsamples = 2M + 2048`. The exploratory notebooks behind the original submission used `l1_reg = "num_features(10)"` with `nsamples = 200`, which applies AIC feature selection and forces all but ten attributions to exactly zero. That distorts any measure of attribution concentration, so all attribution results here were recomputed with regularisation disabled.

**Regression targets.** Downstream regression metrics inverse-transform both predictions and targets to original units. `notebooks/11_downstream_eval.ipynb` inverse-transforms predictions only, which leaves RMSE, MAE and R2 comparing quantities on different scales; Pearson and Spearman are unaffected. The notebook is published as it was run, and `notebooks/README.md` records the discrepancy. The corrected behaviour is in `xai_audit/finetune.py`.

**Underpowered groups.** Groups whose smaller arm falls below 20 molecules are computed, flagged with `underpowered = True`, and excluded from the hypothesis-test family. They are reported but not interpreted.

**Amplification.** Defined as `abs(model gap) - abs(label gap)`. Signed gaps are retained separately, because a sign flip is not the same finding as a magnitude increase.

**Scope of the model contrast.** ManualAug-CL differs from GMACORE-TM in the momentum encoder and age-weighted negative sampling as well as the augmentation operator. Contrast columns are labelled `amplified_under_gmacore_tm` and `reduced_under_gmacore_tm` rather than attributed to the generator, because the comparison supports claims about the complete training pipeline and not a causal estimate for the generative component alone.

## Data

Benchmarks are the MoleculeNet distributions of BACE, ESOL, ClinTox and QM9. Pretraining uses the PubChem 10M clean SMILES list. Neither is redistributed here; `docs/reproducing.md` gives the sources and the expected directory layout.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
