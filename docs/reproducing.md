# Reproducing the paper

## 1. Environment

Python 3.10 or newer. Install `torch` and `torch-geometric` for your CUDA version first, following the instructions at pytorch.org and pytorch-geometric.readthedocs.io, then:

```bash
pip install -r requirements.txt
python scripts/check_environment.py
```

`check_environment.py` reports which packages are importable and whether a GPU is visible. Stages 1, 1b, 3, 6 and 7 run on CPU. Stages 2, 4 and 5 use a GPU if one is present.

## 2. Data

Expected layout, one directory per benchmark, matching the MoleculeNet distribution used by MolCLR:

```
<DATA>/
  bace/bace.csv
  esol/esol.csv
  clintox/clintox.csv
  qm9/qm9.csv
  pubchem-10m-clean.txt
```

The loader resolves file names and column names by alias, so `delaney-processed.csv` in place of `esol.csv`, or a `mol` column in place of `smiles`, are both handled.

Sources:

- BACE, ESOL, ClinTox, QM9: MoleculeNet, http://moleculenet.org
- PubChem 10M clean SMILES: the list released with MolCLR, https://github.com/yuyangw/MolCLR

QM9 has six target properties. All analysis in the paper uses the HOMO-LUMO gap; pass `--qm9-target gap` where a stage accepts it.

## 3. Checkpoints

Stages 2, 4 and 5 read:

```
<RUNS>/
  manualaug_cl_fixed/best_model.pt, feature_stats.pt, run_config.json
  gmacore_tm_fixedgan/best_model.pt, feature_stats.pt, run_config.json
```

Both are produced by `notebooks/06_pretrain_gmacore_tm.ipynb` and `notebooks/07_pretrain_manualaug_cl.ipynb` after `notebooks/01_prepare_graphs.ipynb`. Pretraining both encoders takes several GPU-hours. The analysis does not require re-running it.

The encoder is reconstructed in `xai_audit/gmacore_model.py` and loads `best_model.pt` with `strict=True`, so a silent architecture mismatch is not possible.

## 4. Stages

Run in this order. `<DATA>` and `<RUNS>` are the directories above.

```bash
# Data-level profile, no checkpoint, ~2 minutes
python scripts/run_01_data_profile.py --data-root <DATA> --output-dir results/table_a --qm9-target gap

# Pretraining corpus profile, no checkpoint, ~20 minutes
python scripts/run_01b_corpus_profile.py --corpus <DATA>/pubchem-10m-clean.txt --data-root <DATA> --output-dir results/corpus_profile

# Predictions from both checkpoints, GPU, ~30 minutes
python scripts/run_02_predict.py --data-root <DATA> --runs-root <RUNS> --output-dir results/predictions

# Table B, seconds, both splits
python scripts/run_03_bias.py --predictions results/predictions --output-dir results/table_b --split test
python scripts/run_03_bias.py --predictions results/predictions --output-dir results/table_b --split all

# Figure 9, seconds
python scripts/run_06_figures.py --table-b results/table_b/table_b_aggregated.csv --output-dir results/figures

# Generator gates, GPU, ~5 minutes
python scripts/run_05_gates.py --data-root <DATA> --runs-root <RUNS> --output-dir results/generator_gates

# Leakage, ~15 minutes
python scripts/run_07_leakage.py --data-root <DATA> --corpus <DATA>/pubchem-10m-clean.txt --output-dir results/leakage

# Attribution, GPU, longest stage
python scripts/run_04_attribution.py --data-root <DATA> --runs-root <RUNS> --output-dir results/table_b_attribution --n-molecules 150
```

Then the figure stages:

```bash
# Figures 2 and 4, GPU, KernelSHAP over every dataset and both models
python scripts/run_08_shap_figures.py --data-root <DATA> --runs-root <RUNS> --output-dir results/figures_shap --n-molecules 150

# Figure 3, GPU. The paper shows BACE and ESOL
python scripts/run_09_counterfactual.py --data-root <DATA> --runs-root <RUNS> --output-dir results/figures_counterfactual --datasets bace esol

# Figure 6, GPU, one molecule. Pass --smiles to pick the molecule explicitly
python scripts/run_10_local_explanations.py --data-root <DATA> --runs-root <RUNS> --output-dir results/figures_local --dataset bace

# Figures 7 and 8, seconds, no checkpoint, reads the stage 2 CSVs
python scripts/run_11_fairness_diagnostics.py --predictions results/predictions --output-dir results/figures_fairness
```

Stage 8 runs KernelSHAP once per molecule per dataset per model, so it is the
longest of the figure stages. Reduce `--n-molecules` for a first pass.

Stage 9 needs attributions before it can rank motifs, so it runs KernelSHAP too.
`--top-k` defaults to 3, matching Appendix Table A.4.

Stage 10 picks the molecule with the largest absolute prediction unless you pass
`--smiles`. Pass the SMILES explicitly to regenerate the exact molecule shown in
the paper. `--atom` selects the atom for the LIME panel; the default is the atom
with the largest absolute contribution.

Stage 11 is the one to run first when checking that everything is wired up: it
needs no GPU and no checkpoint.

Stage 4 fine-tunes eight probes and then runs KernelSHAP on each, so run it once the other stages confirm the paths resolve.

## 5. What to expect

Stage 8 skips Figure 4 for a dataset and model when fewer than three molecules
fall in either novelty arm, which happens if `--n-molecules` is small; it says so
rather than drawing an empty panel. Stage 9 reports how many ablations produced a
valid molecule, and the Spearman correlation between absolute attribution and
absolute delta. Stage 10 and stage 11 skip the LIME and DALEX panels with a
message if `lime` or `dalex` is not installed.

Stage 3 prints the family size, the number of underpowered groups excluded, the count significant after Benjamini-Hochberg correction, and the direction of change for every group. Stage 4 prints the probe's validation score before each KernelSHAP run; the ESOL and QM9 probes do not exceed a mean predictor on the validation scaffold fold, which is stated in the paper and is not a configuration error.

Results are seed-sensitive at the third decimal place. Stage 2 defaults to three seeds; `--seeds` takes more. Headline counts in the paper are stable across seed choices, individual group-level differences between the two models are not, which is why the paper does not claim any.

## 6. Publishing results

After a run, copy the outputs into `results/`:

```powershell
.\tools\collect_results.ps1 -RunRoot "<your run>\results"
```

Check `git status` before committing. Per-molecule prediction CSVs and model checkpoints are excluded by `.gitignore` and should stay out of the repository.
