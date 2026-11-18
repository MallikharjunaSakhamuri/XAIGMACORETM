# Results

Every table and figure reported in the paper. All files are produced by the scripts in `../scripts/`; nothing here is hand-edited.

| Folder | Files | Reported as |
| --- | --- | --- |
| `table_a/` | `table_a_groups.csv`, `table_a_rule_partitions.csv` | Table A, benchmark data profile |
| `corpus_profile/` | `corpus_motif_profile.csv`, `corpus_property_summary.csv`, `corpus_vs_benchmark_motifs.csv`, `corpus_rule_compliance.csv` | Table A, pretraining corpus profile |
| `table_b/` | `table_b_aggregated.csv`, `table_b_model_contrast.csv`, `table_b_per_seed.csv`, and `_allsplits` variants | Table B, group disparity |
| `table_b_attribution/` | `attribution_summary.csv`, `motif_importance.csv`, `settings.json` | Table B, attribution |
| `generator_gates/` | `generator_gate_profile.csv` | Figure 10 |
| `figures/` | `figure9_slope_<dataset>` and `figure9_amplification_<dataset>`, PNG and PDF | Figure 9 |
| `leakage/` | `leakage_summary.csv`, `leakage_similarity_distribution.csv`, `settings.json` | Leakage analysis |
| `figures_shap/` | `figure2_motif_attribution_*`, `figure4_novelty_*`, `attribution_long.csv` | Figures 2 and 4 |
| `figures_counterfactual/` | `figure3_ablation_*`, `figure3_summary_*`, `counterfactual_ablations.csv`, `counterfactual_concordance.csv` | Figure 3 |
| `figures_local/` | `figure6a_atom_map_*`, `figure6b_atom_ranking_*`, `figure6c_waterfall_*`, `figure6d_lime_*`, plus the two CSVs | Figure 6 |
| `figures_fairness/` | `figure7a_*`, `figure7b_*`, `figure8a_radar_*`, `figure8b_gaps_*`, plus three CSVs | Figures 7 and 8 |
| `predictions/` | `performance_gate.csv`, `manifest.json` | Predictive performance, context only |

Per-molecule prediction CSVs are not committed. They are large, they are reproducible from stage 2, and `performance_gate.csv` carries the summary that the paper cites.

`_allsplits` files pool training, validation and test molecules for statistical power. The held-out files without that suffix are the primary analysis. Any statement in the paper drawn from the pooled version says so.
