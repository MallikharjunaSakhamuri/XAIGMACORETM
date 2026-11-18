# Notebooks

These are the notebooks that prepared the data and pretrained the two encoders. They are published as they were run, on a cloud GPU instance. Paths inside them point at that instance's working directory and will need changing.

| Notebook | Purpose |
| --- | --- |
| `01_prepare_graphs.ipynb` | Reads the PubChem SMILES list, de-duplicates, converts to PyTorch Geometric graphs and writes sharded tensors. Caps the corpus at 100,000 unique molecules. |
| `06_pretrain_gmacore_tm.ipynb` | Pretrains GMACORE-TM: generative augmentation, momentum encoder, age-weighted negatives, memory queue. |
| `07_pretrain_manualaug_cl.ipynb` | Pretrains ManualAug-CL: handcrafted augmentation, in-batch negatives, no queue. |
| `11_downstream_eval.ipynb` | Fine-tunes and evaluates both encoders on the MoleculeNet benchmarks. |

Running the notebooks is not required to reproduce the paper's analysis. Stages 2, 4 and 5 read the saved checkpoints directly.

## Known issue in `11_downstream_eval.ipynb`

In `evaluate()`, regression predictions are returned to original units with `outputs = outputs * scaler[1] + scaler[0]`, but the targets are left standardized. RMSE, MAE and R2 therefore compare quantities on different scales and are not interpretable as reported; Pearson and Spearman correlations are unaffected because they are scale-invariant.

The symptom is visible in the original QM7 numbers, where targets are of order -1500 and the reported RMSE is 1531.

`xai_audit/finetune.py` inverse-transforms both sides and is what every result in `results/` was computed with. The notebook is kept unmodified so that the published figures of the original submission can be traced to the code that produced them.
