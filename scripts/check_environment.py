r"""Confirm the environment is ready for stage 2.

    python scripts\check_environment.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    import numpy
    import pandas
    import rdkit
    import sklearn
    import torch
    import torch_geometric

    print(f"python           : {sys.version.split()[0]}")
    print(f"torch            : {torch.__version__}")
    print(f"torch_geometric  : {torch_geometric.__version__}")
    print(f"rdkit            : {rdkit.__version__}")
    print(f"numpy / pandas   : {numpy.__version__} / {pandas.__version__}")
    print(f"scikit-learn     : {sklearn.__version__}")
    print(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        probe = torch.randn(512, 512, device="cuda")
        _ = (probe @ probe).sum().item()
        del probe
        torch.cuda.empty_cache()
        print("allocation test  : OK")
    else:
        print("  Running on CPU. The small benchmarks are fine; QM9 will be slow.")

    from torch_geometric.nn import GCNConv, global_mean_pool  # noqa: F401
    from xai_audit.gmacore_model import GraphEncoder, featurize

    graph = featurize("O=S(=O)(N)c1ccc(F)cc1CC(=O)O")
    encoder = GraphEncoder()
    mean = torch.zeros(10)
    std = torch.ones(10)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    out = encoder.encode(graph, mean, std, 10.0)
    print(f"encoder forward  : OK, output {tuple(out.shape)}")
    print("\nEnvironment ready.")


if __name__ == "__main__":
    main()
