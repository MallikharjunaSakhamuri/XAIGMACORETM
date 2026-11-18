"""XAI audit package for the synthetic-data bias revision.

Anaconda ships its own OpenMP runtime (libiomp5md.dll via MKL) and the PyTorch
wheels bundle another. Importing both in one process raises OMP Error #15 on
Windows and aborts. The environment variable below permits the duplicate, and
is set here, before any submodule imports torch, so the behaviour is recorded
in the code rather than depending on shell state.

The duplicate-runtime warning is about numerical reproducibility under heavy
multi-threaded BLAS. The guard against that here is empirical: fine-tuning runs
three seeds per configuration and reports the spread, so a result that moved
because of thread non-determinism would show up as seed variance.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
