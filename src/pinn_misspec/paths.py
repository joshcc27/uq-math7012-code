"""Where results are read and written.

The results live outside the package: by default in ``results/`` at the root of
the repository, or wherever ``PINN_MISSPEC_RESULTS`` points.
"""

import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("PINN_MISSPEC_RESULTS", REPOSITORY / "results"))
OSCILLATOR = RESULTS / "oscillator"
ADVECTION_DIFFUSION = RESULTS / "advection_diffusion"
THESIS_FIGURES = RESULTS / "thesis_figures"
