import os

import numpy as np

from truss.optimize import run_optimization
from truss.report import write_report


def test_run_optimization_smoke(tmp_path):
    r = run_optimization(span=100.0, max_mass_g=15.0, max_deflection=20.0,
                         d_range=(20.0, 30.0), h_range=(3.0, 4.0), el_size=2.0)
    assert r.p_max > 0 and np.isfinite(r.p_max)
    # 4 mm min width is a hard floor; mass may exceed the soft budget a bit
    assert r.mass_g > 0
    paths = write_report(r, str(tmp_path))
    assert os.path.exists(paths["stl"])
    assert os.path.getsize(paths["stl"]) > 0
    assert os.path.exists(paths["png"])
    assert "p_max" in open(paths["txt"]).read()
