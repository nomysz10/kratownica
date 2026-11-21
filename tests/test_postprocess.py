import numpy as np

from truss.postprocess import (
    enforce_min_width,
    keep_largest_component,
    min_feature_width,
    skeletonize_members,
)


def test_min_width_enforced():
    img = np.zeros((100, 100), dtype=bool)
    img[:, 49:51] = True            # pręt 2 mm (pixel = 1 mm)
    out = enforce_min_width(img, pixel_size=1.0, min_mm=4.0)
    assert min_feature_width(out, 1.0) >= 4.0


def test_min_width_measures_thinnest_member():
    img = np.zeros((100, 100), dtype=bool)
    img[:, 10:12] = True            # pręt 2 mm
    img[:, 40:48] = True            # pręt 8 mm
    assert min_feature_width(img, 1.0) < 4.0      # mierzy najcieńszy, nie najgrubszy
    out = enforce_min_width(img, pixel_size=1.0, min_mm=4.0)
    assert min_feature_width(out, 1.0) >= 4.0


def test_largest_component():
    img = np.zeros((50, 50), dtype=bool)
    img[10:40, 10:40] = True        # główny blok
    img[0:3, 0:3] = True            # wyspa odłączona
    out = keep_largest_component(img)
    assert out[0:3, 0:3].sum() == 0
    assert out[10:40, 10:40].all()


def test_skeleton_member():
    img = np.zeros((30, 70), dtype=bool)
    img[10:20, 5:65] = True         # prostokąt 60×10 mm
    members = skeletonize_members(img, pixel_size=1.0)
    assert len(members) <= 3
    assert any(abs(m.length - 60.0) < 6.0 and 8.0 < m.width < 12.0 for m in members)
