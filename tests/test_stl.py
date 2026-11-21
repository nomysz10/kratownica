import numpy as np

from truss.stl import extrude_contour_to_triangles, write_binary_stl


def test_extrude_rectangle_triangle_count_and_z():
    poly = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
    tris = extrude_contour_to_triangles(poly, height=4.0)
    assert tris.shape == (12, 3, 3)          # 4 ściany + 2 pokrywy
    assert set(np.round(tris[:, :, 2].ravel(), 6)) <= {0.0, 4.0}


def test_binary_stl_roundtrip(tmp_path):
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    p = tmp_path / "a.stl"
    write_binary_stl(str(p), tri)
    data = p.read_bytes()
    assert data[:80] == b" " * 80          # nagłówek
    assert int.from_bytes(data[80:84], "little") == 1
    assert len(data) == 84 + 50 * 1
