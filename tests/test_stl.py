import numpy as np

from truss.stl import extrude_binary_to_triangles, mesh_volume, write_binary_stl


def test_extrude_binary_solid_volume():
    binary = np.ones((3, 2), dtype=bool)
    tris = extrude_binary_to_triangles(binary, el_size=2.0, H=4.0)
    assert abs(mesh_volume(tris) - 3 * 2 * 2.0 ** 2 * 4.0) < 1e-6


def test_extrude_binary_donut_not_solid():
    binary = np.ones((5, 5), dtype=bool)
    binary[1:4, 1:4] = False          # otwór 3x3 -> materiał 25-9=16 komórek
    tris = extrude_binary_to_triangles(binary, el_size=1.0, H=4.0)
    vol = mesh_volume(tris)
    assert abs(vol - 16 * 4.0) < 1e-6
    assert vol < 25 * 4.0             # otwór nie jest wypełniony


def test_binary_stl_roundtrip(tmp_path):
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    p = tmp_path / "a.stl"
    write_binary_stl(str(p), tri)
    data = p.read_bytes()
    assert data[:80] == b" " * 80
    assert int.from_bytes(data[80:84], "little") == 1
    assert len(data) == 84 + 50 * 1
