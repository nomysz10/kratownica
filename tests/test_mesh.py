from truss.mesh import make_rect_mesh


def test_mesh_sizes():
    m = make_rect_mesh(200.0, 10.0, 2.5)
    assert m.nx == 80 and m.ny == 4
    assert m.nodes.shape == ((81 * 5), 2)
    assert m.elements.shape == (320, 4)


def test_mesh_origin_and_span():
    m = make_rect_mesh(200.0, 10.0, 2.5)
    assert m.nodes[:, 0].min() == 0.0 and m.nodes[:, 0].max() == 200.0
    assert m.nodes[:, 1].min() == 0.0 and m.nodes[:, 1].max() == 10.0


def test_mesh_rejects_non_square():
    import pytest
    with pytest.raises(ValueError):
        make_rect_mesh(200.0, 11.0, 2.5)
