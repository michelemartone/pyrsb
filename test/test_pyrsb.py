from rsb import rsb_matrix


def test_init_tuple():
    V = [11.0, 12.0, 22.0]
    I = [0, 0, 1]
    J = [0, 1, 1]
    mat = rsb_matrix((V, I, J))
    assert mat.shape == (2, 2)
