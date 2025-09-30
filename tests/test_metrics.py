import numpy as np
from utils.metrics import mae


def test_mae_basic_vector():
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.0, 1.0, 4.0])
    assert mae(y, yhat) == np.mean([0.0, 1.0, 1.0])


def test_mae_matrix():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    yhat = np.array([[0.0, 2.0], [5.0, 5.0]])
    # abs diffs: [[1,0],[2,1]] -> mean = 1.0
    assert mae(y, yhat) == 1.0


def test_mae_raises_on_shape_mismatch():
    y = np.array([1.0, 2.0])
    yhat = np.array([1.0, 2.0, 3.0])
    try:
        mae(y, yhat)
        assert False, "should raise"
    except ValueError as e:
        assert "shape mismatch" in str(e)


def test_mae_raises_on_empty():
    try:
        mae(np.array([]), np.array([]))
        assert False, "should raise"
    except ValueError as e:
        assert "empty arrays" in str(e)
