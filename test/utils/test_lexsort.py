import numpy as np
import paddle

from paddle_geometric.utils import lexsort


def test_lexsort():
    keys = [paddle.randn(100) for _ in range(3)]

    expected = np.lexsort([key.numpy() for key in keys])
    assert paddle.equal_all(lexsort(keys), paddle.from_numpy(expected)).item()
