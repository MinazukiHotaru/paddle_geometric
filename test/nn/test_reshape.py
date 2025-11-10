import paddle

from paddle_geometric.nn.reshape import Reshape


def test_reshape():
    x = paddle.randn(10, 4)

    op = Reshape(5, 2, 4)
    assert str(op) == 'Reshape(5, 2, 4)'

    assert op(x).size() == (5, 2, 4)
    assert paddle.equal_all(op(x).view(10, 4), x).item()
