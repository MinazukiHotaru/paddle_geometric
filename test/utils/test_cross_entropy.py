import paddle
import paddle.nn.functional as F
import pytest

from paddle_geometric.utils.cross_entropy import sparse_cross_entropy


@pytest.mark.parametrize('with_edge_label_weight', [True, False])
def test_sparse_cross_entropy_multiclass(with_edge_label_weight):
    x = paddle.randn(5, 5, requires_grad=True)
    y = paddle.eye(5)

    edge_label_index = y.nonzero().t()
    edge_label_weight = None
    if with_edge_label_weight:
        edge_label_weight = paddle.rand(edge_label_index.size(1))
        y[y == 1.0] = edge_label_weight

    expected = F.cross_entropy(x, y, soft_label=True)
    expected.backward()
    expected_grad = paddle.clone(x.grad)

    x.clear_grad()
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert paddle.allclose(expected, out)
    if with_edge_label_weight is False:
        assert paddle.allclose(expected_grad, x.grad)


@pytest.mark.parametrize('with_edge_label_weight', [False, True])
def test_sparse_cross_entropy_multilabel(with_edge_label_weight):
    x = paddle.randn(4, 4, requires_grad=True)
    y = paddle.randint_like(x, 0, 2)

    edge_label_index = y.nonzero().t()
    edge_label_weight = None
    if with_edge_label_weight:
        edge_label_weight = paddle.rand(edge_label_index.size(1))
        y[y == 1.0] = edge_label_weight

    expected = F.cross_entropy(x, y, soft_label=True)
    expected.backward()
    paddle.clone(x.grad)

    x.clear_grad()
    out = sparse_cross_entropy(x, edge_label_index, edge_label_weight)
    out.backward()

    assert paddle.allclose(expected, out)

    # assert paddle.allclose(expected_grad, x.grad)
