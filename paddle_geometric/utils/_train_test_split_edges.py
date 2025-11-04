import math

import paddle

import paddle_geometric
from paddle_geometric.deprecation import deprecated
from paddle_geometric.utils import to_undirected


@deprecated("use 'transforms.RandomLinkSplit' instead")
def train_test_split_edges(
        data: "paddle_geometric.data.Data", val_ratio: float = 0.05,
        test_ratio: float = 0.1) -> "paddle_geometric.data.Data":
    """Splits the edges of a :class:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges.
    As such, it will replace the :obj:`edge_index` attribute with
    :obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
    :obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
    :obj:`test_pos_edge_index` attributes.
    If :obj:`data` has edge features named :obj:`edge_attr`, then
    :obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
    :obj:`test_pos_edge_attr` will be added as well.

    .. warning::

        :meth:`~torch_geometric.utils.train_test_split_edges` is deprecated and
        will be removed in a future release.
        Use :class:`torch_geometric.transforms.RandomLinkSplit` instead.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation edges.
            (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test edges.
            (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    assert "batch" not in data
    assert data.num_nodes is not None
    assert data.edge_index is not None
    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    del data.edge_index
    del data.edge_attr
    mask = row < col
    row, col = row[mask], col[mask]
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    n_v = int(math.floor(val_ratio * row.shape[0]))
    n_t = int(math.floor(test_ratio * row.shape[0]))
    perm = paddle.randperm(n=row.shape[0])
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = paddle.stack(x=[r, c], axis=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = paddle.stack(x=[r, c], axis=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = paddle.stack(x=[r, c], axis=0)
    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    neg_adj_mask = paddle.ones(shape=[num_nodes, num_nodes], dtype="int32")
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to("bool")
    neg_adj_mask[row, col] = 0
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = paddle.randperm(n=neg_row.shape[0])[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]
    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask
    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = paddle.stack(x=[row, col], axis=0)
    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = paddle.stack(x=[row, col], axis=0)
    return data
