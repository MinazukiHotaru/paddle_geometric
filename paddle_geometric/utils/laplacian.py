from typing import Optional, Tuple

import paddle
from paddle import Tensor

from paddle_geometric.typing import OptTensor
from paddle_geometric.utils import add_self_loops, remove_self_loops, scatter
from paddle_geometric.utils.num_nodes import maybe_num_nodes


def get_laplacian(
    edge_index: Tensor,
    edge_weight: OptTensor = None,
    normalization: Optional[str] = None,
    dtype: Optional[paddle.dtype] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Args:
        edge_index (Tensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        dtype (paddle.dtype, optional): The desired data type of returned tensor
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    Examples:
        >>> edge_index = paddle.to_tensor([[0, 1, 1, 2],
        ...                                [1, 0, 2, 1]])
        >>> edge_weight = paddle.to_tensor([1., 2., 2., 4.])

        >>> # No normalization
        >>> lap = get_laplacian(edge_index, edge_weight)

        >>> # Symmetric normalization
        >>> lap_sym = get_laplacian(edge_index, edge_weight,
                                    normalization='sym')

        >>> # Random-walk normalization
        >>> lap_rw = get_laplacian(edge_index, edge_weight, normalization='rw')
    """
    if normalization is not None:
        assert normalization in ["sym", "rw"]
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = paddle.ones(shape=edge_index.shape[1], dtype=dtype)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce="sum")
    if normalization is None:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = paddle.concat(x=[-edge_weight, deg], axis=0)
    elif normalization == "sym":
        deg_inv_sqrt = deg.pow_(y=-0.5)
        deg_inv_sqrt.masked_fill_(mask=deg_inv_sqrt == float("inf"), value=0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        assert isinstance(edge_weight, paddle.Tensor)
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight,
                                                 fill_value=1.0,
                                                 num_nodes=num_nodes)
    else:
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(mask=deg_inv == float("inf"), value=0)
        edge_weight = deg_inv[row] * edge_weight
        assert isinstance(edge_weight, paddle.Tensor)
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight,
                                                 fill_value=1.0,
                                                 num_nodes=num_nodes)
    return edge_index, edge_weight
