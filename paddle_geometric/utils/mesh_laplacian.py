from typing import Optional, Tuple

import paddle

from paddle_geometric.utils import add_self_loops, scatter, to_undirected


def get_mesh_laplacian(
    pos: paddle.Tensor, face: paddle.Tensor,
    normalization: Optional[str] = None
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Computes the mesh Laplacian of a mesh given by :obj:`pos` and
    :obj:`face`.

    Computation is based on the cotangent matrix defined as

    .. math::
        \\mathbf{C}_{ij} = \\begin{cases}
            \\frac{\\cot \\angle_{ikj}~+\\cot \\angle_{ilj}}{2} &
            \\text{if } i, j \\text{ is an edge} \\\\
            -\\sum_{j \\in N(i)}{C_{ij}} &
            \\text{if } i \\text{ is in the diagonal} \\\\
            0 & \\text{otherwise}
      \\end{cases}

    Normalization depends on the mass matrix defined as

    .. math::
        \\mathbf{M}_{ij} = \\begin{cases}
            a(i) & \\text{if } i \\text{ is in the diagonal} \\\\
            0 & \\text{otherwise}
      \\end{cases}

    where :math:`a(i)` is obtained by joining the barycenters of the
    triangles around vertex :math:`i`.

    Args:
        pos (Tensor): The node positions.
        face (LongTensor): The face indices.
        normalization (str, optional): The normalization scheme for the mesh
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\\mathbf{L} = \\mathbf{C}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\\mathbf{L} = \\mathbf{M}^{-1/2} \\mathbf{C}\\mathbf{M}^{-1/2}`

            3. :obj:`"rw"`: Row-wise normalization
            :math:`\\mathbf{L} = \\mathbf{M}^{-1} \\mathbf{C}`
    """
    assert pos.shape[1] == 3 and face.shape[0] == 3
    num_nodes = tuple(pos.shape)[0]

    def get_cots(left: paddle.Tensor, centre: paddle.Tensor,
                 right: paddle.Tensor) -> paddle.Tensor:
        left_pos, central_pos, right_pos = pos[left], pos[centre], pos[right]
        left_vec = left_pos - central_pos
        right_vec = right_pos - central_pos
        dot = paddle.einsum("ij, ij -> i", left_vec, right_vec)
        cross = paddle.linalg.norm(
            x=paddle.cross(x=left_vec, y=right_vec, axis=1), axis=1)
        cot = dot / cross
        return cot / 2.0

    cot_021 = get_cots(face[0], face[2], face[1])
    cot_102 = get_cots(face[1], face[0], face[2])
    cot_012 = get_cots(face[0], face[1], face[2])
    cot_weight = paddle.concat(x=[cot_021, cot_102, cot_012])
    cot_index = paddle.concat(x=[face[:2], face[1:], face[::2]], axis=1)
    cot_index, cot_weight = to_undirected(cot_index, cot_weight)
    cot_deg = scatter(cot_weight, cot_index[0], 0, num_nodes, reduce="sum")
    edge_index, _ = add_self_loops(cot_index, num_nodes=num_nodes)
    edge_weight = paddle.concat(x=[cot_weight, -cot_deg], axis=0)
    if normalization is not None:

        def get_areas(left: paddle.Tensor, centre: paddle.Tensor,
                      right: paddle.Tensor) -> paddle.Tensor:
            central_pos = pos[centre]
            left_vec = pos[left] - central_pos
            right_vec = pos[right] - central_pos
            cross = paddle.linalg.norm(
                x=paddle.cross(x=left_vec, y=right_vec, axis=1), axis=1)
            area = cross / 6.0
            return area / 2.0

        area_021 = get_areas(face[0], face[2], face[1])
        area_102 = get_areas(face[1], face[0], face[2])
        area_012 = get_areas(face[0], face[1], face[2])
        area_weight = paddle.concat(x=[area_021, area_102, area_012])
        area_index = paddle.concat(x=[face[:2], face[1:], face[::2]], axis=1)
        area_index, area_weight = to_undirected(area_index, area_weight)
        area_deg = scatter(area_weight, area_index[0], 0, num_nodes, "sum")
        if normalization == "sym":
            area_deg_inv_sqrt = area_deg.pow_(y=-0.5)
            area_deg_inv_sqrt[area_deg_inv_sqrt == float("inf")] = 0.0
            edge_weight = (area_deg_inv_sqrt[edge_index[0]] * edge_weight *
                           area_deg_inv_sqrt[edge_index[1]])
        elif normalization == "rw":
            area_deg_inv = 1.0 / area_deg
            area_deg_inv[area_deg_inv == float("inf")] = 0.0
            edge_weight = area_deg_inv[edge_index[0]] * edge_weight
    return edge_index, edge_weight
