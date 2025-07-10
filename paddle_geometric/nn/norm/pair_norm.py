from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer
from paddle_geometric.typing import OptTensor
from paddle_geometric.utils import scatter

# @finshed
class PairNorm(Layer):
    r"""Applies pair normalization over node features as described in the
    `"PairNorm: Tackling Oversmoothing in GNNs"
    <https://arxiv.org/abs/1909.12223>`_ paper.

    .. math::
        \mathbf{x}_i^c &= \mathbf{x}_i - \frac{1}{n}
        \sum_{i=1}^n \mathbf{x}_i \\

        \mathbf{x}_i^{\prime} &= s \cdot
        \frac{\mathbf{x}_i^c}{\sqrt{\frac{1}{n} \sum_{i=1}^n
        {\| \mathbf{x}_i^c \|}^2_2}}

    Args:
        scale (float, optional): Scaling factor :math:`s` of normalization.
            (default, :obj:`1.`)
        scale_individually (bool, optional): If set to :obj:`True`, will
            compute the scaling step as :math:`\mathbf{x}^{\prime}_i = s \cdot
            \frac{\mathbf{x}_i^c}{{\| \mathbf{x}_i^c \|}_2}`.
            (default: :obj:`False`)
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, scale: float = 1., scale_individually: bool = False,
                 eps: float = 1e-5):
        super().__init__()

        self.scale = scale
        self.scale_individually = scale_individually
        self.eps = eps

    def forward(self, x: Tensor, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (paddle.Tensor): The source tensor.
            batch (paddle.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        scale = self.scale
        if batch is None:
            x = x - x.mean(axis=0, keepdim=True)
            if not self.scale_individually:
                return scale * x / (self.eps + x.pow(y=2).sum(axis=-1).mean()).sqrt()
            else:
                return scale * x / (self.eps + x.norm(p=2, axis=-1, keepdim=True))
        else:
            mean = scatter(x, batch, dim=0, dim_size=batch_size, reduce="mean")
            x = x - mean.index_select(axis=0, index=batch)
            if not self.scale_individually:
                return (
                    scale
                    * x
                    / paddle.sqrt(
                        x=self.eps
                        + scatter(
                            x.pow(y=2).sum(axis=-1, keepdim=True),
                            batch,
                            dim=0,
                            dim_size=batch_size,
                            reduce="mean",
                        ).index_select(0, batch)
                    )
                )
            else:
                return scale * x / (self.eps + x.norm(p=2, axis=-1, keepdim=True))

    def __repr__(self):
        return f'{self.__class__.__name__}()'
