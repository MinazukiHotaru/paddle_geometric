import warnings
from typing import Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor

from paddle_geometric import place2devicestr


def is_integer_dtype(dtype):
    """Checks if the given dtype is an integer type.

    Args:
        dtype (paddle.dtype): The data type to check.

    Returns:
        bool: True if the dtype is an integer type, False otherwise.
    """
    return dtype in [paddle.int8, paddle.int16, paddle.int32, paddle.int64]


def map_index(
    src: Tensor,
    index: Tensor,
    max_index: Optional[Union[int, Tensor]] = None,
    inclusive: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Maps indices in `src` to the positional value of their
    corresponding occurrence in `index`.
    Indices must be strictly positive.

    Args:
        src (paddle.Tensor): The source tensor to map.
        index (paddle.Tensor): The index tensor that denotes the new mapping.
        max_index (int, optional): The maximum index value.
            (default :obj:`None`)
        inclusive (bool, optional): If set to True, it is assumed that
            every entry in `src` has a valid entry in `index`.
            Can speed-up computation. (default: `False`)

    Returns:
        Tuple[paddle.Tensor, Optional[paddle.Tensor]]

    Example:
        >>> src = paddle.to_tensor([2, 0, 1, 0, 3], dtype='int64')
        >>> index = paddle.to_tensor([3, 2, 0, 1], dtype='int64')
        >>> map_index(src, index)
        (Tensor([1, 2, 3, 2, 0], dtype=int64), Tensor([True, True, True, True, True]))

        >>> src = paddle.to_tensor([2, 0, 1, 0, 3], dtype='int64')
        >>> index = paddle.to_tensor([3, 2, 0], dtype='int64')
        >>> map_index(src, index)
        (Tensor([1, 2, -1, 2, 0], dtype=int64), Tensor([True, True, False, True, True]))
    """
    if src.is_floating_point():
        raise ValueError(f"Expected 'src' to be an index (got '{src.dtype}')")
    if index.is_floating_point():
        raise ValueError(
            f"Expected 'index' to be an index (got '{index.dtype}')")
    if src.place != index.place:
        raise ValueError(
            f"Both 'src' and 'index' must be on the same device (got '{src.place}' and '{index.place}')"
        )

    if max_index is None:
        max_index = paddle.maximum(x=src.max(), y=index.max())
    THRESHOLD = 40000000 if src.place.is_gpu_place() else 10000000

    if max_index <= THRESHOLD:
        if inclusive:
            assoc = paddle.empty(shape=(max_index + 1, ), dtype=src.dtype,
                                 device=src.place)
        else:
            assoc = paddle.full(shape=(max_index + 1, ), fill_value=-1,
                                dtype=src.dtype, device=src.place)
        assoc[index] = paddle.arange(dtype=src.dtype, end=index.size,
                                     device=src.place)
        out = assoc[src]
        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    WITH_CUDF = False
    if src.place.is_gpu_place():
        try:
            import cudf

            WITH_CUDF = True
        except ImportError:
            import pandas as pd

            warnings.warn(
                "Using CPU-based processing within 'map_index' which "
                "may cause slowdowns and device synchronization. Consider "
                "installing 'cudf' to accelerate computation")
    else:
        import pandas as pd
    if not WITH_CUDF:
        left_ser = pd.Series(src.cpu().numpy(), name="left_ser")
        right_ser = pd.Series(
            index=index.cpu().numpy(),
            data=pd.RangeIndex(0, index.shape[0]),
            name="right_ser",
        )
        result = pd.merge(left_ser, right_ser, how="left", left_on="left_ser",
                          right_index=True)
        out_numpy = result["right_ser"].values
        if place2devicestr(index.place) == "mps" and issubclass(
                out_numpy.dtype.type, np.floating):
            out_numpy = out_numpy.astype(np.float32)
        out = paddle.to_tensor(data=out_numpy).to(index.place)
        if out.is_floating_point() and inclusive:
            raise ValueError(
                "Found invalid entries in 'src' that do not have a "
                "corresponding entry in 'index'. Set `inclusive=False` to "
                "ignore these entries.")
        if out.is_floating_point():
            _x_dtype_ = paddle.isnan(x=out).dtype
            mask = paddle.isnan(x=out).logical_not_().cast_(_x_dtype_)
            out = out[mask].to(index.dtype)
            return out, mask
        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask
    else:
        left_ser = cudf.Series(src, name="left_ser")
        right_ser = cudf.Series(index=index,
                                data=cudf.RangeIndex(0, index.shape[0]),
                                name="right_ser")
        result = cudf.merge(
            left_ser,
            right_ser,
            how="left",
            left_on="left_ser",
            right_index=True,
            sort=True,
        )
        if inclusive:
            try:
                out = paddle.utils.dlpack.from_dlpack(
                    dlpack=result["right_ser"].to_dlpack())
            except ValueError as err:
                raise ValueError(
                    "Found invalid entries in 'src' that do not have a "
                    "corresponding entry in 'index'. Set `inclusive=False` to "
                    "ignore these entries.") from err

        else:
            out = paddle.utils.dlpack.from_dlpack(
                dlpack=result["right_ser"].fillna(-1).to_dlpack())
        out = out[src.argsort().argsort()]
        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask
