import itertools
import numbers
from typing import Any

import paddle


def repeat(src: Any, length: int) -> Any:
    if src is None:
        return None
    if isinstance(src, paddle.Tensor):
        if src.size == 1:
            return src.tile(repeat_times=length)
        if src.size > length:
            return src[:length]
        if src.size < length:
            last_elem = src[-1].unsqueeze(axis=0)
            padding = last_elem.tile(repeat_times=length - src.size)
            return paddle.concat(x=[src, padding])
        return src
    if isinstance(src, numbers.Number):
        return list(itertools.repeat(src, length))
    if len(src) > length:
        return src[:length]
    if len(src) < length:
        return src + list(itertools.repeat(src[-1], length - len(src)))
    return src
