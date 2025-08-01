from typing import Any
import paddle


def is_mps_available() -> bool:
    """
    Returns a bool indicating if Metal Performance Shaders (MPS) is currently available in PaddlePaddle.
    Note: PaddlePaddle does not support MPS directly, so this always returns False.
    """
    # Placeholder for PaddlePaddle as it doesn't support MPS
    return False


def is_xpu_available() -> bool:
    """
    Returns a bool indicating if XPU (Intel Extension for PaddlePaddle) is currently available.
    """
    return False


def device2str(type=None, index=None, *, device=None):
    type = device if device else type
    if isinstance(type, int):
        type = f'gpu:{type}'
    elif isinstance(type, str):
        if 'cuda' in type:
            type = type.replace('cuda', 'gpu')
        if 'cpu' in type:
            type = 'cpu'
        elif index is not None:
            type = f'{type}:{index}'
    elif isinstance(type, paddle.CPUPlace) or (type is None):
        type = 'cpu'
    elif isinstance(type, paddle.CUDAPlace):
        type = f'gpu:{type.get_device_id()}'

    return type

def device(device: Any) -> paddle.device:
    """
    Returns a PaddlePaddle device.

    If 'auto' is specified, returns the optimal device depending on available hardware.
    """
    if device != 'auto':
        return device2str(device)
    if paddle.device.cuda.device_count() >= 1:
        return device2str("cuda")
    if is_mps_available():
        return device2str("mps")
    if is_xpu_available():
        return device2str("xpu")
    return device2str("cpu")


def place2devicestr(place):
    if  place.is_cpu_place():
        device = 'cpu'
    elif place.is_gpu_place():
        device_id = place.gpu_device_id()
        device = 'gpu:' + str(device_id)
    else:
        raise ValueError(f"The device specification {place} is invalid")

    return device
