import pytest
import paddle

from paddle_geometric.nn import BatchNorm, HeteroBatchNorm
from paddle_geometric.testing import is_full_test, withDevice


@withDevice
@pytest.mark.parametrize("conf", [True, False])
def test_batch_norm(device, conf):
    x = paddle.randn(shape=[100, 16])
    norm = BatchNorm(16, affine=conf, track_running_stats=conf).to(device)
    norm.reset_running_stats()
    norm.reset_parameters()
    assert (
        str(norm)
        == "BatchNorm(num_features=16, momentum=0.9, epsilon=1e-05, data_format=NCL)"
    )
    if is_full_test():
        paddle.jit.to_static(function=norm)
    out = norm(x)
    assert tuple(out.shape) == (100, 16)


# def test_batch_norm_single_element():
#     x = paddle.randn(shape=[1, 16])
#     norm = BatchNorm(16)
#     with pytest.raises(ValueError, match="Expected more than 1 value"):
#         norm(x)
#     with pytest.raises(ValueError, match="requires 'track_running_stats'"):
#         norm = BatchNorm(16, track_running_stats=False, allow_single_element=True)
#     norm = BatchNorm(16, track_running_stats=True, allow_single_element=True)
#     out = norm(x)
#     assert paddle.allclose(x=out, y=x).item()


@withDevice
@pytest.mark.parametrize("conf", [True, False])
def test_hetero_batch_norm(device, conf):
    x = paddle.ones(shape=(100, 16))
    norm = BatchNorm(16, affine=conf, track_running_stats=conf).to(device)
    expected = norm(x)
    type_vec = paddle.zeros(shape=[100], dtype="int64")
    norm = HeteroBatchNorm(16, num_types=1, affine=conf, track_running_stats=conf).to(
        device
    )
    norm.reset_running_stats()
    norm.reset_parameters()
    assert str(norm) == "HeteroBatchNorm(16, num_types=1)"
    out = norm(x, type_vec)
    assert tuple(out.shape) == (100, 16)
    
    print(out)
    assert paddle.allclose(x=out, y=expected, atol=0.001).item()


    type_vec = paddle.randint(low=0, high=5, shape=(100,))
    norm = HeteroBatchNorm(16, num_types=5, affine=conf, track_running_stats=conf).to(
        device
    )
    out = norm(x, type_vec)
    assert tuple(out.shape) == (100, 16)
    for i in range(5):
        mean = out[type_vec == i].mean()
        std = out[type_vec == i].std(unbiased=False)
        assert paddle.allclose(x=mean, y=paddle.zeros_like(x=mean), atol=1e-07).item()
        assert paddle.allclose(x=std, y=paddle.ones_like(x=std), atol=1e-07).item()

if __name__ == "__main__":
    test_hetero_batch_norm('cpu', True)