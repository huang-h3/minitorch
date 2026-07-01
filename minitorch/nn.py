from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw

    x = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    x = x.permute(0, 1, 2, 4, 3, 5)
    x = x.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    return x, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    x, new_height, new_width = tile(input, kernel)
    x = x.mean(4)
    x = x.contiguous().view(batch, channel, new_height, new_width)
    return x



max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        dim_int = int(dim.item())
        ctx.save_for_backward(input, dim)
        return max_reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        dim_int = int(dim.item())
        # argmax mask: 1 where input == max, 0 elsewhere
        return grad_output * (max_reduce(input, dim_int) == input), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    exp_x = input.exp()
    sum_exp = exp_x.sum(dim)
    return exp_x / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    m = max_reduce(input, dim)
    shifted = input - m
    z = shifted - shifted.exp().sum(dim).log()
    return z


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    x, new_height, new_width = tile(input, kernel)
    x = max(x,4)
    x = x.contiguous().view(batch, channel, new_height, new_width)
    return x



def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if ignore:
        return input

    mask = rand(input.shape, backend=input.backend) > rate
    return input * mask
