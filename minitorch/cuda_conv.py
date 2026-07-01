from __future__ import annotations

from typing import Tuple

from numba import cuda

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import Shape, Storage, Strides, to_index, index_to_position
from .tensor_functions import Function


THREADS_PER_BLOCK = 32

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)

def _blocks(size: int) -> int:
    return (size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    CUDA 1D convolution kernel.

    Shapes:
      input:  batch x in_channels x width
      weight: out_channels x in_channels x k_width
      out:    batch x out_channels x width

    `reverse` decides whether the kernel is anchored left (`False`) or right
    (`True`), matching the CPU version in `fast_conv.py`.
    """

    # batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # a_index = cuda.local.array(MAX_DIMS, numba.int32)
    # b_index = cuda.local.array(MAX_DIMS, numba.int32)
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape
    w = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    oc = cuda.blockIdx.y
    b = cuda.blockIdx.z
    if b < batch and oc < out_channels and w < out_width:
        acc = 0.0

        for ic in range(in_channels):
            for k in range(kw):
                in_w = w + k if not reverse else w - k  # reverse=False
                if 0 <= in_w < width:
                    in_pos = b * input_strides[0] + ic * input_strides[1] + in_w * input_strides[2]
                    wt_pos = oc * weight_strides[0] + ic * weight_strides[1] + k * weight_strides[2]
                    acc += input[in_pos] * weight[wt_pos]

        out_pos = b * out_strides[0] + oc * out_strides[1] + w * out_strides[2]
        out[out_pos] = acc


tensor_conv1d = cuda.jit(_tensor_conv1d)


def _launch_conv1d(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    reverse: bool,
) -> None:
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (
        (out.shape[2] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
        out.shape[1],
        out.shape[0],
    )
    threadsperblock = THREADS_PER_BLOCK
    tensor_conv1d[blockspergrid, threadsperblock](  # type: ignore
        *out.tuple(), out.size, *input.tuple(), *weight.tuple(), reverse
    )


class Conv1dFunCuda(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D convolution.

        Args:
            ctx: Autograd context.
            input: batch x in_channels x width.
            weight: out_channels x in_channels x k_width.

        Returns:
            batch x out_channels x width.
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, width = input.shape
        out_channels, in_channels2, _ = weight.shape
        assert in_channels == in_channels2

        output = input.zeros((batch, out_channels, width))
        _launch_conv1d(output, input, weight, False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, width = input.shape
        out_channels, _, k_width = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, k_width))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        _launch_conv1d(grad_weight, new_input, new_grad_output, False)
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, width))
        new_weight = weight.permute(1, 0, 2)
        _launch_conv1d(grad_input, grad_output, new_weight, True)
        return grad_input, grad_weight


conv1d = Conv1dFunCuda.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    CUDA 2D convolution kernel.

    Shapes:
      input:  batch x in_channels x height x width
      weight: out_channels x in_channels x k_height x k_width
      out:    batch x out_channels x height x width

    `reverse` decides whether the kernel is anchored top-left (`False`) or
    bottom-right (`True`), matching the CPU version in `fast_conv.py`.
    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    # [b, oc, w, h]
    w = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    h = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    batch_channel = cuda.blockIdx.z # batch * out_channels
    b = batch_channel // out_channels
    oc = batch_channel % out_channels

    if b < batch_ and oc < out_channels and h < out_height and w < out_width:
        out_pos = (
            b * out_strides[0]
            + oc * out_strides[1]
            + h * out_strides[2]
            + w * out_strides[3]
        )

        acc = 0.0
        for ic in range(in_channels):
            for k1 in range(kh):
                for k2 in range(kw):
                    in_h = h + k1 if not reverse else h - k1
                    in_w = w + k2 if not reverse else w - k2
                    if 0 <= in_h < height and 0 <= in_w < width:
                        in_pos = (
                            b * input_strides[0]
                            + ic * input_strides[1]
                            + in_h * input_strides[2]
                            + in_w * input_strides[3]
                        )
                        wt_pos = (
                            oc * weight_strides[0]
                            + ic * weight_strides[1]
                            + k1 * weight_strides[2]
                            + k2 * weight_strides[3]
                        )
                        acc += input[in_pos] * weight[wt_pos]

        out[out_pos] = acc


tensor_conv2d = cuda.jit(_tensor_conv2d)


def _launch_conv2d(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    reverse: bool,
) -> None:
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = (
        (out.shape[3] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
        (out.shape[2] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
        out.shape[0] * out.shape[1],
    )
    tensor_conv2d[blockspergrid, threadsperblock](  # type: ignore
        *out.tuple(), out.size, *input.tuple(), *weight.tuple(), reverse
    )


class Conv2dFunCuda(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D convolution.

        Args:
            ctx: Autograd context.
            input: batch x in_channels x height x width.
            weight: out_channels x in_channels x k_height x k_width.

        Returns:
            batch x out_channels x height x width.
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, height, width = input.shape
        out_channels, in_channels2, _, _ = weight.shape
        assert in_channels == in_channels2

        output = input.zeros((batch, out_channels, height, width))
        _launch_conv2d(output, input, weight, False)
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, height, width = input.shape
        out_channels, _, k_height, k_width = weight.shape

        grad_weight = grad_output.zeros(
            (in_channels, out_channels, k_height, k_width)
        )
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        _launch_conv2d(grad_weight, new_input, new_grad_output, False)
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, height, width))
        new_weight = weight.permute(1, 0, 2, 3)
        _launch_conv2d(grad_input, grad_output, new_weight, True)
        return grad_input, grad_weight


conv2d = Conv2dFunCuda.apply
