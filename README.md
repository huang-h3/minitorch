# My implementation of minitorch, for reference

My implementation for [minitorch](https://github.com/minitorch/minitorch), an excellent project for understanding how PyTorch works, as well as CUDA kernels.

The official guidance of this course can be found [here](https://minitorch.github.io/).

## Progress
- [x] Scalar autograd
- [x] Tensor broadcasting
- [x] Backpropagation
- [x] Neural network modules
- [x] CUDA convolution kernels
- [ ] Successfully run `project/run_sentiment.py` and `project/run_mnist_multiclass.py`

## Usage

Activate the local environment first:

```bash
conda activate minitorch
```

Run the regular test suites from the repository root:

```bash
pytest tests/test_operators.py
pytest tests/test_module.py
pytest tests/test_scalar.py tests/test_scalar_autograd.py tests/test_autodiff.py
pytest tests/test_tensor_data.py tests/test_tensor.py
pytest tests/test_tensor_general.py -m task3_1
pytest tests/test_tensor_general.py -m task3_2
pytest tests/test_nn.py -m task4_4
pytest tests/test_conv.py
```

CUDA tests also run from the repository root. The `minitorch` conda environment
is configured to set the required WSL CUDA paths automatically on activation, so
after `conda activate minitorch` this should be enough:

```bash
pytest tests/test_tensor_general.py -m task3_3
pytest tests/test_tensor_general.py -m task3_4
pytest tests/test_conv_cuda.py -m task4_4b
```

CUDA sanity check:

```bash
python -c "import numba, llvmlite; from numba import cuda; print(numba.__version__, llvmlite.__version__); print(cuda.is_available()); cuda.detect()"
```

Use `-m taskX_Y` for MiniTorch task markers. `-k` filters by test name, so
`pytest tests/test_tensor_general.py -k task3_3` can select zero tests.

## Configuration

The project requirements files are the upstream course requirements:

```text
requirements.txt:
  colorama==0.4.3
  hypothesis==6.54
  mypy==0.971
  numba==0.56
  numpy==1.22
  pre-commit==2.20.0
  pytest==7.1.2
  pytest-env
  pytest-runner==5.2
  typing_extensions

requirements.extra.txt:
  datasets==2.4.0
  embeddings==0.0.8
  networkx==2.4
  plotly==4.14.3
  pydot==1.4.1
  python-mnist
  streamlit==1.12.0
  streamlit-ace
  torch
  watchdog==1.0.2
```

For this WSL2 machine, the active `minitorch` environment intentionally differs
from the course pin `numba==0.56` so CUDA kernels compile with the installed
driver/toolkit stack.

Current CUDA/runtime snapshot:

```text
OS: WSL2 Linux, Ubuntu 24.04
GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
NVIDIA driver: 572.94
nvidia-smi CUDA version: 12.8
Numba CUDA detect: 1 supported device, compute capability 12.0
PyTorch: 2.8.0+cu128
```

Relevant conda packages currently installed:

```text
python                  3.10.20
numba                   0.61.2
llvmlite                0.44.0
numba-cuda              0.18.1
cuda-version            12.8
cuda-bindings           12.8.0
cuda-core               0.3.2
cuda-nvcc               12.8.93
cuda-nvrtc              12.8.93
cuda-nvvm               12.8.93
libnvjitlink            12.8.93
pytest                  7.1.2
hypothesis              6.54.0
```

Relevant pip packages currently visible:

```text
minitorch               0.4 editable at this repository
numba                   0.61.2
llvmlite                0.44.0
numba-cuda              0.18.1
cuda-bindings           12.8.0
cuda-core               0.3.2
torch                   2.8.0
pytest                  7.1.2
hypothesis              6.54.0
```

Note: package metadata previously reported `numpy==1.22.4`, but the active
Python import currently resolves to NumPy `2.2.6` in the same environment. Check
the runtime value with:

```bash
python -c "import numpy; print(numpy.__version__, numpy.__file__)"
```

CUDA/WSL configuration is handled by conda activation scripts:

```text
$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
```

They set:

```bash
export NUMBA_CUDA_DRIVER=/usr/lib/wsl/lib/libcuda.so.1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/nvvm/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
```

This was needed because under WSL, Numba may otherwise pick the wrong
`libcuda.so` and report `cuda.is_available() == False`, even when `nvidia-smi`
works.

CUDA debugging notes from this setup:

- If `pytest -m task3_3` selects zero tests, first check
  `python -c "from numba import cuda; print(cuda.is_available())"`. CUDA tests
  are only registered when Numba sees a CUDA device.
- On WSL2, set `NUMBA_CUDA_DRIVER=/usr/lib/wsl/lib/libcuda.so.1` so Numba uses
  the WSL driver shim.
- `numba==0.56` with system CUDA 12.8 failed here with NVVM IR version mismatch.
- `numba-cuda==0.30.x` pulled CUDA 12.9 bindings and failed against this CUDA
  12.8 driver stack with `nvJitLink` errors.
- The working stack here is `numba==0.61.2`, `llvmlite==0.44.0`,
  `numba-cuda==0.18.1`, and conda-forge CUDA 12.8 packages.
- Small MiniTorch CUDA tests can emit occupancy and host-copy warnings. These
  warnings are expected for tiny test tensors; assertion failures are what
  matter.

To inspect the full package state:

```bash
conda list -n minitorch
conda run -n minitorch python -m pip list
```

## Current Test Status

Last checked from the repository root with CUDA visible to Numba:

```bash
pytest tests -q
```

Result:

```text
288 passed, 10 failed, 4 xfailed
```

The failures were:

```text
tests/test_scalar.py::test_one_args[fn4]
tests/test_scalar.py::test_one_args[fn5]
tests/test_scalar.py::test_one_args[fn13]
tests/test_scalar.py::test_one_derivative[fn4]
tests/test_scalar.py::test_one_derivative[fn5]
tests/test_scalar.py::test_one_derivative[fn13]
tests/test_tensor_general.py::test_two_grad_broadcast[fast-fn0]
tests/test_tensor_general.py::test_two_grad_broadcast[fast-fn4]
tests/test_tensor_general.py::test_two_grad_broadcast[fast-fn5]
tests/test_tensor_general.py::test_two_grad_broadcast[cuda-fn5]
```

The scalar failures come from `ScalarFunction.apply` passing Python integer
constants such as `5` and `200` directly into `Neg.forward`. `Neg.forward`
therefore returns an `int`, but `ScalarFunction.apply` asserts that every
forward result is a `float`. The local fix is to convert non-`Scalar` inputs to
a `Scalar` first and append `scalar.data` to `raw_vals`, instead of appending the
raw Python value.

The `test_two_grad_broadcast` failures are Hypothesis health-check failures
(`data_too_large`), not numerical assertion failures. To make the local test
suite ignore that health check, add `HealthCheck.data_too_large` to the loaded
Hypothesis profile in `tests/strategies.py` and `tests/tensor_strategies.py`.

`tests/test_conv_cuda.py` uses the marker `task4_4b`; adding `task4_4b` to the
`[tool:pytest] markers` list in `setup.cfg` removes the unknown-marker warning.

## Project Scripts

`project/run_mnist_multiclass.py` currently fails before training starts:

```text
FileNotFoundError: project/data/train-labels-idx1-ubyte
```

The script uses `python-mnist`, which expects local uncompressed MNIST files
under `project/data/`. Create that directory and place the IDX files there:

```text
project/data/train-images-idx3-ubyte
project/data/train-labels-idx1-ubyte
project/data/t10k-images-idx3-ubyte
project/data/t10k-labels-idx1-ubyte
```

Run the script from the repository root so the relative path `project/data/`
resolves correctly:

```bash
python project/run_mnist_multiclass.py
```

`project/run_sentiment.py` currently fails during import:

```text
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
```

The active environment has a broken NumPy install state: package metadata says
`numpy==1.22.4`, but Python imports NumPy `2.2.6`. `pandas==2.0.3` was built for
a different NumPy ABI, so importing `datasets` fails when it imports pandas.
`pip check` also reports:

```text
datasets 2.4.0 requires huggingface-hub<1.0.0,>=0.1.0, but huggingface-hub 1.21.0 is installed.
numba 0.61.2 requires numpy<2.3,>=1.24, but package metadata reports numpy 1.22.4.
```

Fix the environment by reinstalling a consistent NumPy/Pandas/Datasets stack.
For this CUDA/Numba setup, keep NumPy in Numba's supported range:

```bash
python -m pip uninstall -y numpy pandas datasets huggingface-hub pyarrow
python -m pip install --no-cache-dir --force-reinstall "numpy>=1.24,<2.3" "pandas>=2.2,<3" "datasets>=2.18,<3" "huggingface-hub<1.0" "pyarrow>=12"
python -m pip check
```

Then verify imports before running the sentiment trainer:

```bash
python -c "import numpy, pandas, datasets; print(numpy.__version__, pandas.__version__, datasets.__version__)"
python project/run_sentiment.py
```

`run_sentiment.py` downloads GLUE/SST-2 and GloVe embeddings on first run, so it
needs network access and can take a while before training begins.

<!-- # Recommend to read

## In Chinese

### Author @别再晚睡 from 知乎(zhihu)
- minitorch Efficiency (https://zhuanlan.zhihu.com/p/6359756397) -->
