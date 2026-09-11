
import numpy as np

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
KB_TO_BYTES = 1024  # Kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion


class Tensor:

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype
        self.ndim = self.data.ndim
        
    def __repr__ (self):
        return f"Tensor(data = {self.data}, shape={self.shape})"

    def __str__(self):
        return f"Tensor({self.data})"

    def numpy(self):
        return self.data

    def contiguous(self):
        """Return a contiguous copy of the tensor."""
        return Tensor(self.data.copy())

    def memory_footprint(self):
        return self.data.nbytes

    def numel(self):
        """Total number of elements (PyTorch-style alias for size)."""
        return self.size

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __radd__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __rsub__(self, other):
        """Enable subtraction when Tensor is on the right side."""
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        else:
            return Tensor(other - self.data)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data*other.data)
        else:
            return Tensor(self.data*other)
            
    def __rmul__(self, other):
        """Enable multiplication when Tensor is on the right side."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide two tensor element-wise"""
        if isinstance(other, Tensor):
            return Tensor(self.data/other.data)
        else:
            return Tensor(self.data/other)

    def __rtruediv__(self, other):
        """Enable division when Tensor is on the right side."""
        if isinstance(other, Tensor):
            return Tensor(other.data / self.data)
        else:
            return Tensor(other / self.data)

    def _validate_matmul_shapes(self, other):
        """Validate tensor shapes for matrix multiplication."""

        if not isinstance(other, Tensor):
            raise TypeError(
                f"Matrix multiplication requires Tensor, got {type(other).__name__}"
            )

        if len(self.shape) == 0 or len(other.shape) == 0:
            raise ValueError(
                "Both arguments to matmul need to be at least 1D, but one is 0D"
            )

        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Inner dimensions don't match: "
                    f"{self.shape[-1]} vs {other.shape[-2]}"
                )

    def matmul(self, other):
        """Matrix multiplication of two tensors."""
        
        self._validate_matmul_shapes(other)

        a = self.data
        b = other.data

        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape

            result_data = np.zeros((M, N), dtype=np.float32)

            for i in range(M):
                for j in range(N):
                    for k in range(K):
                        result_data[i, j] += a[i, k] * b[k, j]
        else:
            result_data = np.matmul(a, b)

        return Tensor(result_data)


    def __matmul__(self, other):
        """Enable @ operator for matrix multiplication"""
        return self.matmul(other)

    def __getitem__(self, key):
        """Enable indexing and slicing of the tensor"""
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)

    def reshape(self, *shape):
        """Reshape the tensor to the specified shape."""

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape

        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError(
                    "Can only specify one unknown dimension with -1"
                )

            known_size = 1
            unknown_idx = new_shape.index(-1)

            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim

            unknown_dim = self.size // known_size

            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Element count mismatch: "
                f"{self.size} elements vs {target_size} elements"
            )

        reshaped_data = np.reshape(self.data, new_shape)

        return Tensor(reshaped_data)

    def transpose(self, dim0 = None, dim1 = None):
        """Transpose tensor dimension"""
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)

    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis"""
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def mean(self, axis=None, keepdims=False):
        """Compute the mean of tensor along specified axis"""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def max(self, axis=None, keepdims=False):
        """Compute the maximum of tensor along specified axis"""
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)