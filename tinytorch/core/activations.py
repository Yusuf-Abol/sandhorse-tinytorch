import numpy as np
rng = np.random.default_rng(7)
from typing import Optional

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor

# Constants for numerical comparisons
TOLERANCE = 1e-10  # Small tolerance for floating-point comparisons in tests

# Export only activation classes
__all__ = ['Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax']


class Sigmoid:
    def parameters(self):
        return []
    def forward(self, x: Tensor) -> Tensor:
        """Apply the sigmoid function elementwise to the input tensor."""
        x_data = x.data
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.where(
                x_data >= 0,
                1.0 / (1.0 + np.exp(-x_data)),
                np.exp(x_data) / (1.0 + np.exp(x_data))
            )
        return Tensor(result)


class ReLU:
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []
    def forward(self, x: Tensor) -> Tensor:
        """Apply the ReLU function elementwise to the input tensor."""
        result = np.maximum(0, x.data)
        return Tensor(result)
    def __call__(self, x: Tensor) -> Tensor:
        """Allow the ReLU instance to be called like a function."""
        return self.forward(x)


class Tanh:
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []
    def forward(self, x: Tensor) -> Tensor:
        result = np.tanh(x.data)
        return Tensor(result)
    def __call__(self, x: Tensor) -> Tensor:
        """Allow the Tanh instance to be called like a function."""
        return self.forward(x)


class GELU:
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []
    def forward(self, x: Tensor) -> Tensor:
        """Apply the GELU function elementwise to the input tensor."""
        return sigmoid.forward(x*1.702)*x
    def __call__(self, x: Tensor) -> Tensor:
        """Allow the GELU instance to be called like a function."""
        return self.forward(x)


class Softmax:
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []
    def forward(self, x: Tensor, dim=-1) -> Tensor:
        """Apply the softmax function to the input tensor along the specified axis."""
        # Subtract max for numerical stability
        x_max = np.max(x.data, axis=dim, keepdims=True)       
        x_shifted = x.data - x_max

        # compute exponentials  
        exp_values = np.exp(x_shifted)

        # suum along dimension
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True)

        # Normalize to get probabilities
        result = exp_values / exp_sum
        return Tensor(result)
    
    def __call__(self, x: Tensor, dim=-1) -> Tensor:
        """Allow the Softmax instance to be called like a function."""
        return self.forward(x, dim=dim)