
import numpy as np

rng = np.random.default_rng(7)

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid

# Constants for weight initialization
# Note: True Xavier/Glorot uses sqrt(2/(fan_in+fan_out)), but we use the simpler
# LeCun-style sqrt(1/fan_in) for pedagogical clarity. Both achieve stable gradients.
INIT_SCALE_FACTOR = 1.0  # LeCun-style initialization: sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0  # He initialization uses sqrt(2/fan_in) for ReLU

# Constants for dropout
DROPOUT_MIN_PROB = 0.0  # Minimum dropout probability (no dropout)
DROPOUT_MAX_PROB = 1.0  # Maximum dropout probability (drop everything)


class Layer:
    """
    Base class for all neural network layers.

    All layers should inherit from this class and implement:
    - forward(x): Compute layer output
    - parameters(): Return list of trainable parameters

    The __call__ method is provided to make layers callable.
    """

    def forward(self, x):
        """
        Forward pass through the layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor after transformation
        """
        raise NotImplementedError(
            f"forward() not implemented in {self.__class__.__name__}\n"
            f"  ❌ The Layer base class requires subclasses to implement forward()\n"
            f"  💡 forward() defines how input data is transformed by this layer\n"
            f"  🔧 Add this method to your class:\n"
            f"     def forward(self, x):\n"
            f"         # Your transformation logic here\n"
            f"         return transformed_x"
        )

    def __call__(self, x, *args, **kwargs):
        """Allow layer to be called like a function."""
        return self.forward(x, *args, **kwargs)

    def parameters(self):
        """
        Return list of trainable parameters.

        Returns:
            List of Tensor objects (weights and biases)
        """
        return []  # Base class has no parameters

    def __repr__(self):
        """String representation of the layer."""
        return f"{self.__class__.__name__}()"



class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        # LeCun-style initialization for stable gradients
        scale = np.sqrt(INIT_SCALE_FACTOR / in_features)
        weight_data = rng.standard_normal((in_features, out_features))*scale
        self.weight = Tensor(weight_data)

        # Initialize bias zeros or None
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        """
        Forward pass through linear later.

        """
        # Linear Transformation: y = xW + b
        output = x.matmul(self.weight)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        return output

    def parameters(self):
        """Return list of trainable parameters (weights and bias)."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def __repr__(self):
        """String representation for debugging."""
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"



class Dropout(Layer):
    def __init__(self, p=0.5):
        """Initialize Dropout layer"""
        if not (DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB):
            raise ValueError(f"Dropout probability p must be in [{DROPOUT_MIN_PROB}, {DROPOUT_MAX_PROB}]")
            f"Invalid dropout probability:{p}\n"
            f" p must between {DROPOUT_MIN_PROB} and {DROPOUT_MAX_PROB}\n"
            f" p is the probability of DROPPING a neuron (not keeping it!)\n"
            f" p = 0.0 means keep all neurons, p = 1.0 means drop all neurons\n"
            f" p = 0.5 means drop 50% of neurons randomly\n"
            f" p = 1.0 means drop all neurons (no output)\n"
            f" common values: Dropout(0.1) for light, Dropout(0.3) for moderate, Dropout(0.5) for aggresive\n"
        self.p = p

    def _should_apply_dropout(self, training):
        """Determine if dropout should be applied based on training mode."""
        return training and self.p > DROPOUT_MIN_PROB

    def _generate_dropout_mask(self, shape):
        keep_prob = 1.0 - self.p
        binary_mask = (rng.random(shape) < keep_prob).astype(np.float32)
        scale = 1.0 / keep_prob
        return Tensor(binary_mask*scale)
    
    def forward(self, x, training=True):
        if not self._should_apply_dropout(training):
            return x  # No dropout applied during evaluation or if p=0 
        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(x.data))  # Drop all neurons
        mask = self._generate_dropout_mask(x.data.shape)
        return x * mask  # Apply dropout mask to input

    def __call__(self, x, training=True):
        """Allows the layer to be called like a function."""
        return self.forward(x, training)

    def parameters(self):
        """Dropout has no parameters."""
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"



class Sequential:
    """
    Container that chains layers together sequentially.

    After you understand explicit layer composition, Sequential provides
    a convenient way to bundle layers together.

    """

    def __init__(self, *layers):
        """Initialize with layers to chain together."""
        # Accept both Sequential(layer1, layer2) and Sequential([layer1, layer2])
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self, x, training=True):
        """Forward pass through all layers sequentially.

        Passes training=True/False to layers that support it (e.g. Dropout),
        and falls back to a plain forward(x) call for layers that don't.
        This lets you switch between training and eval mode with one flag:

            output = model.forward(x, training=False)   # eval: Dropout disabled
            output = model.forward(x, training=True)    # train: Dropout active
        """
        for layer in self.layers:
            try:
                x = layer.forward(x, training=training)
            except TypeError:
                x = layer.forward(x)
        return x

    def __call__(self, x, training=True):
        """Allow model to be called like a function."""
        return self.forward(x, training=training)

    def parameters(self):
        """Collect all parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_reprs = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"