"""
Layer definitions for neural network components.

This module defines the base Layer class and specific layer types
for building neural network diagrams.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union


class Layer(ABC):
    """Base class for all neural network layers."""
    
    def __init__(self, layer_type: str, **kwargs):
        """
        Initialize a layer.
        
        Args:
            layer_type: Type identifier for the layer
            **kwargs: Layer-specific parameters
        """
        self.layer_type = layer_type
        self.params = kwargs
        self.name = kwargs.get('name', f"{layer_type}_{id(self)}")
        self.display_name = kwargs.get('display_name', layer_type)  # For LaTeX support
        
    @abstractmethod
    def get_shape_info(self) -> Dict[str, Any]:
        """Get shape and dimension information for rendering."""
        pass
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


class InputLayer(Layer):
    """Input layer for neural networks."""
    
    def __init__(self, shape: Union[Tuple[int, ...], int], **kwargs):
        """
        Initialize an input layer.
        
        Args:
            shape: Input shape (height, width) or (channels, height, width) or single int
            **kwargs: Additional parameters
        """
        super().__init__("input", shape=shape, **kwargs)
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get input shape information."""
        return {
            "shape": self.shape,
            "type": "input",
            "display_text": f"Input {self.shape}"
        }


class ConvLayer(Layer):
    """Convolutional layer."""
    
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = 1, activation: str = "relu", **kwargs):
        """
        Initialize a convolutional layer.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            activation: Activation function name
            **kwargs: Additional parameters
        """
        super().__init__("conv", filters=filters, kernel_size=kernel_size,
                         stride=stride, activation=activation, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.activation = activation
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get convolution layer information."""
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "activation": self.activation,
            "type": "conv",
            "display_text": f"Conv {self.filters}@{self.kernel_size[0]}x{self.kernel_size[1]}"
        }


class DenseLayer(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, units: int, activation: str = "relu", **kwargs):
        """
        Initialize a dense layer.
        
        Args:
            units: Number of output units
            activation: Activation function name
            **kwargs: Additional parameters
        """
        super().__init__("dense", units=units, activation=activation, **kwargs)
        self.units = units
        self.activation = activation
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get dense layer information."""
        return {
            "units": self.units,
            "activation": self.activation,
            "type": "dense",
            "display_text": f"Dense {self.units}"
        }


class FlattenLayer(Layer):
    """Flatten layer for converting multi-dimensional input to 1D."""
    
    def __init__(self, **kwargs):
        """Initialize a flatten layer."""
        super().__init__("flatten", **kwargs)
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get flatten layer information."""
        return {
            "type": "flatten",
            "display_text": "Flatten"
        }


class DropoutLayer(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, rate: float, **kwargs):
        """
        Initialize a dropout layer.
        
        Args:
            rate: Dropout rate (0.0 to 1.0)
            **kwargs: Additional parameters
        """
        super().__init__("dropout", rate=rate, **kwargs)
        self.rate = rate
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get dropout layer information."""
        return {
            "rate": self.rate,
            "type": "dropout",
            "display_text": f"Dropout {self.rate}"
        }


class OutputLayer(Layer):
    """Output layer for neural networks."""
    
    def __init__(self, units: int, activation: str = "softmax", **kwargs):
        """
        Initialize an output layer.
        
        Args:
            units: Number of output units
            activation: Activation function name
            **kwargs: Additional parameters
        """
        super().__init__("output", units=units, activation=activation, **kwargs)
        self.units = units
        self.activation = activation
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get output layer information."""
        return {
            "units": self.units,
            "activation": self.activation,
            "type": "output",
            "display_text": f"Output {self.units}"
        }


class MaxPoolLayer(Layer):
    """Max pooling layer."""
    
    def __init__(self, pool_size: Union[int, Tuple[int, int]] = 2, 
                 stride: Optional[Union[int, Tuple[int, int]]] = None, **kwargs):
        """
        Initialize a max pooling layer.
        
        Args:
            pool_size: Size of pooling window
            stride: Stride of pooling operation
            **kwargs: Additional parameters
        """
        super().__init__("maxpool", pool_size=pool_size, stride=stride, **kwargs)
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        if not isinstance(self.stride, tuple):
            self.stride = (self.stride, self.stride)
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get max pooling layer information."""
        return {
            "pool_size": self.pool_size,
            "stride": self.stride,
            "type": "maxpool",
            "display_text": f"MaxPool {self.pool_size[0]}x{self.pool_size[1]}"
        }


class UpSampleLayer(Layer):
    """Upsampling layer for increasing spatial resolution."""
    
    def __init__(self, size: Union[int, Tuple[int, int]] = 2, 
                 method: str = "nearest", **kwargs):
        """
        Initialize an upsampling layer.
        
        Args:
            size: Upsampling factor
            method: Upsampling method ('nearest', 'bilinear')
            **kwargs: Additional parameters
        """
        super().__init__("upsample", size=size, method=method, **kwargs)
        self.size = size if isinstance(size, tuple) else (size, size)
        self.method = method
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get upsampling layer information."""
        return {
            "size": self.size,
            "method": self.method,
            "type": "upsample",
            "display_text": f"UpSample {self.size[0]}x{self.size[1]}"
        }


class ConvTransposeLayer(Layer):
    """Transposed convolution (deconvolution) layer."""
    
    def __init__(self, filters: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = 1, activation: str = "relu", **kwargs):
        """
        Initialize a transposed convolution layer.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            activation: Activation function name
            **kwargs: Additional parameters
        """
        super().__init__("conv_transpose", filters=filters, kernel_size=kernel_size,
                         stride=stride, activation=activation, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.activation = activation
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get transposed convolution layer information."""
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "activation": self.activation,
            "type": "conv_transpose",
            "display_text": f"ConvTranspose {self.filters}@{self.kernel_size[0]}x{self.kernel_size[1]}"
        }


class BatchNormLayer(Layer):
    """Batch normalization layer."""
    
    def __init__(self, **kwargs):
        """Initialize a batch normalization layer."""
        super().__init__("batch_norm", **kwargs)
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get batch normalization layer information."""
        return {
            "type": "batch_norm",
            "display_text": "BatchNorm"
        }


class LayerNormLayer(Layer):
    """Layer normalization layer."""
    
    def __init__(self, **kwargs):
        """Initialize a layer normalization layer."""
        super().__init__("layer_norm", **kwargs)
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get layer normalization layer information."""
        return {
            "type": "layer_norm",
            "display_text": "LayerNorm"
        }


class MultiHeadAttentionLayer(Layer):
    """Multi-head attention layer for transformers."""
    
    def __init__(self, num_heads: int, key_dim: int, **kwargs):
        """
        Initialize a multi-head attention layer.
        
        Args:
            num_heads: Number of attention heads
            key_dim: Dimension of attention keys/queries
            **kwargs: Additional parameters
        """
        super().__init__("multi_head_attention", num_heads=num_heads, key_dim=key_dim, **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get multi-head attention layer information."""
        return {
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "type": "multi_head_attention",
            "display_text": f"MultiHeadAttn H:{self.num_heads} D:{self.key_dim}"
        }


class EmbeddingLayer(Layer):
    """Embedding layer for transformers."""
    
    def __init__(self, vocab_size: int, embed_dim: int, **kwargs):
        """
        Initialize an embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            **kwargs: Additional parameters
        """
        super().__init__("embedding", vocab_size=vocab_size, embed_dim=embed_dim, **kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get embedding layer information."""
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "type": "embedding",
            "display_text": f"Embedding {self.vocab_size}→{self.embed_dim}"
        }


class PositionalEncodingLayer(Layer):
    """Positional encoding layer for transformers."""
    
    def __init__(self, max_len: int, embed_dim: int, **kwargs):
        """
        Initialize a positional encoding layer.
        
        Args:
            max_len: Maximum sequence length
            embed_dim: Embedding dimension
            **kwargs: Additional parameters
        """
        super().__init__("positional_encoding", max_len=max_len, embed_dim=embed_dim, **kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get positional encoding layer information."""
        return {
            "max_len": self.max_len,
            "embed_dim": self.embed_dim,
            "type": "positional_encoding",
            "display_text": f"PosEnc L:{self.max_len} D:{self.embed_dim}"
        }


class ReshapeLayer(Layer):
    """Reshape layer for changing tensor dimensions."""
    
    def __init__(self, target_shape: Tuple[int, ...], **kwargs):
        """
        Initialize a reshape layer.
        
        Args:
            target_shape: Target shape
            **kwargs: Additional parameters
        """
        super().__init__("reshape", target_shape=target_shape, **kwargs)
        self.target_shape = target_shape
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get reshape layer information."""
        return {
            "target_shape": self.target_shape,
            "type": "reshape",
            "display_text": f"Reshape {self.target_shape}"
        }


class GlobalAvgPoolLayer(Layer):
    """Global average pooling layer."""
    
    def __init__(self, **kwargs):
        """Initialize a global average pooling layer."""
        super().__init__("global_avg_pool", **kwargs)
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get global average pooling layer information."""
        return {
            "type": "global_avg_pool",
            "display_text": "GlobalAvgPool"
        }


class ConcatenateLayer(Layer):
    """Concatenation layer for merging multiple inputs."""
    
    def __init__(self, axis: int = -1, **kwargs):
        """
        Initialize a concatenate layer.
        
        Args:
            axis: Axis along which to concatenate
            **kwargs: Additional parameters
        """
        super().__init__("concatenate", axis=axis, **kwargs)
        self.axis = axis
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get concatenate layer information."""
        return {
            "axis": self.axis,
            "type": "concatenate",
            "display_text": f"Concat axis={self.axis}"
        }


class AddLayer(Layer):
    """Add layer for element-wise addition (residual connections)."""
    
    def __init__(self, **kwargs):
        """Initialize an add layer."""
        super().__init__("add", **kwargs)
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get add layer information."""
        return {
            "type": "add",
            "display_text": "Add"
        }