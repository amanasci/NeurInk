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