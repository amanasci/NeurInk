"""
Core Diagram class for building neural network diagrams.

This module provides the main Diagram class that supports method chaining
for building neural network architectures.
"""

from typing import List, Union, Tuple, Optional
from .layer import (
    Layer, InputLayer, ConvLayer, DenseLayer, 
    FlattenLayer, DropoutLayer, OutputLayer,
    AttentionLayer, LayerNormLayer, EmbeddingLayer,
    PoolingLayer, BatchNormLayer
)
from .renderer import SVGRenderer
from .themes import Theme, IEEETheme


class Diagram:
    """
    Main class for building neural network diagrams.
    
    Supports method chaining for easy construction of network architectures.
    Can render to SVG format with customizable themes.
    """
    
    def __init__(self):
        """Initialize an empty diagram."""
        self.layers: List[Layer] = []
        self._renderer = SVGRenderer()
        
    def input(self, shape: Union[Tuple[int, ...], int]) -> 'Diagram':
        """
        Add an input layer to the diagram.
        
        Args:
            shape: Input shape as tuple (height, width) or (channels, height, width) 
                  or single integer
                  
        Returns:
            Self for method chaining
        """
        layer = InputLayer(shape)
        self.layers.append(layer)
        return self
        
    def conv(self, filters: int, kernel_size: Union[int, Tuple[int, int]], 
             stride: int = 1, activation: str = "relu") -> 'Diagram':
        """
        Add a convolutional layer to the diagram.
        
        Args:
            filters: Number of output filters/channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution (default: 1)
            activation: Activation function name (default: "relu")
            
        Returns:
            Self for method chaining
        """
        layer = ConvLayer(filters, kernel_size, stride, activation)
        self.layers.append(layer)
        return self
        
    def dense(self, units: int, activation: str = "relu") -> 'Diagram':
        """
        Add a dense (fully connected) layer to the diagram.
        
        Args:
            units: Number of output units
            activation: Activation function name (default: "relu")
            
        Returns:
            Self for method chaining
        """
        layer = DenseLayer(units, activation)
        self.layers.append(layer)
        return self
        
    def flatten(self) -> 'Diagram':
        """
        Add a flatten layer to the diagram.
        
        Returns:
            Self for method chaining
        """
        layer = FlattenLayer()
        self.layers.append(layer)
        return self
        
    def dropout(self, rate: float) -> 'Diagram':
        """
        Add a dropout layer to the diagram.
        
        Args:
            rate: Dropout rate between 0.0 and 1.0
            
        Returns:
            Self for method chaining
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
            
        layer = DropoutLayer(rate)
        self.layers.append(layer)
        return self
        
    def output(self, units: int, activation: str = "softmax") -> 'Diagram':
        """
        Add an output layer to the diagram.
        
        Args:
            units: Number of output units
            activation: Activation function name (default: "softmax")
            
        Returns:
            Self for method chaining
        """
        layer = OutputLayer(units, activation)
        self.layers.append(layer)
        return self
        
    def attention(self, num_heads: int = 8, key_dim: int = 64) -> 'Diagram':
        """
        Add a multi-head attention layer to the diagram.
        
        Args:
            num_heads: Number of attention heads (default: 8)
            key_dim: Dimension of keys/queries (default: 64)
            
        Returns:
            Self for method chaining
        """
        layer = AttentionLayer(num_heads, key_dim)
        self.layers.append(layer)
        return self
        
    def layer_norm(self) -> 'Diagram':
        """
        Add a layer normalization layer to the diagram.
        
        Returns:
            Self for method chaining
        """
        layer = LayerNormLayer()
        self.layers.append(layer)
        return self
        
    def embedding(self, vocab_size: int, embed_dim: int) -> 'Diagram':
        """
        Add an embedding layer to the diagram.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            
        Returns:
            Self for method chaining
        """
        layer = EmbeddingLayer(vocab_size, embed_dim)
        self.layers.append(layer)
        return self
        
    def pooling(self, pool_type: str = "max", pool_size: int = 2, stride: int = 2) -> 'Diagram':
        """
        Add a pooling layer to the diagram.
        
        Args:
            pool_type: Type of pooling ('max', 'avg', 'global_avg')
            pool_size: Size of pooling window (default: 2)
            stride: Stride of pooling (default: 2)
            
        Returns:
            Self for method chaining
        """
        layer = PoolingLayer(pool_type, pool_size, stride)
        self.layers.append(layer)
        return self
        
    def batch_norm(self) -> 'Diagram':
        """
        Add a batch normalization layer to the diagram.
        
        Returns:
            Self for method chaining
        """
        layer = BatchNormLayer()
        self.layers.append(layer)
        return self
        
    def render(self, filename: str, theme: Union[str, Theme] = "ieee") -> str:
        """
        Render the diagram to SVG format.
        
        Args:
            filename: Output filename for the SVG file
            theme: Theme name ("ieee", "apj", "minimal", "dark") or Theme object
            
        Returns:
            Path to the generated SVG file
        """
        if isinstance(theme, str):
            theme_obj = self._get_theme_by_name(theme)
        else:
            theme_obj = theme
            
        svg_content = self._renderer.render(self.layers, theme_obj)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
            
        return filename
        
    def _get_theme_by_name(self, theme_name: str) -> Theme:
        """Get theme object by name."""
        from .themes import APJTheme, MinimalTheme, DarkTheme, NNSVGTheme
        
        theme_map = {
            "ieee": IEEETheme(),
            "apj": APJTheme(),
            "minimal": MinimalTheme(), 
            "dark": DarkTheme(),
            "nnsvg": NNSVGTheme()
        }
        
        if theme_name not in theme_map:
            raise ValueError(f"Unknown theme '{theme_name}'. "
                           f"Available themes: {list(theme_map.keys())}")
        
        return theme_map[theme_name]
        
    @classmethod
    def from_string(cls, dsl_text: str) -> 'Diagram':
        """
        Create a diagram from DSL text.
        
        Args:
            dsl_text: DSL text defining the network architecture
            
        Returns:
            Diagram object created from the DSL
        """
        from .parser import DSLParser
        
        parser = DSLParser()
        return parser.parse(dsl_text)
        
    def __len__(self) -> int:
        """Return the number of layers in the diagram."""
        return len(self.layers)
        
    def __repr__(self) -> str:
        """String representation of the diagram."""
        layer_types = [layer.layer_type for layer in self.layers]
        return f"Diagram({layer_types})"