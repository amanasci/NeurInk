"""
Core Diagram class for building neural network diagrams.

This module provides the main Diagram class that supports method chaining
for building neural network architectures with graph-based representation.
"""

from typing import List, Union, Tuple, Optional, Dict, Any
import networkx as nx
from .layer import (
    Layer, InputLayer, ConvLayer, DenseLayer, 
    FlattenLayer, DropoutLayer, OutputLayer,
    MaxPoolLayer, UpSampleLayer, ConvTransposeLayer,
    BatchNormLayer, LayerNormLayer, MultiHeadAttentionLayer,
    EmbeddingLayer, PositionalEncodingLayer, ReshapeLayer,
    GlobalAvgPoolLayer, ConcatenateLayer, AddLayer
)
from .renderer import GraphvizRenderer
from .themes import Theme, IEEETheme


class Diagram:
    """
    Main class for building neural network diagrams.
    
    Supports method chaining for easy construction of network architectures.
    Can render to SVG format with customizable themes using graph-based architecture.
    """
    
    def __init__(self):
        """Initialize an empty diagram."""
        self.graph = nx.DiGraph()
        self._renderer = GraphvizRenderer()
        self._layer_counter = 0
        self._last_added_layer = None
        
    def _generate_layer_name(self, layer_type: str) -> str:
        """Generate a unique layer name."""
        self._layer_counter += 1
        return f"{layer_type}_{self._layer_counter}"
        
    def _add_layer_to_graph(self, layer: Layer, name: Optional[str] = None) -> str:
        """Add layer to graph and return its name."""
        if name is None:
            name = self._generate_layer_name(layer.layer_type)
        
        layer.name = name
        self.graph.add_node(name, layer=layer)
        
        # For backward compatibility with sequential networks,
        # auto-connect to the last added layer
        if self._last_added_layer is not None:
            self.graph.add_edge(self._last_added_layer, name)
        
        self._last_added_layer = name
        return name
        
    def input(self, shape: Union[Tuple[int, ...], int], name: Optional[str] = None) -> 'Diagram':
        """
        Add an input layer to the diagram.
        
        Args:
            shape: Input shape as tuple (height, width) or (channels, height, width) 
                  or single integer
            name: Optional name for the layer. If not provided, auto-generated.
                  
        Returns:
            Self for method chaining
        """
        layer = InputLayer(shape, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def conv(self, filters: int, kernel_size: Union[int, Tuple[int, int]], 
             stride: int = 1, activation: str = "relu", name: Optional[str] = None, **kwargs) -> 'Diagram':
        """
        Add a convolutional layer to the diagram.
        
        Args:
            filters: Number of output filters/channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution (default: 1)
            activation: Activation function name (default: "relu")
            name: Optional name for the layer. If not provided, auto-generated.
            **kwargs: Visual annotation parameters (annotation_color, annotation_shape, etc.)
            
        Returns:
            Self for method chaining
        """
        layer = ConvLayer(filters, kernel_size, stride, activation, name=name, **kwargs)
        self._add_layer_to_graph(layer, name)
        return self
        
    def dense(self, units: int, activation: str = "relu", name: Optional[str] = None, **kwargs) -> 'Diagram':
        """
        Add a dense (fully connected) layer to the diagram.
        
        Args:
            units: Number of output units
            activation: Activation function name (default: "relu")
            name: Optional name for the layer. If not provided, auto-generated.
            **kwargs: Visual annotation parameters (annotation_color, annotation_shape, etc.)
            
        Returns:
            Self for method chaining
        """
        layer = DenseLayer(units, activation, name=name, **kwargs)
        self._add_layer_to_graph(layer, name)
        return self
        
    def flatten(self, name: Optional[str] = None) -> 'Diagram':
        """
        Add a flatten layer to the diagram.
        
        Args:
            name: Optional name for the layer. If not provided, auto-generated.
        
        Returns:
            Self for method chaining
        """
        layer = FlattenLayer(name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def dropout(self, rate: float, name: Optional[str] = None) -> 'Diagram':
        """
        Add a dropout layer to the diagram.
        
        Args:
            rate: Dropout rate between 0.0 and 1.0
            name: Optional name for the layer. If not provided, auto-generated.
            
        Returns:
            Self for method chaining
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
            
        layer = DropoutLayer(rate, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def output(self, units: int, activation: str = "softmax", name: Optional[str] = None) -> 'Diagram':
        """
        Add an output layer to the diagram.
        
        Args:
            units: Number of output units
            activation: Activation function name (default: "softmax")
            name: Optional name for the layer. If not provided, auto-generated.
            
        Returns:
            Self for method chaining
        """
        layer = OutputLayer(units, activation, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def connect(self, source_layer_name: str, dest_layer_name: str,
                connection_type: str = 'default', weight: Optional[float] = None,
                style: str = 'solid', label: str = '') -> 'Diagram':
        """
        Create a connection between two layers in the graph.
        
        Args:
            source_layer_name: Name of the source layer
            dest_layer_name: Name of the destination layer
            connection_type: Type of connection ('default', 'skip', 'residual', 'attention', 'feedback')
            weight: Optional weight for the connection (for weighted connections)
            style: Visual style of the connection ('solid', 'dashed', 'dotted', 'bold')
            label: Optional label for the connection
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If either layer doesn't exist
        """
        if source_layer_name not in self.graph:
            raise ValueError(f"Source layer '{source_layer_name}' does not exist")
        if dest_layer_name not in self.graph:
            raise ValueError(f"Destination layer '{dest_layer_name}' does not exist")
        
        # Store connection attributes as edge data
        edge_data = {
            'type': connection_type,
            'style': style,
            'label': label
        }
        if weight is not None:
            edge_data['weight'] = weight
        
        self.graph.add_edge(source_layer_name, dest_layer_name, **edge_data)
        return self
        
    def maxpool(self, pool_size: Union[int, Tuple[int, int]] = 2, 
                stride: Optional[Union[int, Tuple[int, int]]] = None, name: Optional[str] = None) -> 'Diagram':
        """
        Add a max pooling layer to the diagram.
        
        Args:
            pool_size: Size of pooling window (default: 2)
            stride: Stride of pooling operation (default: same as pool_size)
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = MaxPoolLayer(pool_size, stride, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def upsample(self, size: Union[int, Tuple[int, int]] = 2, method: str = "nearest", 
                 name: Optional[str] = None) -> 'Diagram':
        """
        Add an upsampling layer to the diagram.
        
        Args:
            size: Upsampling factor (default: 2)
            method: Upsampling method ('nearest', 'bilinear') (default: "nearest")
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = UpSampleLayer(size, method, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def conv_transpose(self, filters: int, kernel_size: Union[int, Tuple[int, int]], 
                       stride: int = 1, activation: str = "relu", name: Optional[str] = None) -> 'Diagram':
        """
        Add a transposed convolution layer to the diagram.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of convolution kernel
            stride: Stride of convolution (default: 1)
            activation: Activation function name (default: "relu")
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = ConvTransposeLayer(filters, kernel_size, stride, activation, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def batch_norm(self, name: Optional[str] = None) -> 'Diagram':
        """
        Add a batch normalization layer to the diagram.
        
        Args:
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = BatchNormLayer(name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def layer_norm(self, name: Optional[str] = None) -> 'Diagram':
        """
        Add a layer normalization layer to the diagram.
        
        Args:
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = LayerNormLayer(name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def multi_head_attention(self, num_heads: int, key_dim: int, name: Optional[str] = None) -> 'Diagram':
        """
        Add a multi-head attention layer to the diagram.
        
        Args:
            num_heads: Number of attention heads
            key_dim: Dimension of attention keys/queries
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = MultiHeadAttentionLayer(num_heads, key_dim, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def embedding(self, vocab_size: int, embed_dim: int, name: Optional[str] = None) -> 'Diagram':
        """
        Add an embedding layer to the diagram.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = EmbeddingLayer(vocab_size, embed_dim, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def positional_encoding(self, max_len: int, embed_dim: int, name: Optional[str] = None) -> 'Diagram':
        """
        Add a positional encoding layer to the diagram.
        
        Args:
            max_len: Maximum sequence length
            embed_dim: Embedding dimension
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = PositionalEncodingLayer(max_len, embed_dim, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def reshape(self, target_shape: Tuple[int, ...], name: Optional[str] = None) -> 'Diagram':
        """
        Add a reshape layer to the diagram.
        
        Args:
            target_shape: Target shape
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = ReshapeLayer(target_shape, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def global_avg_pool(self, name: Optional[str] = None) -> 'Diagram':
        """
        Add a global average pooling layer to the diagram.
        
        Args:
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = GlobalAvgPoolLayer(name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def concatenate(self, axis: int = -1, name: Optional[str] = None) -> 'Diagram':
        """
        Add a concatenate layer to the diagram.
        
        Args:
            axis: Axis along which to concatenate (default: -1)
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = ConcatenateLayer(axis, name=name)
        self._add_layer_to_graph(layer, name)
        return self
        
    def add(self, name: Optional[str] = None) -> 'Diagram':
        """
        Add an element-wise addition layer to the diagram.
        
        Args:
            name: Optional name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = AddLayer(name=name)
        self._add_layer_to_graph(layer, name)
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
            
        svg_content = self._renderer.render(self.graph, theme_obj)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
            
        return filename
        
    @property
    def layers(self) -> List[Layer]:
        """
        Get list of layers for backward compatibility.
        
        Returns:
            List of Layer objects in the diagram
        """
        return [self.graph.nodes[node]['layer'] for node in self.graph.nodes()]
    
    def get_layer_names(self) -> List[str]:
        """
        Get list of layer names in the diagram.
        
        Returns:
            List of layer names
        """
        return list(self.graph.nodes())
        
    def _get_theme_by_name(self, theme_name: str) -> Theme:
        """Get theme object by name."""
        from .themes import APJTheme, MinimalTheme, DarkTheme
        
        theme_map = {
            "ieee": IEEETheme(),
            "apj": APJTheme(),
            "minimal": MinimalTheme(), 
            "dark": DarkTheme()
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
        return len(self.graph.nodes())
        
    def __repr__(self) -> str:
        """String representation of the diagram."""
        layer_types = [self.graph.nodes[node]['layer'].layer_type for node in self.graph.nodes()]
        return f"Diagram({layer_types})"