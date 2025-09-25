"""
Core Diagram class for building neural network diagrams.

This module provides the main Diagram class that supports method chaining
for building neural network architectures.
"""

from typing import List, Union, Tuple, Optional, Dict, Any
from contextlib import contextmanager
from .layer import (
    Layer, InputLayer, ConvLayer, DenseLayer, 
    FlattenLayer, DropoutLayer, OutputLayer,
    AttentionLayer, LayerNormLayer, EmbeddingLayer,
    PoolingLayer, BatchNormLayer, SkipConnectionLayer,
    BranchLayer, MergeLayer
)
from .renderer import SVGRenderer
from .professional_renderer import ProfessionalSVGRenderer
from .themes import Theme, IEEETheme


class Connection:
    """Represents a connection between two layers."""
    
    def __init__(self, source_name: str, target_name: str, style: str = "skip", **kwargs):
        """
        Initialize a connection.
        
        Args:
            source_name: Name of the source layer
            target_name: Name of the target layer
            style: Style of connection ('skip', 'attention', 'residual', etc.)
            **kwargs: Additional styling parameters
        """
        self.source_name = source_name
        self.target_name = target_name
        self.style = style
        self.params = kwargs


class LayerGroup:
    """Represents a group of layers with visual styling."""
    
    def __init__(self, name: str, style: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize a layer group.
        
        Args:
            name: Name of the group
            style: Visual styling for the group
            **kwargs: Additional parameters
        """
        self.name = name
        self.style = style or {}
        self.layers: List[str] = []  # Store layer names in this group
        self.params = kwargs


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
        self.connections: List[Connection] = []  # Track custom connections
        self.groups: List[LayerGroup] = []  # Track layer groups
        self._current_group: Optional[LayerGroup] = None  # For context manager
        self._layer_counter = 0  # For generating unique layer names
        
    def _add_layer(self, layer: Layer, name: Optional[str] = None) -> 'Diagram':
        """
        Helper method to add a layer with proper naming and group tracking.
        
        Args:
            layer: Layer to add
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        # Assign name if not provided
        if name:
            layer.name = name
        else:
            self._layer_counter += 1
            layer.name = f"{layer.layer_type}_{self._layer_counter}"
            
        # Add to current group if active
        if self._current_group:
            self._current_group.layers.append(layer.name)
            
        self.layers.append(layer)
        return self
        
    def add_connection(self, source_name: str, target_name: str, style: str = "skip", **kwargs) -> 'Diagram':
        """
        Add a custom connection between two layers.
        
        Args:
            source_name: Name of the source layer
            target_name: Name of the target layer  
            style: Style of connection ('skip', 'attention', 'residual', etc.)
            **kwargs: Additional styling parameters
            
        Returns:
            Self for method chaining
        """
        connection = Connection(source_name, target_name, style, **kwargs)
        self.connections.append(connection)
        return self
        
    @contextmanager
    def group(self, name: str, style: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Context manager for grouping layers.
        
        Args:
            name: Name of the group
            style: Visual styling for the group
            **kwargs: Additional parameters
            
        Example:
            with diagram.group("Encoder", style={"fill": "#f0f0f0"}) as encoder:
                encoder.conv(32, 3)
                encoder.conv(64, 3)
        """
        group = LayerGroup(name, style, **kwargs)
        self.groups.append(group)
        
        # Set as current group
        old_group = self._current_group
        self._current_group = group
        
        try:
            yield self  # Return the diagram for chaining
        finally:
            # Restore previous group
            self._current_group = old_group
        
    def input(self, shape: Union[Tuple[int, ...], int], name: Optional[str] = None) -> 'Diagram':
        """
        Add an input layer to the diagram.
        
        Args:
            shape: Input shape as tuple (height, width) or (channels, height, width) 
                  or single integer
            name: Optional custom name for the layer
                  
        Returns:
            Self for method chaining
        """
        layer = InputLayer(shape)
        return self._add_layer(layer, name)
        
    def conv(self, filters: int, kernel_size: Union[int, Tuple[int, int]], 
             stride: int = 1, activation: str = "relu", name: Optional[str] = None) -> 'Diagram':
        """
        Add a convolutional layer to the diagram.
        
        Args:
            filters: Number of output filters/channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution (default: 1)
            activation: Activation function name (default: "relu")
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = ConvLayer(filters, kernel_size, stride, activation)
        return self._add_layer(layer, name)
        
    def dense(self, units: int, activation: str = "relu", name: Optional[str] = None) -> 'Diagram':
        """
        Add a dense (fully connected) layer to the diagram.
        
        Args:
            units: Number of output units
            activation: Activation function name (default: "relu")
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = DenseLayer(units, activation)
        return self._add_layer(layer, name)
        
    def flatten(self, name: Optional[str] = None) -> 'Diagram':
        """
        Add a flatten layer to the diagram.
        
        Args:
            name: Optional custom name for the layer
        
        Returns:
            Self for method chaining
        """
        layer = FlattenLayer()
        return self._add_layer(layer, name)
        
    def dropout(self, rate: float, name: Optional[str] = None) -> 'Diagram':
        """
        Add a dropout layer to the diagram.
        
        Args:
            rate: Dropout rate between 0.0 and 1.0
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Dropout rate must be between 0.0 and 1.0")
            
        layer = DropoutLayer(rate)
        return self._add_layer(layer, name)
        
    def output(self, units: int, activation: str = "softmax", name: Optional[str] = None) -> 'Diagram':
        """
        Add an output layer to the diagram.
        
        Args:
            units: Number of output units
            activation: Activation function name (default: "softmax")
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = OutputLayer(units, activation)
        return self._add_layer(layer, name)
        
    def attention(self, num_heads: int = 8, key_dim: int = 64, name: Optional[str] = None) -> 'Diagram':
        """
        Add a multi-head attention layer to the diagram.
        
        Args:
            num_heads: Number of attention heads (default: 8)
            key_dim: Dimension of keys/queries (default: 64)
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = AttentionLayer(num_heads, key_dim)
        return self._add_layer(layer, name)
        
    def layer_norm(self, name: Optional[str] = None) -> 'Diagram':
        """
        Add a layer normalization layer to the diagram.
        
        Args:
            name: Optional custom name for the layer
        
        Returns:
            Self for method chaining
        """
        layer = LayerNormLayer()
        return self._add_layer(layer, name)
        
    def embedding(self, vocab_size: int, embed_dim: int, name: Optional[str] = None) -> 'Diagram':
        """
        Add an embedding layer to the diagram.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = EmbeddingLayer(vocab_size, embed_dim)
        return self._add_layer(layer, name)
        
    def pooling(self, pool_type: str = "max", pool_size: int = 2, stride: int = 2, name: Optional[str] = None) -> 'Diagram':
        """
        Add a pooling layer to the diagram.
        
        Args:
            pool_type: Type of pooling ('max', 'avg', 'global_avg')
            pool_size: Size of pooling window (default: 2)
            stride: Stride of pooling (default: 2)
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = PoolingLayer(pool_type, pool_size, stride)
        return self._add_layer(layer, name)
        
    def batch_norm(self, name: Optional[str] = None) -> 'Diagram':
        """
        Add a batch normalization layer to the diagram.
        
        Args:
            name: Optional custom name for the layer
        
        Returns:
            Self for method chaining
        """
        layer = BatchNormLayer()
        return self._add_layer(layer, name)
        
    def branch(self, branch_name: str, name: Optional[str] = None) -> 'Diagram':
        """
        Create a branch point in the diagram.
        
        Args:
            branch_name: Name identifier for the branch
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = BranchLayer(branch_name)
        return self._add_layer(layer, name)
        
    def merge(self, merge_type: str = "add", merge_with: str = None, name: Optional[str] = None) -> 'Diagram':
        """
        Merge with a previously branched path.
        
        Args:
            merge_type: Type of merge operation ('add', 'concat')
            merge_with: Name of branch to merge with
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = MergeLayer(merge_type, merge_with)
        return self._add_layer(layer, name)
        
    def skip_connection(self, connection_type: str = "add", name: Optional[str] = None) -> 'Diagram':
        """
        Add a skip connection (shortcut) to the diagram.
        
        Args:
            connection_type: Type of connection ('add', 'concat')
            name: Optional custom name for the layer
            
        Returns:
            Self for method chaining
        """
        layer = SkipConnectionLayer(connection_type)
        return self._add_layer(layer, name)
        
    def render(self, filename: str, theme: Union[str, Theme] = "ieee") -> str:
        """
        Render the diagram to SVG format.
        
        Args:
            filename: Output filename for the SVG file
            theme: Theme name ("ieee", "apj", "minimal", "dark", "nnsvg") or Theme object
            
        Returns:
            Path to the generated SVG file
        """
        if isinstance(theme, str):
            theme_obj = self._get_theme_by_name(theme)
            # Use professional renderer for nnsvg theme
            if theme == "nnsvg":
                renderer = ProfessionalSVGRenderer()
                svg_content = renderer.render_diagram(self, theme_obj)
            else:
                renderer = self._renderer
                svg_content = renderer.render(self.layers, theme_obj)
        else:
            theme_obj = theme
            renderer = self._renderer
            svg_content = renderer.render(self.layers, theme_obj)
        
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