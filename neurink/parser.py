"""
DSL parser for neural network diagram definitions.

Enhanced parser to support named layers and explicit connections
for complex network architectures.
"""

from typing import Dict, Any, List, Optional
from .diagram import Diagram


class DSLParseError(Exception):
    """Exception raised for DSL parsing errors."""
    pass


class DSLParser:
    """Enhanced parser for NeurInk DSL syntax."""
    
    def __init__(self):
        """Initialize the DSL parser."""
        pass
        
    def parse(self, dsl_text: str) -> Diagram:
        """
        Parse DSL text and return a Diagram object.
        
        Args:
            dsl_text: DSL text defining the network
            
        Returns:
            Diagram object
            
        Raises:
            DSLParseError: If the DSL text contains syntax errors
        """
        if not isinstance(dsl_text, str):
            raise DSLParseError("DSL text must be a string")
        
        # Clean up the DSL text
        dsl_text = dsl_text.strip()
        if not dsl_text:
            return Diagram()
        
        diagram = Diagram()
        lines = [line.strip() for line in dsl_text.strip().split('\n') if line.strip()]
        
        for line_num, line in enumerate(lines, 1):
            try:
                if line.startswith('input'):
                    self._parse_input(line, diagram)
                elif line.startswith('conv_transpose'):
                    self._parse_conv_transpose(line, diagram)
                elif line.startswith('conv'):
                    self._parse_conv(line, diagram)
                elif line.startswith('dense'):
                    self._parse_dense(line, diagram)
                elif line.startswith('flatten'):
                    self._parse_flatten(line, diagram)
                elif line.startswith('dropout'):
                    self._parse_dropout(line, diagram)
                elif line.startswith('output'):
                    self._parse_output(line, diagram)
                elif line.startswith('maxpool'):
                    self._parse_maxpool(line, diagram)
                elif line.startswith('upsample'):
                    self._parse_upsample(line, diagram)
                elif line.startswith('batch_norm'):
                    self._parse_batch_norm(line, diagram)
                elif line.startswith('layer_norm'):
                    self._parse_layer_norm(line, diagram)
                elif line.startswith('multi_head_attention'):
                    self._parse_multi_head_attention(line, diagram)
                elif line.startswith('embedding'):
                    self._parse_embedding(line, diagram)
                elif line.startswith('positional_encoding'):
                    self._parse_positional_encoding(line, diagram)
                elif line.startswith('reshape'):
                    self._parse_reshape(line, diagram)
                elif line.startswith('global_avg_pool'):
                    self._parse_global_avg_pool(line, diagram)
                elif line.startswith('concatenate'):
                    self._parse_concatenate(line, diagram)
                elif line.startswith('add'):
                    self._parse_add(line, diagram)
                elif line.startswith('connect'):
                    self._parse_connect(line, diagram)
                else:
                    raise DSLParseError(f"Unknown layer type at line {line_num}: '{line}'")
            except (ValueError, KeyError, AttributeError) as e:
                raise DSLParseError(f"Error parsing line {line_num} '{line}': {e}") from e
                
        return diagram
        
    def _parse_input(self, line: str, diagram: Diagram) -> None:
        """Parse input layer definition."""
        params = self._parse_params(line)
        
        if 'size' not in params:
            raise ValueError("Input layer missing required 'size' parameter")
            
        size_part = params['size']
        if not size_part:
            raise ValueError("Input size cannot be empty")
            
        try:
            if 'x' in size_part:
                dims = tuple(int(x) for x in size_part.split('x'))
                if any(d <= 0 for d in dims):
                    raise ValueError("All dimensions must be positive")
                shape = dims
            else:
                dim = int(size_part)
                if dim <= 0:
                    raise ValueError("Input dimension must be positive")
                shape = dim
                
            name = params.get('name')
            diagram.input(shape, name=name)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Invalid dimension value in '{size_part}'")
            raise
                
    def _parse_conv(self, line: str, diagram: Diagram) -> None:
        """Parse convolutional layer definition."""
        params = self._parse_params(line)
        
        if 'filters' not in params:
            raise ValueError("Convolutional layer missing required 'filters' parameter")
        if 'kernel' not in params:
            raise ValueError("Convolutional layer missing required 'kernel' parameter")
            
        try:
            filters = int(params['filters'])
            kernel = int(params['kernel'])
            stride = int(params.get('stride', 1))
            activation = params.get('activation', 'relu')
            name = params.get('name')
            
            if filters <= 0:
                raise ValueError("Number of filters must be positive")
            if kernel <= 0:
                raise ValueError("Kernel size must be positive")
            if stride <= 0:
                raise ValueError("Stride must be positive")
                
            diagram.conv(filters, kernel, stride, activation, name=name)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid numeric parameter in conv layer")
            raise
        
    def _parse_dense(self, line: str, diagram: Diagram) -> None:
        """Parse dense layer definition."""
        params = self._parse_params(line)
        
        if 'units' not in params:
            raise ValueError("Dense layer missing required 'units' parameter")
        
        if not params['units']:  # Check for empty value
            raise ValueError("Invalid parameter format")
            
        try:
            units = int(params['units'])
            if units <= 0:
                raise ValueError("Number of units must be positive")
                
            activation = params.get('activation', 'relu')
            name = params.get('name')
            
            diagram.dense(units, activation, name=name)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid units parameter in dense layer")
            raise
    
    def _parse_flatten(self, line: str, diagram: Diagram) -> None:
        """Parse flatten layer definition."""
        params = self._parse_params(line)
        name = params.get('name')
        diagram.flatten(name=name)
        
    def _parse_dropout(self, line: str, diagram: Diagram) -> None:
        """Parse dropout layer definition."""
        params = self._parse_params(line)
        
        if 'rate' not in params:
            raise ValueError("Dropout layer missing required 'rate' parameter")
            
        try:
            rate = float(params['rate'])
            if not 0.0 <= rate <= 1.0:
                raise ValueError("Dropout rate must be between 0.0 and 1.0")
            
            name = params.get('name')
            diagram.dropout(rate, name=name)
        except ValueError as e:
            if "could not convert string to float" in str(e):
                raise ValueError("Invalid rate parameter in dropout layer")
            raise
        
    def _parse_output(self, line: str, diagram: Diagram) -> None:
        """Parse output layer definition."""
        params = self._parse_params(line)
        
        if 'units' not in params:
            raise ValueError("Output layer missing required 'units' parameter")
            
        try:
            units = int(params['units'])
            if units <= 0:
                raise ValueError("Number of output units must be positive")
                
            activation = params.get('activation', 'softmax')
            name = params.get('name')
            
            diagram.output(units, activation, name=name)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid units parameter in output layer")
            raise
    
    def _parse_connect(self, line: str, diagram: Diagram) -> None:
        """Parse connection statement."""
        params = self._parse_params(line)
        
        if 'from' not in params:
            raise ValueError("Connection missing required 'from' parameter")
        if 'to' not in params:
            raise ValueError("Connection missing required 'to' parameter")
        
        source = params['from']
        target = params['to']
        
        if not source or not target:
            raise ValueError("Connection source and target cannot be empty")
        
        diagram.connect(source, target)
        
    def _parse_params(self, line: str) -> Dict[str, str]:
        """Parse key=value parameters from a line."""
        params = {}
        parts = line.split()
        
        for part in parts[1:]:  # Skip the layer type
            if '=' in part:
                key, value = part.split('=', 1)
                if not key:
                    raise ValueError(f"Invalid parameter format: '{part}'. Expected 'key=value'")
                # Allow empty values here, they will be checked in specific parsers
                params[key] = value
            elif part.strip():  # Ignore empty parts but error on non-empty non-parameter parts
                raise ValueError(f"Invalid parameter format: '{part}'. Expected 'key=value'")
                    
        return params
    
    def _parse_conv_transpose(self, line: str, diagram: Diagram) -> None:
        """Parse transposed convolution layer definition."""
        params = self._parse_params(line)
        
        if 'filters' not in params:
            raise ValueError("ConvTranspose layer missing required 'filters' parameter")
        if 'kernel' not in params:
            raise ValueError("ConvTranspose layer missing required 'kernel' parameter")
            
        try:
            filters = int(params['filters'])
            kernel = int(params['kernel'])
            stride = int(params.get('stride', 1))
            activation = params.get('activation', 'relu')
            name = params.get('name')
            
            if filters <= 0:
                raise ValueError("Number of filters must be positive")
            if kernel <= 0:
                raise ValueError("Kernel size must be positive")
            if stride <= 0:
                raise ValueError("Stride must be positive")
                
            diagram.conv_transpose(filters, kernel, stride, activation, name=name)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid numeric parameter in conv_transpose layer")
            raise
    
    def _parse_maxpool(self, line: str, diagram: Diagram) -> None:
        """Parse max pooling layer definition."""
        params = self._parse_params(line)
        
        pool_size = int(params.get('pool_size', 2))
        stride = int(params['stride']) if 'stride' in params else None
        name = params.get('name')
        
        diagram.maxpool(pool_size, stride, name=name)
    
    def _parse_upsample(self, line: str, diagram: Diagram) -> None:
        """Parse upsampling layer definition."""
        params = self._parse_params(line)
        
        size = int(params.get('size', 2))
        method = params.get('method', 'nearest')
        name = params.get('name')
        
        diagram.upsample(size, method, name=name)
    
    def _parse_batch_norm(self, line: str, diagram: Diagram) -> None:
        """Parse batch normalization layer definition."""
        params = self._parse_params(line)
        name = params.get('name')
        diagram.batch_norm(name=name)
    
    def _parse_layer_norm(self, line: str, diagram: Diagram) -> None:
        """Parse layer normalization layer definition."""
        params = self._parse_params(line)
        name = params.get('name')
        diagram.layer_norm(name=name)
    
    def _parse_multi_head_attention(self, line: str, diagram: Diagram) -> None:
        """Parse multi-head attention layer definition."""
        params = self._parse_params(line)
        
        if 'num_heads' not in params:
            raise ValueError("MultiHeadAttention layer missing required 'num_heads' parameter")
        if 'key_dim' not in params:
            raise ValueError("MultiHeadAttention layer missing required 'key_dim' parameter")
            
        try:
            num_heads = int(params['num_heads'])
            key_dim = int(params['key_dim'])
            name = params.get('name')
            
            if num_heads <= 0:
                raise ValueError("Number of heads must be positive")
            if key_dim <= 0:
                raise ValueError("Key dimension must be positive")
                
            diagram.multi_head_attention(num_heads, key_dim, name=name)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid numeric parameter in multi_head_attention layer")
            raise
    
    def _parse_embedding(self, line: str, diagram: Diagram) -> None:
        """Parse embedding layer definition."""
        params = self._parse_params(line)
        
        if 'vocab_size' not in params:
            raise ValueError("Embedding layer missing required 'vocab_size' parameter")
        if 'embed_dim' not in params:
            raise ValueError("Embedding layer missing required 'embed_dim' parameter")
            
        try:
            vocab_size = int(params['vocab_size'])
            embed_dim = int(params['embed_dim'])
            name = params.get('name')
            
            if vocab_size <= 0:
                raise ValueError("Vocabulary size must be positive")
            if embed_dim <= 0:
                raise ValueError("Embedding dimension must be positive")
                
            diagram.embedding(vocab_size, embed_dim, name=name)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid numeric parameter in embedding layer")
            raise
    
    def _parse_positional_encoding(self, line: str, diagram: Diagram) -> None:
        """Parse positional encoding layer definition."""
        params = self._parse_params(line)
        
        if 'max_len' not in params:
            raise ValueError("PositionalEncoding layer missing required 'max_len' parameter")
        if 'embed_dim' not in params:
            raise ValueError("PositionalEncoding layer missing required 'embed_dim' parameter")
            
        try:
            max_len = int(params['max_len'])
            embed_dim = int(params['embed_dim'])
            name = params.get('name')
            
            if max_len <= 0:
                raise ValueError("Maximum length must be positive")
            if embed_dim <= 0:
                raise ValueError("Embedding dimension must be positive")
                
            diagram.positional_encoding(max_len, embed_dim, name=name)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid numeric parameter in positional_encoding layer")
            raise
    
    def _parse_reshape(self, line: str, diagram: Diagram) -> None:
        """Parse reshape layer definition."""
        params = self._parse_params(line)
        
        if 'shape' not in params:
            raise ValueError("Reshape layer missing required 'shape' parameter")
            
        try:
            shape_str = params['shape']
            if 'x' in shape_str:
                shape = tuple(int(x) for x in shape_str.split('x'))
            else:
                shape = (int(shape_str),)
            name = params.get('name')
            
            diagram.reshape(shape, name=name)
        except ValueError as e:
            raise ValueError("Invalid shape parameter in reshape layer")
    
    def _parse_global_avg_pool(self, line: str, diagram: Diagram) -> None:
        """Parse global average pooling layer definition."""
        params = self._parse_params(line)
        name = params.get('name')
        diagram.global_avg_pool(name=name)
    
    def _parse_concatenate(self, line: str, diagram: Diagram) -> None:
        """Parse concatenate layer definition."""
        params = self._parse_params(line)
        
        axis = int(params.get('axis', -1))
        name = params.get('name')
        
        diagram.concatenate(axis, name=name)
    
    def _parse_add(self, line: str, diagram: Diagram) -> None:
        """Parse add layer definition."""
        params = self._parse_params(line)
        name = params.get('name')
        diagram.add(name=name)