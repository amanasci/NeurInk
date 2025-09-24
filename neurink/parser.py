"""
DSL parser for neural network diagram definitions.

Parses a lightweight domain-specific language for defining
neural network architectures.
"""

from typing import Dict, Any, List
from .diagram import Diagram


class DSLParseError(Exception):
    """Exception raised for DSL parsing errors."""
    pass


class DSLParser:
    """Parser for NeurInk DSL syntax."""
    
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
            
        diagram = Diagram()
        
        # Basic line-by-line parsing (will be enhanced with lark/pyparsing)
        lines = [line.strip() for line in dsl_text.strip().split('\n') if line.strip()]
        
        for line_num, line in enumerate(lines, 1):
            try:
                if line.startswith('input'):
                    self._parse_input(line, diagram)
                elif line.startswith('conv'):
                    self._parse_conv(line, diagram)
                elif line.startswith('dense'):
                    self._parse_dense(line, diagram)
                elif line.startswith('flatten'):
                    diagram.flatten()
                elif line.startswith('dropout'):
                    self._parse_dropout(line, diagram)
                elif line.startswith('output'):
                    self._parse_output(line, diagram)
                else:
                    raise DSLParseError(f"Unknown layer type at line {line_num}: '{line}'")
            except (ValueError, KeyError, AttributeError) as e:
                raise DSLParseError(f"Error parsing line {line_num} '{line}': {e}") from e
                
        return diagram
        
    def _parse_input(self, line: str, diagram: Diagram) -> None:
        """Parse input layer definition."""
        # Example: input size=64x64 or input size=784
        if 'size=' not in line:
            raise ValueError("Input layer missing required 'size' parameter")
            
        size_part = line.split('size=')[1].strip()
        if not size_part:
            raise ValueError("Input size cannot be empty")
            
        try:
            if 'x' in size_part:
                # Multi-dimensional input
                dims = tuple(int(x) for x in size_part.split('x'))
                if any(d <= 0 for d in dims):
                    raise ValueError("All dimensions must be positive")
                diagram.input(dims)
            else:
                # Single dimension input
                dim = int(size_part)
                if dim <= 0:
                    raise ValueError("Input dimension must be positive")
                diagram.input(dim)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Invalid dimension value in '{size_part}'")
            raise
                
    def _parse_conv(self, line: str, diagram: Diagram) -> None:
        """Parse convolutional layer definition."""
        # Example: conv filters=32 kernel=3 stride=1 activation=relu
        params = self._parse_params(line)
        
        if 'filters' not in params:
            raise ValueError("Convolutional layer missing required 'filters' parameter")
        if 'kernel' not in params:
            raise ValueError("Convolutional layer missing required 'kernel' parameter")
            
        try:
            filters = int(params['filters'])
            kernel = int(params['kernel'])
            stride = int(params.get('stride', 1))
            
            if filters <= 0:
                raise ValueError("Number of filters must be positive")
            if kernel <= 0:
                raise ValueError("Kernel size must be positive")
            if stride <= 0:
                raise ValueError("Stride must be positive")
                
            activation = params.get('activation', 'relu')
            if not activation:
                raise ValueError("Activation cannot be empty")
                
            diagram.conv(filters, kernel, stride, activation)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid numeric parameter in conv layer")
            raise
        
    def _parse_dense(self, line: str, diagram: Diagram) -> None:
        """Parse dense layer definition."""
        # Example: dense units=128 activation=relu
        params = self._parse_params(line)
        
        if 'units' not in params:
            raise ValueError("Dense layer missing required 'units' parameter")
            
        try:
            units = int(params['units'])
            if units <= 0:
                raise ValueError("Number of units must be positive")
                
            activation = params.get('activation', 'relu')
            if not activation:
                raise ValueError("Activation cannot be empty")
                
            diagram.dense(units, activation)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid units parameter in dense layer")
            raise
        
    def _parse_dropout(self, line: str, diagram: Diagram) -> None:
        """Parse dropout layer definition."""
        # Example: dropout rate=0.5
        params = self._parse_params(line)
        
        if 'rate' not in params:
            raise ValueError("Dropout layer missing required 'rate' parameter")
            
        try:
            rate = float(params['rate'])
            if not 0.0 <= rate <= 1.0:
                raise ValueError("Dropout rate must be between 0.0 and 1.0")
                
            diagram.dropout(rate)
        except ValueError as e:
            if "could not convert string to float" in str(e):
                raise ValueError("Invalid rate parameter in dropout layer")
            raise
        
    def _parse_output(self, line: str, diagram: Diagram) -> None:
        """Parse output layer definition."""
        # Example: output units=10 activation=softmax
        params = self._parse_params(line)
        
        if 'units' not in params:
            raise ValueError("Output layer missing required 'units' parameter")
            
        try:
            units = int(params['units'])
            if units <= 0:
                raise ValueError("Number of output units must be positive")
                
            activation = params.get('activation', 'softmax')
            if not activation:
                raise ValueError("Activation cannot be empty")
                
            diagram.output(units, activation)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError("Invalid units parameter in output layer")
            raise
        
    def _parse_params(self, line: str) -> Dict[str, str]:
        """Parse key=value parameters from a line."""
        params = {}
        parts = line.split()
        
        for part in parts[1:]:  # Skip the layer type
            if '=' in part:
                try:
                    key, value = part.split('=', 1)
                    if not key or not value:
                        raise ValueError(f"Invalid parameter format: '{part}'")
                    params[key] = value
                except ValueError:
                    raise ValueError(f"Invalid parameter format: '{part}'. Expected 'key=value'")
                    
        return params