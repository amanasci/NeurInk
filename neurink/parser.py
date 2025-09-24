"""
DSL parser for neural network diagram definitions.

Parses a lightweight domain-specific language for defining
neural network architectures.
"""

from typing import Dict, Any, List
from .diagram import Diagram


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
            
        Note:
            This is a basic implementation. Full DSL support will be added later.
        """
        diagram = Diagram()
        
        # Basic line-by-line parsing (will be enhanced with lark/pyparsing)
        lines = [line.strip() for line in dsl_text.strip().split('\n') if line.strip()]
        
        for line in lines:
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
                
        return diagram
        
    def _parse_input(self, line: str, diagram: Diagram) -> None:
        """Parse input layer definition."""
        # Example: input size=64x64 or input size=784
        if 'size=' in line:
            size_part = line.split('size=')[1].strip()
            if 'x' in size_part:
                # Multi-dimensional input
                dims = tuple(int(x) for x in size_part.split('x'))
                diagram.input(dims)
            else:
                # Single dimension input
                diagram.input(int(size_part))
                
    def _parse_conv(self, line: str, diagram: Diagram) -> None:
        """Parse convolutional layer definition."""
        # Example: conv filters=32 kernel=3 stride=1 activation=relu
        params = self._parse_params(line)
        filters = int(params.get('filters', 32))
        kernel = int(params.get('kernel', 3))
        stride = int(params.get('stride', 1))
        activation = params.get('activation', 'relu')
        
        diagram.conv(filters, kernel, stride, activation)
        
    def _parse_dense(self, line: str, diagram: Diagram) -> None:
        """Parse dense layer definition."""
        # Example: dense units=128 activation=relu
        params = self._parse_params(line)
        units = int(params.get('units', 128))
        activation = params.get('activation', 'relu')
        
        diagram.dense(units, activation)
        
    def _parse_dropout(self, line: str, diagram: Diagram) -> None:
        """Parse dropout layer definition."""
        # Example: dropout rate=0.5
        params = self._parse_params(line)
        rate = float(params.get('rate', 0.5))
        
        diagram.dropout(rate)
        
    def _parse_output(self, line: str, diagram: Diagram) -> None:
        """Parse output layer definition."""
        # Example: output units=10 activation=softmax
        params = self._parse_params(line)
        units = int(params.get('units', 10))
        activation = params.get('activation', 'softmax')
        
        diagram.output(units, activation)
        
    def _parse_params(self, line: str) -> Dict[str, str]:
        """Parse key=value parameters from a line."""
        params = {}
        parts = line.split()
        
        for part in parts[1:]:  # Skip the layer type
            if '=' in part:
                key, value = part.split('=', 1)
                params[key] = value
                
        return params