"""
Utility functions for NeurInk library.

Common helper functions and utilities used throughout the library.
"""

from typing import List, Dict, Any, Tuple


def validate_shape(shape: Tuple[int, ...]) -> bool:
    """
    Validate that a shape tuple contains positive integers.
    
    Args:
        shape: Shape tuple to validate
        
    Returns:
        True if valid, False otherwise
    """
    return all(isinstance(dim, int) and dim > 0 for dim in shape)


def format_shape(shape: Tuple[int, ...]) -> str:
    """
    Format a shape tuple as a string.
    
    Args:
        shape: Shape tuple
        
    Returns:
        Formatted shape string
    """
    return "x".join(str(dim) for dim in shape)


def parse_shape(shape_str: str) -> Tuple[int, ...]:
    """
    Parse a shape string into a tuple.
    
    Args:
        shape_str: Shape string like "64x64" or "224x224x3"
        
    Returns:
        Shape tuple
    """
    return tuple(int(dim) for dim in shape_str.split('x'))


def calculate_output_size(input_size: int, kernel_size: int, 
                         stride: int = 1, padding: int = 0) -> int:
    """
    Calculate output size for a convolutional layer.
    
    Args:
        input_size: Input dimension size
        kernel_size: Kernel size
        stride: Stride (default: 1)
        padding: Padding (default: 0)
        
    Returns:
        Output dimension size
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


def estimate_parameters(layer_type: str, **kwargs) -> int:
    """
    Estimate number of parameters for a layer.
    
    Args:
        layer_type: Type of layer
        **kwargs: Layer parameters
        
    Returns:
        Estimated parameter count
    """
    if layer_type == "dense":
        # Assuming previous layer has 'input_units' parameters
        input_units = kwargs.get('input_units', 1000)  # Default estimate
        output_units = kwargs.get('units', 100)
        return input_units * output_units + output_units  # weights + biases
        
    elif layer_type == "conv":
        # Simplified calculation
        filters = kwargs.get('filters', 32)
        kernel_size = kwargs.get('kernel_size', 3)
        input_channels = kwargs.get('input_channels', 3)
        return filters * kernel_size * kernel_size * input_channels + filters
        
    else:
        return 0  # Other layers typically have no parameters


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.
    
    Args:
        hex_color: Hex color string like "#ffffff"
        
    Returns:
        RGB tuple (r, g, b)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to hex color string.
    
    Args:
        r: Red component (0-255)
        g: Green component (0-255) 
        b: Blue component (0-255)
        
    Returns:
        Hex color string
    """
    return f"#{r:02x}{g:02x}{b:02x}"


def darken_color(hex_color: str, factor: float = 0.8) -> str:
    """
    Darken a hex color by a factor.
    
    Args:
        hex_color: Original hex color
        factor: Darkening factor (0.0 to 1.0)
        
    Returns:
        Darkened hex color
    """
    r, g, b = hex_to_rgb(hex_color)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return rgb_to_hex(r, g, b)