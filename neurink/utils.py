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


def process_latex_string(text: str) -> str:
    """
    Process a string containing LaTeX mathematical expressions.
    
    Converts simple LaTeX expressions to Unicode equivalents for better display.
    More complex expressions are left as-is for potential future LaTeX rendering.
    
    Args:
        text: String potentially containing LaTeX expressions
        
    Returns:
        Processed string with Unicode replacements where possible
    """
    import re
    
    # Basic LaTeX to Unicode mappings
    latex_symbols = {
        r'\\alpha': 'α',
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        r'\\delta': 'δ',
        r'\\epsilon': 'ε',
        r'\\theta': 'θ',
        r'\\lambda': 'λ',
        r'\\mu': 'μ',
        r'\\pi': 'π',
        r'\\sigma': 'σ',
        r'\\tau': 'τ',
        r'\\phi': 'φ',
        r'\\psi': 'ψ',
        r'\\omega': 'ω',
        r'\\Gamma': 'Γ',
        r'\\Delta': 'Δ',
        r'\\Theta': 'Θ',
        r'\\Lambda': 'Λ',
        r'\\Pi': 'Π',
        r'\\Sigma': 'Σ',
        r'\\Phi': 'Φ',
        r'\\Psi': 'Ψ',
        r'\\Omega': 'Ω',
        r'\\rightarrow': '→',
        r'\\leftarrow': '←',
        r'\\Rightarrow': '⇒',
        r'\\Leftarrow': '⇐',
        r'\\infty': '∞',
        r'\\partial': '∂',
        r'\\nabla': '∇',
        r'\\sum': '∑',
        r'\\prod': '∏',
        r'\\int': '∫',
        r'\\pm': '±',
        r'\\leq': '≤',
        r'\\geq': '≥',
        r'\\neq': '≠',
        r'\\approx': '≈',
    }
    
    # Replace basic LaTeX symbols first
    result = text
    for latex, unicode_char in latex_symbols.items():
        result = re.sub(latex + r'\b', unicode_char, result)
    
    # Handle simple subscripts and superscripts
    # Convert _n to subscript n, ^n to superscript n
    subscript_map = str.maketrans('0123456789+-=()abcdefghijklmnopqrstuvwxyz',
                                  '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐᵦᵧₐₑfgₕᵢⱼₖₗₘₙₒₚqᵣₛₜᵤᵥwₓᵧz')
    superscript_map = str.maketrans('0123456789+-=()abcdefghijklmnopqrstuvwxyz',
                                    '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖqʳˢᵗᵘᵛʷˣʸᶻ')
    
    # Simple subscript replacement
    result = re.sub(r'_(\w)', lambda m: m.group(1).translate(subscript_map), result)
    # Simple superscript replacement  
    result = re.sub(r'\^(\w)', lambda m: m.group(1).translate(superscript_map), result)
    
    return result


def escape_html(text: str) -> str:
    """
    Escape HTML special characters in text.
    
    Args:
        text: Text to escape
        
    Returns:
        HTML-escaped text
    """
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&#x27;",
        ">": "&gt;",
        "<": "&lt;",
    }
    return "".join(html_escape_table.get(c, c) for c in text)