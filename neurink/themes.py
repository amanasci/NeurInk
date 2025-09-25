"""
Theme system for customizing diagram appearance.

Provides predefined themes for different publication styles and
custom theme support.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Theme(ABC):
    """Base class for diagram themes."""
    
    @abstractmethod
    def get_colors(self) -> Dict[str, str]:
        """Get color scheme for the theme."""
        pass
        
    @abstractmethod  
    def get_styles(self) -> Dict[str, Any]:
        """Get styling parameters for the theme."""
        pass
        
    @abstractmethod
    def get_typography(self) -> Dict[str, str]:
        """Get typography settings for the theme."""
        pass


class IEEETheme(Theme):
    """IEEE publication style theme - clean and professional."""
    
    def get_colors(self) -> Dict[str, str]:
        """IEEE standard color scheme."""
        return {
            "background": "#ffffff",
            "layer_fill": "#f0f0f0", 
            "layer_stroke": "#333333",
            "input_fill": "#e1f5fe",
            "conv_fill": "#fff3e0",
            "dense_fill": "#f3e5f5",
            "output_fill": "#e8f5e8",
            "connection": "#666666",
            "text": "#000000"
        }
        
    def get_styles(self) -> Dict[str, Any]:
        """IEEE styling parameters."""
        return {
            "layer_width": 120,
            "layer_height": 60,
            "layer_spacing_x": 150,
            "layer_spacing_y": 100,
            "border_radius": 8,
            "stroke_width": 2,
            "connection_width": 1.5,
            "padding": 40
        }
        
    def get_typography(self) -> Dict[str, str]:
        """IEEE typography settings."""
        return {
            "font_family": "Arial, sans-serif",
            "font_size": "12px",
            "font_weight": "normal",
            "text_anchor": "middle"
        }


class APJTheme(Theme):
    """APJ (Astrophysical Journal) publication style theme."""
    
    def get_colors(self) -> Dict[str, str]:
        """APJ color scheme - more conservative."""
        return {
            "background": "#ffffff",
            "layer_fill": "#f8f8f8",
            "layer_stroke": "#000000", 
            "input_fill": "#f0f8ff",
            "conv_fill": "#fffacd",
            "dense_fill": "#f5f5dc",
            "output_fill": "#f0fff0",
            "connection": "#000000",
            "text": "#000000"
        }
        
    def get_styles(self) -> Dict[str, Any]:
        """APJ styling parameters."""
        return {
            "layer_width": 100,
            "layer_height": 50,
            "layer_spacing_x": 130,
            "layer_spacing_y": 80,
            "border_radius": 4,
            "stroke_width": 1.5,
            "connection_width": 1,
            "padding": 30
        }
        
    def get_typography(self) -> Dict[str, str]:
        """APJ typography settings."""
        return {
            "font_family": "Times, serif",
            "font_size": "11px", 
            "font_weight": "normal",
            "text_anchor": "middle"
        }


class MinimalTheme(Theme):
    """Minimal theme - clean and simple."""
    
    def get_colors(self) -> Dict[str, str]:
        """Minimal color scheme."""
        return {
            "background": "#ffffff",
            "layer_fill": "#ffffff",
            "layer_stroke": "#cccccc",
            "input_fill": "#ffffff", 
            "conv_fill": "#ffffff",
            "dense_fill": "#ffffff",
            "output_fill": "#ffffff",
            "connection": "#cccccc",
            "text": "#333333"
        }
        
    def get_styles(self) -> Dict[str, Any]:
        """Minimal styling parameters."""
        return {
            "layer_width": 100,
            "layer_height": 40,
            "layer_spacing_x": 120,
            "layer_spacing_y": 70,
            "border_radius": 2,
            "stroke_width": 1,
            "connection_width": 1,
            "padding": 20
        }
        
    def get_typography(self) -> Dict[str, str]:
        """Minimal typography settings."""
        return {
            "font_family": "Arial, sans-serif",
            "font_size": "10px",
            "font_weight": "300",
            "text_anchor": "middle"
        }


class DarkTheme(Theme):
    """Dark theme for presentations and screens."""
    
    def get_colors(self) -> Dict[str, str]:
        """Dark color scheme."""
        return {
            "background": "#2b2b2b",
            "layer_fill": "#404040",
            "layer_stroke": "#ffffff",
            "input_fill": "#1e3a8a",
            "conv_fill": "#7c2d12", 
            "dense_fill": "#581c87",
            "output_fill": "#166534",
            "connection": "#ffffff",
            "text": "#ffffff"
        }
        
    def get_styles(self) -> Dict[str, Any]:
        """Dark theme styling parameters."""
        return {
            "layer_width": 120,
            "layer_height": 60,
            "layer_spacing_x": 150,
            "layer_spacing_y": 100,
            "border_radius": 8,
            "stroke_width": 2,
            "connection_width": 2,
            "padding": 40
        }
        
    def get_typography(self) -> Dict[str, str]:
        """Dark theme typography settings."""
        return {
            "font_family": "Arial, sans-serif",
            "font_size": "12px",
            "font_weight": "normal",
            "text_anchor": "middle"
        }


class NNSVGTheme(Theme):
    """NN-SVG inspired theme - 3D layered blocks with gradients and depth."""
    
    def get_colors(self) -> Dict[str, str]:
        """NN-SVG color scheme with gradients and depth."""
        return {
            "background": "#f8f9fa",
            "layer_fill": "#ffffff",
            "layer_stroke": "#495057",
            "input_fill": "#4fc3f7",
            "conv_fill": "#ff7043", 
            "dense_fill": "#ab47bc",
            "output_fill": "#66bb6a",
            "flatten_fill": "#ffa726",
            "dropout_fill": "#ec407a",
            "attention_fill": "#7e57c2",
            "layernorm_fill": "#5c6bc0",
            "embedding_fill": "#42a5f5",
            "pooling_fill": "#26a69a",
            "batchnorm_fill": "#78909c",
            "connection": "#6c757d",
            "text": "#212529",
            "shadow": "#00000020"
        }
        
    def get_styles(self) -> Dict[str, Any]:
        """NN-SVG styling parameters with 3D effect."""
        return {
            "layer_width": 80,
            "layer_height": 50,
            "layer_depth": 12,  # 3D depth effect
            "layer_spacing_x": 140,
            "layer_spacing_y": 100,
            "border_radius": 6,
            "stroke_width": 1.5,
            "connection_width": 2,
            "padding": 50,
            "shadow_offset": 4,
            "gradient_opacity": 0.3
        }
        
    def get_typography(self) -> Dict[str, str]:
        """NN-SVG typography settings."""
        return {
            "font_family": "Source Sans Pro, Arial, sans-serif",
            "font_size": "11px",
            "font_weight": "600",
            "text_anchor": "middle"
        }