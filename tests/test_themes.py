"""
Tests for theme system functionality.
"""

import pytest
from neurink.themes import Theme, IEEETheme, APJTheme, MinimalTheme, DarkTheme


class TestThemes:
    """Test cases for theme system."""
    
    def test_ieee_theme(self):
        """Test IEEE theme properties."""
        theme = IEEETheme()
        
        colors = theme.get_colors()
        styles = theme.get_styles()
        typography = theme.get_typography()
        
        # Test color scheme
        assert colors["background"] == "#ffffff"
        assert colors["layer_fill"] == "#f0f0f0"
        assert colors["input_fill"] == "#e1f5fe"
        assert colors["conv_fill"] == "#fff3e0"
        assert colors["dense_fill"] == "#f3e5f5"
        assert colors["output_fill"] == "#e8f5e8"
        
        # Test styles
        assert styles["layer_width"] == 120
        assert styles["layer_height"] == 60
        assert styles["border_radius"] == 8
        assert styles["stroke_width"] == 2
        
        # Test typography
        assert typography["font_family"] == "Arial, sans-serif"
        assert typography["font_size"] == "12px"
        assert typography["text_anchor"] == "middle"
        
    def test_apj_theme(self):
        """Test APJ theme properties."""
        theme = APJTheme()
        
        colors = theme.get_colors()
        styles = theme.get_styles()
        typography = theme.get_typography()
        
        # Test color scheme - more conservative
        assert colors["background"] == "#ffffff"
        assert colors["layer_stroke"] == "#000000"
        assert colors["connection"] == "#000000"
        
        # Test styles - smaller dimensions
        assert styles["layer_width"] == 100
        assert styles["layer_height"] == 50
        assert styles["border_radius"] == 4
        assert styles["stroke_width"] == 1.5
        
        # Test typography - serif font
        assert typography["font_family"] == "Times, serif"
        assert typography["font_size"] == "11px"
        
    def test_minimal_theme(self):
        """Test Minimal theme properties."""
        theme = MinimalTheme()
        
        colors = theme.get_colors()
        styles = theme.get_styles()
        typography = theme.get_typography()
        
        # Test minimal color scheme
        assert colors["background"] == "#ffffff"
        assert colors["layer_fill"] == "#ffffff"
        assert colors["layer_stroke"] == "#cccccc"
        assert colors["connection"] == "#cccccc"
        
        # Test minimal styles
        assert styles["border_radius"] == 2
        assert styles["stroke_width"] == 1
        assert styles["padding"] == 20
        
        # Test minimal typography
        assert typography["font_weight"] == "300"
        assert typography["font_size"] == "10px"
        
    def test_dark_theme(self):
        """Test Dark theme properties."""
        theme = DarkTheme()
        
        colors = theme.get_colors()
        styles = theme.get_styles()
        typography = theme.get_typography()
        
        # Test dark color scheme
        assert colors["background"] == "#2b2b2b"
        assert colors["layer_fill"] == "#404040"
        assert colors["text"] == "#ffffff"
        assert colors["connection"] == "#ffffff"
        
        # Test dark styles
        assert styles["connection_width"] == 2
        assert styles["stroke_width"] == 2
        
        # Test dark typography
        assert typography["font_family"] == "Arial, sans-serif"
        assert typography["font_size"] == "12px"
        
    def test_theme_interface(self):
        """Test that all themes implement the Theme interface."""
        themes = [IEEETheme(), APJTheme(), MinimalTheme(), DarkTheme()]
        
        for theme in themes:
            assert isinstance(theme, Theme)
            
            # Test required methods exist and return correct types
            colors = theme.get_colors()
            assert isinstance(colors, dict)
            assert "background" in colors
            assert "text" in colors
            
            styles = theme.get_styles()
            assert isinstance(styles, dict)
            assert "layer_width" in styles
            assert "layer_height" in styles
            
            typography = theme.get_typography()
            assert isinstance(typography, dict)
            assert "font_family" in typography
            assert "font_size" in typography
            
    def test_color_values(self):
        """Test that color values are valid hex colors."""
        themes = [IEEETheme(), APJTheme(), MinimalTheme(), DarkTheme()]
        
        for theme in themes:
            colors = theme.get_colors()
            for color_name, color_value in colors.items():
                # Should be hex color starting with #
                assert isinstance(color_value, str)
                assert color_value.startswith("#")
                assert len(color_value) == 7  # #RRGGBB format
                
                # Should be valid hex
                hex_part = color_value[1:]
                assert all(c in "0123456789abcdefABCDEF" for c in hex_part)
                
    def test_style_values(self):
        """Test that style values are valid numeric types."""
        themes = [IEEETheme(), APJTheme(), MinimalTheme(), DarkTheme()]
        
        numeric_style_keys = [
            "layer_width", "layer_height", "layer_spacing_x", "layer_spacing_y",
            "border_radius", "stroke_width", "connection_width", "padding"
        ]
        
        for theme in themes:
            styles = theme.get_styles()
            for key in numeric_style_keys:
                if key in styles:
                    value = styles[key]
                    assert isinstance(value, (int, float))
                    assert value > 0  # Should be positive
                    
    def test_typography_values(self):
        """Test that typography values are valid strings."""
        themes = [IEEETheme(), APJTheme(), MinimalTheme(), DarkTheme()]
        
        for theme in themes:
            typography = theme.get_typography()
            
            # Font family should be string
            assert isinstance(typography["font_family"], str)
            assert len(typography["font_family"]) > 0
            
            # Font size should be string with units
            font_size = typography["font_size"]
            assert isinstance(font_size, str)
            assert font_size.endswith("px") or font_size.endswith("pt")
            
            # Font weight should be valid
            font_weight = typography["font_weight"]
            assert isinstance(font_weight, str)
            assert font_weight in ["normal", "bold", "300", "400", "500", "600", "700"]
            
            # Text anchor should be valid SVG value
            text_anchor = typography["text_anchor"]
            assert text_anchor in ["start", "middle", "end"]