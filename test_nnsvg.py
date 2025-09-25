#!/usr/bin/env python3
"""
Test script for the new NN-SVG theme.
"""

from neurink import Diagram

def test_nnsvg_theme():
    """Test the new NN-SVG theme."""
    print("Testing NN-SVG theme...")
    
    # Create a simple network
    diagram = (Diagram()
               .input((64, 64))
               .conv(32, 3)
               .conv(64, 3, stride=2)
               .flatten()
               .dense(128)
               .dropout(0.5)
               .output(10))
    
    # Render with NNSVG theme
    diagram.render("test_nnsvg.svg", theme="nnsvg")
    print("Rendered test_nnsvg.svg")
    
    # Compare with other themes
    diagram.render("test_ieee.svg", theme="ieee")
    diagram.render("test_minimal.svg", theme="minimal")
    
    print("Created comparison files: test_nnsvg.svg, test_ieee.svg, test_minimal.svg")

if __name__ == "__main__":
    test_nnsvg_theme()