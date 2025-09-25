"""
NeurInk: Publication-quality neural network diagram generation library.

A research-grade library for creating beautiful, customizable neural network
diagrams with Python API and DSL support.
"""

__version__ = "0.1.0"
__author__ = "NeurInk Contributors"

from .diagram import Diagram
from .layer import Layer, InputLayer, ConvLayer, DenseLayer, FlattenLayer, DropoutLayer, OutputLayer, AttentionLayer, LayerNormLayer, EmbeddingLayer, PoolingLayer, BatchNormLayer, SkipConnectionLayer, BranchLayer, MergeLayer
from .renderer import SVGRenderer
from .themes import Theme, IEEETheme, APJTheme, MinimalTheme, DarkTheme, NNSVGTheme
from .parser import DSLParser, DSLParseError
from .templates import ResNetTemplate, UNetTemplate, TransformerTemplate, MLPTemplate

__all__ = [
    "Diagram",
    "Layer",
    "InputLayer",
    "ConvLayer", 
    "DenseLayer",
    "FlattenLayer",
    "DropoutLayer",
    "OutputLayer",
    "AttentionLayer",
    "LayerNormLayer", 
    "EmbeddingLayer",
    "PoolingLayer",
    "BatchNormLayer",
    "SkipConnectionLayer",
    "BranchLayer", 
    "MergeLayer",
    "SVGRenderer",
    "Theme",
    "IEEETheme",
    "APJTheme",
    "MinimalTheme",
    "DarkTheme",
    "NNSVGTheme",
    "DSLParser",
    "DSLParseError",
    "ResNetTemplate",
    "UNetTemplate", 
    "TransformerTemplate",
    "MLPTemplate",
]