# LaTeX Integration Guide

This guide explains how to integrate NeurInk-generated diagrams into LaTeX documents for academic papers, theses, and presentations.

## Prerequisites

Ensure your LaTeX installation includes:
- `graphicx` package for image inclusion
- `svg` package for SVG support (recommended)
- `inkscape` for SVG to PDF conversion (if using svg package)

## Basic Integration

### Method 1: Using the svg Package (Recommended)

The `svg` package provides the best quality and automatic scaling:

```latex
\documentclass{article}
\usepackage{svg}
\usepackage{float}  % For [H] positioning

\begin{document}

\begin{figure}[H]
    \centering
    \includesvg[width=0.8\textwidth]{figures/neural-network}
    \caption{Convolutional Neural Network Architecture}
    \label{fig:cnn_architecture}
\end{figure}

\end{document}
```

**Advantages:**
- Perfect vector quality at any scale
- Automatic PDF conversion
- Preserves text searchability
- Smallest file sizes

### Method 2: Convert SVG to PDF

If the `svg` package is unavailable, convert SVG to PDF first:

```bash
# Using Inkscape (recommended)
inkscape neural-network.svg --export-pdf=neural-network.pdf

# Using cairosvg (alternative)
cairosvg neural-network.svg -o neural-network.pdf
```

Then include in LaTeX:

```latex
\documentclass{article}
\usepackage{graphicx}
\usepackage{float}

\begin{document}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/neural-network.pdf}
    \caption{Convolutional Neural Network Architecture}
    \label{fig:cnn_architecture}
\end{figure}

\end{document}
```

### Method 3: Convert SVG to PNG (Not Recommended)

For compatibility with older systems:

```bash
# Using Inkscape
inkscape neural-network.svg --export-png=neural-network.png --export-dpi=300

# Using cairosvg
cairosvg neural-network.svg -o neural-network.png --dpi=300
```

```latex
\documentclass{article}
\usepackage{graphicx}
\usepackage{float}

\begin{document}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/neural-network.png}
    \caption{Convolutional Neural Network Architecture}
    \label{fig:cnn_architecture}
\end{figure}

\end{document}
```

## Advanced Configuration

### Custom Figure Sizing

Control exact dimensions:

```latex
% Fixed width, maintain aspect ratio
\includesvg[width=12cm]{figures/neural-network}

% Fixed height, maintain aspect ratio
\includesvg[height=8cm]{figures/neural-network}

% Specific width and height
\includesvg[width=12cm,height=8cm]{figures/neural-network}

% Scale relative to text width
\includesvg[width=0.9\textwidth]{figures/neural-network}

% Scale relative to column width (for two-column documents)
\includesvg[width=\columnwidth]{figures/neural-network}
```

### Multiple Subfigures

Display multiple architectures side by side:

```latex
\documentclass{article}
\usepackage{svg}
\usepackage{subcaption}
\usepackage{float}

\begin{document}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \includesvg[width=\textwidth]{figures/simple-cnn}
        \caption{Simple CNN}
        \label{fig:simple_cnn}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \includesvg[width=\textwidth]{figures/resnet-block}
        \caption{ResNet Block}
        \label{fig:resnet_block}
    \end{subfigure}
    \caption{Neural Network Architectures}
    \label{fig:architectures}
\end{figure}

\end{document}
```

### Rotating Large Diagrams

For tall architectures, use landscape orientation:

```latex
\documentclass{article}
\usepackage{svg}
\usepackage{rotating}
\usepackage{float}

\begin{document}

\begin{sidewaysfigure}
    \centering
    \includesvg[height=0.8\textheight]{figures/tall-network}
    \caption{Deep Neural Network Architecture}
    \label{fig:deep_network}
\end{sidewaysfigure}

\end{document}
```

## Best Practices for Academic Papers

### File Organization

Organize your files for easy management:

```
paper/
├── main.tex
├── figures/
│   ├── neural-networks/
│   │   ├── cnn-architecture.svg
│   │   ├── transformer-block.svg
│   │   └── attention-mechanism.svg
│   └── results/
│       ├── accuracy-plot.pdf
│       └── loss-curves.pdf
├── sections/
│   ├── introduction.tex
│   ├── methodology.tex
│   └── results.tex
└── bibliography.bib
```

### Naming Conventions

Use descriptive, consistent names:

```latex
% Good naming
\label{fig:vgg16_architecture}
\label{fig:attention_mechanism}
\label{fig:encoder_decoder}

% Poor naming
\label{fig:network1}
\label{fig:diagram}
\label{fig:fig1}
```

### Caption Writing

Write informative captions:

```latex
\caption{Convolutional Neural Network architecture for image classification. 
The network consists of three convolutional blocks (Conv-BatchNorm-ReLU-MaxPool) 
followed by two fully connected layers. Input images are $224 \times 224 \times 3$ 
RGB images, and the output is a 1000-dimensional probability distribution over ImageNet classes.}
\label{fig:cnn_classification}
```

### Cross-Referencing

Reference figures in text:

```latex
The proposed architecture, shown in Figure~\ref{fig:cnn_classification}, 
achieves state-of-the-art performance on the ImageNet dataset.

% For multiple figures
Figures~\ref{fig:encoder} and~\ref{fig:decoder} show the encoder and decoder 
architectures respectively.

% For subfigures
The attention mechanism (Figure~\ref{fig:attention_mechanism}) consists of 
query, key, and value projections as illustrated in Figure~\ref{fig:qkv_computation}.
```

## Journal-Specific Guidelines

### IEEE Journals

```latex
\documentclass[journal]{IEEEtran}
\usepackage{svg}
\usepackage{cite}

% IEEE prefers figures at column width
\begin{figure}[!t]
    \centering
    \includesvg[width=\columnwidth]{figures/network-architecture}
    \caption{Proposed neural network architecture.}
    \label{fig:proposed_architecture}
\end{figure}
```

### ACM Journals

```latex
\documentclass[acmsmall]{acmart}
\usepackage{svg}

% ACM allows flexible sizing
\begin{figure}[h]
    \centering
    \includesvg[width=0.8\textwidth]{figures/system-overview}
    \caption{System architecture overview showing the data flow through 
    the neural network pipeline.}
    \label{fig:system_overview}
\end{figure}
```

### Springer Journals

```latex
\documentclass{svjour3}
\usepackage{svg}
\usepackage{float}

% Springer often prefers single-column figures
\begin{figure}[H]
    \centering
    \includesvg[width=0.9\textwidth]{figures/methodology}
    \caption{Methodology flowchart illustrating the proposed approach.}
    \label{fig:methodology}
\end{figure}
```

## Conference Templates

### NeurIPS

```latex
\documentclass{neurips_2023}
\usepackage{svg}

\begin{figure}[tb]
    \centering
    \includesvg[width=\textwidth]{figures/model-architecture}
    \caption{Model architecture. The network processes input sequences 
    through multiple transformer blocks before generating predictions.}
    \label{fig:model}
\end{figure}
```

### ICML

```latex
\documentclass[twoside]{article}
\usepackage{icml2023}
\usepackage{svg}

\begin{figure}[htbp]
    \centering
    \includesvg[width=0.85\textwidth]{figures/training-pipeline}
    \caption{Training pipeline architecture.}
    \label{fig:training}
\end{figure}
```

## Troubleshooting

### Common Issues and Solutions

**Issue: SVG not displaying**
```latex
% Solution: Ensure svg package and inkscape are installed
\usepackage{svg}
\svgsetup{inkscape=forced}  % Force inkscape usage
```

**Issue: Figure too large/small**
```latex
% Solution: Use relative sizing
\includesvg[width=0.8\textwidth]{figure}  % Instead of fixed widths
```

**Issue: Text in SVG too small**
- Use monochrome theme in NeurInk for better text visibility
- Increase font sizes when creating diagrams
- Scale figures appropriately in LaTeX

**Issue: Colors not reproducing correctly**
```latex
% Solution: Use monochrome theme for publications
% Or specify color space in your document class
\PassOptionsToPackage{cmyk}{xcolor}
\documentclass{article}
```

**Issue: Compilation errors with svg package**
```bash
# Ensure inkscape is in PATH
which inkscape

# Install inkscape if missing
# Ubuntu/Debian: sudo apt install inkscape
# macOS: brew install inkscape
# Windows: Download from inkscape.org
```

### Build Scripts

Automate figure conversion:

```bash
#!/bin/bash
# convert-figures.sh

# Create output directory
mkdir -p figures/converted

# Convert all SVG files to PDF
for svg in figures/*.svg; do
    base=$(basename "$svg" .svg)
    inkscape "$svg" --export-pdf="figures/converted/${base}.pdf"
    echo "Converted $svg to PDF"
done
```

```makefile
# Makefile for LaTeX with SVG conversion
FIGURES_SVG = $(wildcard figures/*.svg)
FIGURES_PDF = $(FIGURES_SVG:figures/%.svg=figures/converted/%.pdf)

paper.pdf: main.tex $(FIGURES_PDF)
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

figures/converted/%.pdf: figures/%.svg
	@mkdir -p figures/converted
	inkscape $< --export-pdf=$@

clean:
	rm -f *.aux *.bbl *.blg *.log *.out
	rm -rf figures/converted

.PHONY: clean
```

## Performance Tips

1. **Use vector formats**: SVG → PDF → LaTeX gives best quality
2. **Optimize SVG files**: Remove unnecessary metadata before inclusion
3. **Cache conversions**: Don't reconvert unchanged files
4. **Use appropriate DPI**: 300 DPI minimum for raster formats
5. **Test early**: Check figure quality in your target format early in the writing process

## Example Document

Complete example with multiple figures:

```latex
\documentclass[twocolumn]{article}
\usepackage{svg}
\usepackage{float}
\usepackage{subcaption}

\title{Neural Network Architecture Design}
\author{Your Name}

\begin{document}
\maketitle

\section{Introduction}

Our proposed architecture, shown in Figure~\ref{fig:overview}, 
combines convolutional and attention mechanisms for improved performance.

\begin{figure}[H]
    \centering
    \includesvg[width=\columnwidth]{figures/architecture-overview}
    \caption{Overview of the proposed hybrid CNN-Transformer architecture.}
    \label{fig:overview}
\end{figure}

\section{Methodology}

The detailed components are illustrated in Figure~\ref{fig:components}.

\begin{figure*}[t]
    \centering
    \begin{subfigure}[b]{0.3\textwidth}
        \includesvg[width=\textwidth]{figures/cnn-block}
        \caption{CNN Block}
        \label{fig:cnn_block}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.3\textwidth}
        \includesvg[width=\textwidth]{figures/attention-block}
        \caption{Attention Block}
        \label{fig:attention_block}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.3\textwidth}
        \includesvg[width=\textwidth]{figures/fusion-module}
        \caption{Fusion Module}
        \label{fig:fusion}
    \end{subfigure}
    \caption{Architecture components: (a) CNN block for local feature extraction, 
    (b) attention block for global context modeling, and (c) fusion module for 
    combining features.}
    \label{fig:components}
\end{figure*}

\end{document}
```

This guide should help you successfully integrate NeurInk diagrams into your LaTeX documents for professional academic publications.