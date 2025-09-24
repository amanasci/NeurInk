# NeurInk - Neural Network Diagram Generator

NeurInk is a web-based platform for generating publication-quality neural network diagrams with a focus on customizability, precision, and exportability.

![NeurInk Screenshot](https://github.com/user-attachments/assets/5fc37f80-2970-49d1-9b1c-c955637663e9)

## Features

### ✨ Core Features
- **Visual Editor**: Drag-and-drop interface for building neural network architectures
- **Markup Language**: Custom DSL for describing neural networks in text format
- **Split View**: Synchronized markup editor and visual canvas
- **SVG Export**: High-quality vector graphics suitable for publications
- **Multiple Themes**: Light, dark, and monochrome themes for different use cases

### 🎨 Diagram Quality
- Clean, minimalistic design suitable for academic papers
- Publication-quality SVG rendering
- Customizable colors, fonts, and spacing
- Professional arrowheads and connections
- Anti-aliased rendering for crisp visuals

### 🧠 Neural Network Support
- **Layer Types**: Input, Conv2D, MaxPool, Dense, Dropout, Flatten, LSTM, Attention, Output
- **Customizable Parameters**: Filters, kernel sizes, activations, units, etc.
- **Connection Types**: Straight lines, Bezier curves, step connections
- **Grouping**: Support for modular architectures (encoder/decoder blocks)

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/amanasci/NeurInk.git
   cd NeurInk
   ```

2. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install --legacy-peer-deps
   ```

3. **Install backend dependencies**
   ```bash
   cd ../backend
   npm install
   ```

### Running the Application

1. **Start the backend server** (optional - for advanced parsing)
   ```bash
   cd backend
   npm run dev
   ```

2. **Start the frontend development server**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open your browser** to `http://localhost:5173`

## Usage

### Using the Visual Editor

1. **Add Layers**: Click on layer types in the sidebar to add them to your diagram
2. **Edit Properties**: Select a layer to edit its parameters in the sidebar
3. **Drag & Drop**: Move layers around by dragging them on the canvas
4. **Export**: Click "Export SVG" to download your diagram

### Using the Markup Language

![Split View Screenshot](https://github.com/user-attachments/assets/0016b1de-bf48-4751-8f53-e8d93f2b34ff)

1. **Toggle Split View**: Click the "Split View" button in the header
2. **Edit Markup**: Write your network description in the left pane
3. **Real-time Preview**: See the visual representation update in real-time
4. **Export**: Export the generated diagram as SVG

## Markup Language Reference

### Basic Syntax

```
input size=224x224x3
conv filters=64 kernel=3x3 stride=1 activation=relu
pool kernel=2x2 stride=2
conv filters=128 kernel=3x3 stride=1 activation=relu
pool kernel=2x2 stride=2
flatten
dense units=128 activation=relu
dropout rate=0.5
output units=10 activation=softmax
```

### Supported Layer Types

| Layer Type | Parameters | Example |
|------------|------------|---------|
| `input` | `size` | `input size=224x224x3` |
| `conv` | `filters`, `kernel`, `stride`, `activation` | `conv filters=64 kernel=3x3 activation=relu` |
| `pool` | `kernel`, `stride` | `pool kernel=2x2 stride=2` |
| `dense` | `units`, `activation` | `dense units=128 activation=relu` |
| `dropout` | `rate` | `dropout rate=0.5` |
| `flatten` | - | `flatten` |
| `lstm` | `units`, `return_sequences` | `lstm units=128` |
| `attention` | `heads`, `dim` | `attention heads=8 dim=64` |
| `output` | `units`, `activation` | `output units=10 activation=softmax` |

### Advanced Features (Future)

```
# Hierarchical grouping
encoder {
  conv filters=32 kernel=3x3
  conv filters=64 kernel=3x3
  pool kernel=2x2
}

decoder {
  dense units=128
  dense units=64
  output units=10
}

# Comments
// This is a comment
conv filters=64 kernel=3x3  // Inline comment
```

## Themes

NeurInk supports multiple visual themes:

- **Light Theme**: Clean white background, perfect for presentations
- **Dark Theme**: Dark background for reduced eye strain during development
- **Monochrome**: Black and white theme suitable for publications with strict color requirements

![Dark Theme](https://github.com/user-attachments/assets/861415ab-9b17-4816-9f92-f962d87007a6)

## Export Options

### SVG Export
- Vector format perfect for publications
- Scalable to any size without quality loss
- Compatible with LaTeX, Word, PowerPoint
- Transparent background support

### Future Export Options
- PNG: Raster format for web use
- PDF: Direct PDF export for publications
- LaTeX: Direct LaTeX code generation

## Integration with LaTeX

The exported SVG files can be easily integrated into LaTeX documents:

```latex
\documentclass{article}
\usepackage{graphicx}
\usepackage{svg}

\begin{document}
\begin{figure}[h]
    \centering
    \includesvg{neural-network-diagram.svg}
    \caption{Neural Network Architecture}
    \label{fig:network}
\end{figure}
\end{document}
```

## Development

### Project Structure

```
NeurInk/
├── frontend/          # React frontend application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── context/       # React context for state management
│   │   ├── utils/         # Utility functions and parsers
│   │   └── types/         # TypeScript type definitions
├── backend/           # Node.js backend (optional)
│   └── src/
│       └── server.js      # Express server for advanced parsing
├── docs/              # Documentation
└── PRD/               # Product Requirements Document
```

### Technology Stack

**Frontend:**
- React 19 with TypeScript
- TailwindCSS for styling
- SVG for diagram rendering
- Custom markup parser

**Backend (Optional):**
- Node.js with Express
- CORS for cross-origin requests
- PEG.js for advanced parsing

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Roadmap

- [ ] Bezier curve connections
- [ ] Advanced auto-layout algorithms
- [ ] Template library (ResNet, UNet, Vision Transformer)
- [ ] PNG/PDF export options
- [ ] LaTeX integration
- [ ] Collaboration features
- [ ] Version history
- [ ] Command palette
- [ ] Math expression support with MathJax

## Support

For questions, feature requests, or bug reports, please open an issue on GitHub.

---

**Built with ❤️ for the research community**