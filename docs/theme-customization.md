# Theme Customization Guide

NeurInk provides multiple visual themes to suit different use cases and preferences. This guide explains how to use and customize themes.

## Built-in Themes

### Light Theme (Default)
- **Background**: Light gray (`#f9fafb`)
- **Nodes**: White background with colored borders
- **Text**: Dark gray (`#1f2937`)
- **Connections**: Medium gray (`#6b7280`)
- **Use Case**: Presentations, web displays, general use

### Dark Theme
- **Background**: Dark gray (`#1f2937`)
- **Nodes**: Semi-transparent with colored borders
- **Text**: Light gray (`#f9fafb`)
- **Connections**: Light gray (`#6b7280`)
- **Use Case**: Reduced eye strain, dark environments, night work

### Monochrome Theme
- **Background**: Pure white (`#ffffff`)
- **Nodes**: White with black borders
- **Text**: Pure black (`#000000`)
- **Connections**: Black (`#000000`)
- **Use Case**: Publications with strict color requirements, printing

## Switching Themes

### Using the Interface
1. Click the theme dropdown in the header (palette icon)
2. Select your desired theme:
   - Light
   - Dark
   - Monochrome

The diagram will update immediately to reflect the new theme.

### Programmatic Theme Changes

If you're extending NeurInk, you can change themes programmatically:

```javascript
import { useDiagram } from '../context/DiagramContext'

function MyComponent() {
  const { dispatch } = useDiagram()
  
  const setTheme = (themeName) => {
    dispatch({ type: 'SET_THEME', payload: themeName })
  }
  
  return (
    <button onClick={() => setTheme('dark')}>
      Switch to Dark Theme
    </button>
  )
}
```

## Layer Colors by Theme

### Light Theme Colors
- **Input**: Blue (`#3b82f6`)
- **Conv2D**: Green (`#10b981`)
- **MaxPool**: Amber (`#f59e0b`)
- **Dense**: Purple (`#8b5cf6`)
- **Dropout**: Red (`#ef4444`)
- **Flatten**: Gray (`#6b7280`)
- **LSTM**: Pink (`#ec4899`)
- **Attention**: Teal (`#14b8a6`)
- **Output**: Orange (`#f97316`)

### Dark Theme Colors
Same colors as light theme but with:
- Reduced opacity (0.8) for better contrast
- Lighter stroke weights
- Enhanced glow effects

### Monochrome Colors
- All layers use black (`#000000`) borders
- Different line styles (solid, dashed, dotted) to distinguish layer types
- Varying stroke weights for hierarchy

## Customizing Themes

### Method 1: CSS Custom Properties

You can override theme colors using CSS custom properties:

```css
/* Add to your custom CSS file */
.neural-diagram-light {
  --node-input-color: #1e40af;     /* Custom blue for input layers */
  --node-conv-color: #059669;      /* Custom green for conv layers */
  --connection-color: #374151;     /* Custom connection color */
  --background-color: #f8fafc;     /* Custom background */
}

.neural-diagram-dark {
  --node-input-color: #3b82f6;
  --node-conv-color: #10b981;
  --connection-color: #9ca3af;
  --background-color: #111827;
}
```

### Method 2: Extending the Theme Context

For more advanced customization, extend the theme system:

```javascript
// Create a custom theme configuration
const customTheme = {
  name: 'custom',
  colors: {
    background: '#f0f9ff',
    nodeStroke: '#0c4a6e',
    nodeText: '#0c4a6e',
    connectionStroke: '#0369a1',
    layers: {
      input: '#0ea5e9',
      conv: '#06b6d4',
      pool: '#0891b2',
      dense: '#0e7490',
      dropout: '#dc2626',
      flatten: '#64748b',
      lstm: '#7c3aed',
      attention: '#059669',
      output: '#ea580c'
    }
  }
}

// Add to your DiagramContext
const themeColors = getThemeColors(state.theme, customTheme)
```

### Method 3: Dynamic Theme Generation

Create themes based on user preferences:

```javascript
function generateCustomTheme(baseHue, saturation = 70, lightness = 50) {
  const hsl = (h, s, l) => `hsl(${h}, ${s}%, ${l}%)`
  
  return {
    name: 'generated',
    colors: {
      background: hsl(baseHue, 10, 97),
      layers: {
        input: hsl(baseHue, saturation, lightness),
        conv: hsl((baseHue + 60) % 360, saturation, lightness),
        pool: hsl((baseHue + 120) % 360, saturation, lightness),
        dense: hsl((baseHue + 180) % 360, saturation, lightness),
        dropout: hsl((baseHue + 240) % 360, saturation, lightness),
        flatten: hsl((baseHue + 300) % 360, saturation, lightness),
        // ... more layers
      }
    }
  }
}
```

## Export Considerations

### SVG Export with Themes
When exporting SVG files, the current theme is preserved:

- **Light theme exports**: Best for presentations and web use
- **Dark theme exports**: May not display well in all applications
- **Monochrome exports**: Perfect for academic publications

### Publication Guidelines

#### For Academic Papers
1. Use **Monochrome theme** for journals with strict color policies
2. Use **Light theme** with high contrast for color publications
3. Ensure exported SVGs have transparent backgrounds
4. Test compatibility with your target publication format

#### For Presentations
1. **Light theme**: Good for bright rooms, projectors
2. **Dark theme**: Better for dark rooms, LED screens
3. Consider audience and viewing conditions

## Theme-Specific Features

### Light Theme Features
- Grid lines for precise alignment
- Drop shadows for depth perception
- Subtle gradients on hover states
- High contrast for accessibility

### Dark Theme Features
- Reduced eye strain in low-light conditions
- Subtle glow effects on nodes
- Muted colors to prevent visual fatigue
- Enhanced focus on selected elements

### Monochrome Features
- Maximum compatibility with printing
- Clear visual hierarchy through line weights
- Pattern-based differentiation for accessibility
- Optimal for grayscale reproduction

## Accessibility Considerations

### Color Blindness Support
- All themes use sufficient contrast ratios
- Monochrome theme eliminates color dependency
- Shape and pattern differentiation available
- Text labels always visible

### High Contrast Mode
Enable high contrast mode for better accessibility:

```css
@media (prefers-contrast: high) {
  .neural-diagram {
    --node-stroke-width: 3px;
    --connection-stroke-width: 3px;
    --text-contrast: 1;
  }
}
```

### Reduced Motion
Respect user preferences for reduced motion:

```css
@media (prefers-reduced-motion: reduce) {
  .neural-diagram * {
    transition: none !important;
    animation: none !important;
  }
}
```

## Best Practices

### Choosing the Right Theme
- **Development**: Use dark theme to reduce eye strain
- **Presentations**: Use light theme for better visibility
- **Publications**: Use monochrome for maximum compatibility
- **Collaboration**: Consider your team's preferences

### Customization Guidelines
1. Maintain sufficient contrast ratios (WCAG 2.1 AA: 4.5:1)
2. Use semantic color meanings (red for dropout/loss, green for growth/success)
3. Test themes with different layer types
4. Ensure readability at different zoom levels
5. Validate exports in target applications

### Performance Considerations
- Avoid complex gradients in large diagrams
- Use CSS transforms instead of changing positions
- Limit the number of custom colors
- Test performance with 100+ nodes

## Troubleshooting

### Common Issues

**Theme not applying correctly:**
- Clear browser cache
- Check CSS custom properties
- Ensure theme name matches exactly

**Export colors different from display:**
- Some applications interpret SVG colors differently
- Test exports in target applications
- Use absolute color values instead of relative ones

**Poor performance with custom themes:**
- Simplify color calculations
- Cache theme computations
- Use CSS custom properties efficiently

**Accessibility issues:**
- Test with screen readers
- Verify contrast ratios
- Provide alternative text for all elements