import type { LayerNode, Connection } from '../context/DiagramContext'

const layerColors: Record<string, string> = {
  input: '#3b82f6',
  conv: '#10b981',
  pool: '#f59e0b',
  dense: '#8b5cf6',
  dropout: '#ef4444',
  flatten: '#6b7280',
  lstm: '#ec4899',
  attention: '#14b8a6',
  output: '#f97316',
}

export function parseMarkup(markup: string): { nodes: LayerNode[]; connections: Connection[] } {
  const lines = markup.split('\n').filter(line => line.trim() !== '')
  const nodes: LayerNode[] = []
  const connections: Connection[] = []
  
  let yPosition = 50
  const xPosition = 200
  const layerSpacing = 100

  lines.forEach((line, index) => {
    const trimmed = line.trim()
    if (trimmed === '' || trimmed.startsWith('//')) return

    const parts = trimmed.split(' ')
    const layerType = parts[0].toLowerCase()
    
    // Parse parameters
    const parameters: Record<string, any> = {}
    
    for (let i = 1; i < parts.length; i++) {
      const part = parts[i]
      if (part.includes('=')) {
        const [key, value] = part.split('=')
        // Try to parse as number, otherwise keep as string
        const numValue = parseFloat(value)
        parameters[key] = isNaN(numValue) ? value : numValue
      }
    }

    // Create node
    const node: LayerNode = {
      id: `${layerType}-${index}`,
      type: layerType,
      label: formatLabel(layerType, parameters),
      position: { x: xPosition, y: yPosition },
      parameters,
      style: {
        color: layerColors[layerType] || '#6b7280',
        size: { width: 150, height: 60 }
      }
    }

    nodes.push(node)

    // Create connection to previous node
    if (index > 0) {
      const connection: Connection = {
        id: `conn-${index}`,
        source: nodes[index - 1].id,
        target: node.id,
        type: 'straight'
      }
      connections.push(connection)
    }

    yPosition += layerSpacing
  })

  return { nodes, connections }
}

function formatLabel(type: string, parameters: Record<string, any>): string {
  const baseLabel = type.charAt(0).toUpperCase() + type.slice(1)
  
  switch (type) {
    case 'input':
      return `Input ${parameters.size || ''}`
    case 'conv':
      return `Conv2D (${parameters.filters || '?'})`
    case 'pool':
      return `MaxPool ${parameters.kernel || ''}`
    case 'dense':
      return `Dense (${parameters.units || '?'})`
    case 'dropout':
      return `Dropout (${parameters.rate || '?'})`
    case 'lstm':
      return `LSTM (${parameters.units || '?'})`
    case 'attention':
      return `Attention (${parameters.heads || '?'}h)`
    case 'output':
      return `Output (${parameters.units || '?'})`
    default:
      return baseLabel
  }
}

export function generateMarkup(nodes: LayerNode[]): string {
  return nodes.map(node => {
    const params = Object.entries(node.parameters)
      .map(([key, value]) => `${key}=${value}`)
      .join(' ')
    
    return `${node.type.toLowerCase()}${params ? ` ${params}` : ''}`
  }).join('\n')
}