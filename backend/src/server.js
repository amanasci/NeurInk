import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'

dotenv.config()

const app = express()
const PORT = process.env.PORT || 3001

// Middleware
app.use(cors())
app.use(express.json())

// Routes
app.get('/', (req, res) => {
  res.json({ 
    message: 'NeurInk Backend API',
    version: '1.0.0',
    endpoints: {
      'GET /': 'API information',
      'POST /parse': 'Parse neural network markup',
      'GET /health': 'Health check'
    }
  })
})

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() })
})

app.post('/parse', (req, res) => {
  try {
    const { markup } = req.body
    
    if (!markup) {
      return res.status(400).json({ error: 'Markup is required' })
    }

    // Simple markup parser (could be enhanced with PEG.js later)
    const lines = markup.split('\n').filter(line => line.trim() !== '')
    const nodes = []
    const connections = []
    
    let yPosition = 50
    const xPosition = 200
    const layerSpacing = 100

    lines.forEach((line, index) => {
      const trimmed = line.trim()
      if (trimmed === '' || trimmed.startsWith('//')) return

      const parts = trimmed.split(' ')
      const layerType = parts[0].toLowerCase()
      
      // Parse parameters
      const parameters = {}
      
      for (let i = 1; i < parts.length; i++) {
        const part = parts[i]
        if (part.includes('=')) {
          const [key, value] = part.split('=')
          const numValue = parseFloat(value)
          parameters[key] = isNaN(numValue) ? value : numValue
        }
      }

      // Create node
      const node = {
        id: `${layerType}-${index}`,
        type: layerType,
        label: formatLabel(layerType, parameters),
        position: { x: xPosition, y: yPosition },
        parameters,
        style: {
          color: getLayerColor(layerType),
          size: { width: 150, height: 60 }
        }
      }

      nodes.push(node)

      // Create connection to previous node
      if (index > 0) {
        const connection = {
          id: `conn-${index}`,
          source: nodes[index - 1].id,
          target: node.id,
          type: 'straight'
        }
        connections.push(connection)
      }

      yPosition += layerSpacing
    })

    res.json({ nodes, connections })
  } catch (error) {
    console.error('Parsing error:', error)
    res.status(500).json({ error: 'Failed to parse markup' })
  }
})

function formatLabel(type, parameters) {
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

function getLayerColor(type) {
  const colors = {
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
  return colors[type] || '#6b7280'
}

app.listen(PORT, () => {
  console.log(`NeurInk Backend running on port ${PORT}`)
  console.log(`Health check: http://localhost:${PORT}/health`)
})