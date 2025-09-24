import React from 'react'
import { Plus, Layers, Settings } from 'lucide-react'
import { useDiagram } from '../context/DiagramContext'

const layerTypes = [
  { type: 'input', label: 'Input', color: '#3b82f6', description: 'Input layer' },
  { type: 'conv', label: 'Conv2D', color: '#10b981', description: 'Convolutional layer' },
  { type: 'pool', label: 'MaxPool', color: '#f59e0b', description: 'Pooling layer' },
  { type: 'dense', label: 'Dense', color: '#8b5cf6', description: 'Fully connected layer' },
  { type: 'dropout', label: 'Dropout', color: '#ef4444', description: 'Dropout layer' },
  { type: 'flatten', label: 'Flatten', color: '#6b7280', description: 'Flatten layer' },
  { type: 'lstm', label: 'LSTM', color: '#ec4899', description: 'LSTM layer' },
  { type: 'attention', label: 'Attention', color: '#14b8a6', description: 'Attention layer' },
  { type: 'output', label: 'Output', color: '#f97316', description: 'Output layer' },
]

const Sidebar: React.FC = () => {
  const { state, dispatch } = useDiagram()

  const addLayer = (layerType: string) => {
    const newNode = {
      id: `${layerType}-${Date.now()}`,
      type: layerType,
      label: layerType.charAt(0).toUpperCase() + layerType.slice(1),
      position: { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 },
      parameters: getDefaultParameters(layerType),
      style: {
        color: layerTypes.find(l => l.type === layerType)?.color || '#6b7280',
        size: { width: 120, height: 60 }
      }
    }
    
    dispatch({ type: 'ADD_NODE', payload: newNode })
  }

  const getDefaultParameters = (type: string): Record<string, any> => {
    switch (type) {
      case 'input':
        return { size: '224x224x3' }
      case 'conv':
        return { filters: 64, kernel: '3x3', stride: 1, activation: 'relu' }
      case 'pool':
        return { kernel: '2x2', stride: 2 }
      case 'dense':
        return { units: 128, activation: 'relu' }
      case 'dropout':
        return { rate: 0.5 }
      case 'lstm':
        return { units: 128, return_sequences: false }
      case 'attention':
        return { heads: 8, dim: 64 }
      case 'output':
        return { units: 10, activation: 'softmax' }
      default:
        return {}
    }
  }

  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
      {/* Layer Palette */}
      <div className="p-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
          <Layers className="w-4 h-4 mr-2" />
          Layer Palette
        </h3>
        
        <div className="space-y-2">
          {layerTypes.map((layer) => (
            <button
              key={layer.type}
              onClick={() => addLayer(layer.type)}
              className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-all duration-200 group"
            >
              <div className="flex items-center space-x-3">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: layer.color }}
                />
                <div>
                  <div className="text-sm font-medium text-gray-800">
                    {layer.label}
                  </div>
                  <div className="text-xs text-gray-500">
                    {layer.description}
                  </div>
                </div>
                <Plus className="w-4 h-4 text-gray-400 group-hover:text-blue-500 ml-auto" />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Node Properties */}
      {state.selectedNode && (
        <div className="p-4 border-t border-gray-200">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
            <Settings className="w-4 h-4 mr-2" />
            Properties
          </h3>
          
          <NodeProperties />
        </div>
      )}

      {/* Diagram Info */}
      <div className="mt-auto p-4 border-t border-gray-200 text-xs text-gray-500">
        <div>Nodes: {state.nodes.length}</div>
        <div>Connections: {state.connections.length}</div>
      </div>
    </div>
  )
}

const NodeProperties: React.FC = () => {
  const { state, dispatch } = useDiagram()
  const selectedNode = state.nodes.find(node => node.id === state.selectedNode)

  if (!selectedNode) return null

  const updateNodeParameter = (key: string, value: any) => {
    dispatch({
      type: 'UPDATE_NODE',
      payload: {
        id: selectedNode.id,
        updates: {
          parameters: {
            ...selectedNode.parameters,
            [key]: value
          }
        }
      }
    })
  }

  return (
    <div className="space-y-3">
      <div>
        <label className="block text-xs font-medium text-gray-700 mb-1">
          Label
        </label>
        <input
          type="text"
          value={selectedNode.label}
          onChange={(e) => dispatch({
            type: 'UPDATE_NODE',
            payload: {
              id: selectedNode.id,
              updates: { label: e.target.value }
            }
          })}
          className="w-full text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>

      {Object.entries(selectedNode.parameters).map(([key, value]) => (
        <div key={key}>
          <label className="block text-xs font-medium text-gray-700 mb-1 capitalize">
            {key.replace('_', ' ')}
          </label>
          <input
            type={typeof value === 'number' ? 'number' : 'text'}
            value={value}
            onChange={(e) => updateNodeParameter(
              key,
              typeof value === 'number' ? parseFloat(e.target.value) : e.target.value
            )}
            className="w-full text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      ))}
    </div>
  )
}

export default Sidebar