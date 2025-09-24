import React from 'react'
import { useDiagram } from '../context/DiagramContext'
import { parseMarkup } from '../utils/markupParser'

const MarkupEditor: React.FC = () => {
  const { state, dispatch } = useDiagram()

  const handleMarkupChange = (value: string) => {
    dispatch({ type: 'UPDATE_MARKUP', payload: value })
    
    // Parse the markup and update the diagram
    try {
      const { nodes, connections } = parseMarkup(value)
      dispatch({ type: 'LOAD_FROM_MARKUP', payload: { nodes, connections } })
    } catch (error) {
      console.error('Markup parsing error:', error)
    }
  }

  return (
    <div className="h-full flex flex-col bg-gray-900 text-white">
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold">Markup Editor</h3>
        <p className="text-sm text-gray-400 mt-1">
          Define your neural network architecture using our markup language
        </p>
      </div>

      <div className="flex-1 relative">
        <textarea
          value={state.markupText}
          onChange={(e) => handleMarkupChange(e.target.value)}
          className="w-full h-full p-4 bg-gray-900 text-gray-100 font-mono text-sm resize-none focus:outline-none"
          placeholder="Enter your neural network markup here..."
          spellCheck={false}
        />
        
        {/* Line Numbers */}
        <div className="absolute left-0 top-0 p-4 text-gray-500 font-mono text-sm pointer-events-none select-none">
          {state.markupText.split('\n').map((_, index) => (
            <div key={index} className="leading-5">
              {index + 1}
            </div>
          ))}
        </div>
      </div>

      <div className="p-4 border-t border-gray-700">
        <div className="text-xs text-gray-400">
          <div className="mb-2">
            <strong>Syntax Examples:</strong>
          </div>
          <div className="space-y-1 font-mono">
            <div>input size=224x224x3</div>
            <div>conv filters=64 kernel=3x3 activation=relu</div>
            <div>pool kernel=2x2 stride=2</div>
            <div>dense units=128 activation=relu</div>
            <div>output units=10 activation=softmax</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default MarkupEditor