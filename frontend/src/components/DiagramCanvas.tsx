import React, { useEffect, useRef, useState } from 'react'
import { useDiagram } from '../context/DiagramContext'
import type { LayerNode, Connection } from '../context/DiagramContext'
import { parseMarkup } from '../utils/markupParser'

const DiagramCanvas: React.FC = () => {
  const { state, dispatch } = useDiagram()
  const svgRef = useRef<SVGSVGElement>(null)
  const [draggedNode, setDraggedNode] = useState<string | null>(null)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })

  useEffect(() => {
    if (state.markupText && state.nodes.length === 0) {
      // Initial parse of markup
      try {
        const { nodes, connections } = parseMarkup(state.markupText)
        dispatch({ type: 'LOAD_FROM_MARKUP', payload: { nodes, connections } })
      } catch (error) {
        console.error('Initial markup parsing error:', error)
      }
    }
  }, [state.markupText, state.nodes.length, dispatch])

  const handleNodeMouseDown = (nodeId: string, event: React.MouseEvent) => {
    const node = state.nodes.find(n => n.id === nodeId)
    if (!node) return

    const svgRect = svgRef.current?.getBoundingClientRect()
    if (!svgRect) return

    setDraggedNode(nodeId)
    setDragOffset({
      x: event.clientX - svgRect.left - node.position.x,
      y: event.clientY - svgRect.top - node.position.y
    })
    
    dispatch({ type: 'SELECT_NODE', payload: nodeId })
  }

  const handleMouseMove = (event: React.MouseEvent) => {
    if (!draggedNode) return

    const svgRect = svgRef.current?.getBoundingClientRect()
    if (!svgRect) return

    const newX = event.clientX - svgRect.left - dragOffset.x
    const newY = event.clientY - svgRect.top - dragOffset.y

    dispatch({
      type: 'UPDATE_NODE',
      payload: {
        id: draggedNode,
        updates: {
          position: { x: Math.max(0, newX), y: Math.max(0, newY) }
        }
      }
    })
  }

  const handleMouseUp = () => {
    setDraggedNode(null)
    setDragOffset({ x: 0, y: 0 })
  }

  const getThemeColors = () => {
    switch (state.theme) {
      case 'dark':
        return {
          background: '#1f2937',
          nodeStroke: '#4b5563',
          nodeText: '#f9fafb',
          connectionStroke: '#6b7280'
        }
      case 'monochrome':
        return {
          background: '#ffffff',
          nodeStroke: '#000000',
          nodeText: '#000000',
          connectionStroke: '#000000'
        }
      default:
        return {
          background: '#f9fafb',
          nodeStroke: '#d1d5db',
          nodeText: '#1f2937',
          connectionStroke: '#6b7280'
        }
    }
  }

  const themeColors = getThemeColors()

  return (
    <div className="flex-1 relative overflow-hidden" id="diagram-canvas">
      <svg
        ref={svgRef}
        className="w-full h-full cursor-move"
        style={{ backgroundColor: themeColors.background }}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon
              points="0 0, 10 3.5, 0 7"
              fill={themeColors.connectionStroke}
            />
          </marker>
        </defs>

        {/* Grid */}
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path
            d="M 20 0 L 0 0 0 20"
            fill="none"
            stroke={state.theme === 'dark' ? '#374151' : '#f3f4f6'}
            strokeWidth="1"
          />
        </pattern>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* Connections */}
        {state.connections.map(connection => (
          <ConnectionLine
            key={connection.id}
            connection={connection}
            nodes={state.nodes}
            stroke={themeColors.connectionStroke}
          />
        ))}

        {/* Nodes */}
        {state.nodes.map(node => (
          <NodeComponent
            key={node.id}
            node={node}
            isSelected={state.selectedNode === node.id}
            onMouseDown={(e) => handleNodeMouseDown(node.id, e)}
            themeColors={themeColors}
          />
        ))}
      </svg>

      {state.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center text-gray-500">
            <div className="text-lg font-medium mb-2">Start Building Your Network</div>
            <div className="text-sm">
              Add layers from the sidebar or edit the markup to get started
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface NodeComponentProps {
  node: LayerNode
  isSelected: boolean
  onMouseDown: (event: React.MouseEvent) => void
  themeColors: any
}

const NodeComponent: React.FC<NodeComponentProps> = ({ 
  node, 
  isSelected, 
  onMouseDown, 
  themeColors 
}) => {
  const nodeColor = node.style.color
  
  return (
    <g
      transform={`translate(${node.position.x}, ${node.position.y})`}
      onMouseDown={onMouseDown}
      style={{ cursor: 'grab' }}
    >
      <rect
        width={node.style.size.width}
        height={node.style.size.height}
        rx="8"
        fill={nodeColor}
        fillOpacity="0.1"
        stroke={isSelected ? '#3b82f6' : nodeColor}
        strokeWidth={isSelected ? 3 : 2}
        className="transition-all duration-200"
      />
      
      <text
        x={node.style.size.width / 2}
        y={node.style.size.height / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fill={themeColors.nodeText}
        fontSize="12"
        fontWeight="500"
        className="pointer-events-none select-none"
      >
        {node.label}
      </text>

      {/* Connection points */}
      <circle
        cx={node.style.size.width / 2}
        cy="0"
        r="4"
        fill={nodeColor}
        className="opacity-0 hover:opacity-100 transition-opacity duration-200"
      />
      <circle
        cx={node.style.size.width / 2}
        cy={node.style.size.height}
        r="4"
        fill={nodeColor}
        className="opacity-0 hover:opacity-100 transition-opacity duration-200"
      />
    </g>
  )
}

interface ConnectionLineProps {
  connection: Connection
  nodes: LayerNode[]
  stroke: string
}

const ConnectionLine: React.FC<ConnectionLineProps> = ({ connection, nodes, stroke }) => {
  const sourceNode = nodes.find(n => n.id === connection.source)
  const targetNode = nodes.find(n => n.id === connection.target)

  if (!sourceNode || !targetNode) return null

  const sourceX = sourceNode.position.x + sourceNode.style.size.width / 2
  const sourceY = sourceNode.position.y + sourceNode.style.size.height
  const targetX = targetNode.position.x + targetNode.style.size.width / 2
  const targetY = targetNode.position.y

  if (connection.type === 'bezier') {
    const controlPointOffset = Math.abs(targetY - sourceY) / 3
    const path = `M ${sourceX} ${sourceY} C ${sourceX} ${sourceY + controlPointOffset} ${targetX} ${targetY - controlPointOffset} ${targetX} ${targetY}`
    
    return (
      <path
        d={path}
        stroke={stroke}
        strokeWidth="2"
        fill="none"
        markerEnd="url(#arrowhead)"
        className="transition-all duration-200"
      />
    )
  }

  return (
    <line
      x1={sourceX}
      y1={sourceY}
      x2={targetX}
      y2={targetY}
      stroke={stroke}
      strokeWidth="2"
      markerEnd="url(#arrowhead)"
      className="transition-all duration-200"
    />
  )
}

export default DiagramCanvas