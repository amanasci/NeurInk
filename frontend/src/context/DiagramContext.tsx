import React, { createContext, useContext, useReducer } from 'react'
import type { ReactNode } from 'react'

// Types
export interface LayerNode {
  id: string
  type: string
  label: string
  position: { x: number; y: number }
  parameters: Record<string, any>
  style: {
    color: string
    size: { width: number; height: number }
  }
}

export interface Connection {
  id: string
  source: string
  target: string
  type: 'straight' | 'bezier' | 'step'
}

export interface DiagramState {
  nodes: LayerNode[]
  connections: Connection[]
  markupText: string
  theme: 'light' | 'dark' | 'monochrome'
  selectedNode: string | null
}

// Actions
type DiagramAction = 
  | { type: 'ADD_NODE'; payload: LayerNode }
  | { type: 'UPDATE_NODE'; payload: { id: string; updates: Partial<LayerNode> } }
  | { type: 'DELETE_NODE'; payload: string }
  | { type: 'ADD_CONNECTION'; payload: Connection }
  | { type: 'DELETE_CONNECTION'; payload: string }
  | { type: 'UPDATE_MARKUP'; payload: string }
  | { type: 'SET_THEME'; payload: 'light' | 'dark' | 'monochrome' }
  | { type: 'SELECT_NODE'; payload: string | null }
  | { type: 'LOAD_FROM_MARKUP'; payload: { nodes: LayerNode[]; connections: Connection[] } }

const initialState: DiagramState = {
  nodes: [],
  connections: [],
  markupText: `input size=224x224x3
conv filters=64 kernel=3x3 stride=1 activation=relu
pool kernel=2x2 stride=2
conv filters=128 kernel=3x3 stride=1 activation=relu
pool kernel=2x2 stride=2
flatten
dense units=128 activation=relu
dropout rate=0.5
output units=10 activation=softmax`,
  theme: 'light',
  selectedNode: null,
}

function diagramReducer(state: DiagramState, action: DiagramAction): DiagramState {
  switch (action.type) {
    case 'ADD_NODE':
      return {
        ...state,
        nodes: [...state.nodes, action.payload]
      }
    case 'UPDATE_NODE':
      return {
        ...state,
        nodes: state.nodes.map(node =>
          node.id === action.payload.id
            ? { ...node, ...action.payload.updates }
            : node
        )
      }
    case 'DELETE_NODE':
      return {
        ...state,
        nodes: state.nodes.filter(node => node.id !== action.payload),
        connections: state.connections.filter(
          conn => conn.source !== action.payload && conn.target !== action.payload
        )
      }
    case 'ADD_CONNECTION':
      return {
        ...state,
        connections: [...state.connections, action.payload]
      }
    case 'DELETE_CONNECTION':
      return {
        ...state,
        connections: state.connections.filter(conn => conn.id !== action.payload)
      }
    case 'UPDATE_MARKUP':
      return {
        ...state,
        markupText: action.payload
      }
    case 'SET_THEME':
      return {
        ...state,
        theme: action.payload
      }
    case 'SELECT_NODE':
      return {
        ...state,
        selectedNode: action.payload
      }
    case 'LOAD_FROM_MARKUP':
      return {
        ...state,
        nodes: action.payload.nodes,
        connections: action.payload.connections
      }
    default:
      return state
  }
}

// Context
const DiagramContext = createContext<{
  state: DiagramState
  dispatch: React.Dispatch<DiagramAction>
} | null>(null)

export const DiagramProvider = ({ children }: { children: ReactNode }) => {
  const [state, dispatch] = useReducer(diagramReducer, initialState)

  return (
    <DiagramContext.Provider value={{ state, dispatch }}>
      {children}
    </DiagramContext.Provider>
  )
}

export const useDiagram = () => {
  const context = useContext(DiagramContext)
  if (!context) {
    throw new Error('useDiagram must be used within a DiagramProvider')
  }
  return context
}