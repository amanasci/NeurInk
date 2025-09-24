import React from 'react'
import { Download, Palette, Code, Eye } from 'lucide-react'
import { useDiagram } from '../context/DiagramContext'

interface HeaderProps {
  showMarkupEditor: boolean
  onToggleMarkupEditor: () => void
}

const Header: React.FC<HeaderProps> = ({ showMarkupEditor, onToggleMarkupEditor }) => {
  const { state, dispatch } = useDiagram()

  const handleExportSVG = () => {
    const svgElement = document.querySelector('#diagram-canvas svg')
    if (svgElement) {
      const svgData = new XMLSerializer().serializeToString(svgElement)
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' })
      const url = URL.createObjectURL(svgBlob)
      
      const link = document.createElement('a')
      link.href = url
      link.download = 'neural-network-diagram.svg'
      link.click()
      
      URL.revokeObjectURL(url)
    }
  }

  const handleThemeChange = (theme: 'light' | 'dark' | 'monochrome') => {
    dispatch({ type: 'SET_THEME', payload: theme })
  }

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-4">
        <h1 className="text-2xl font-bold text-gray-800">
          NeurInk
        </h1>
        <div className="text-sm text-gray-500">
          Neural Network Diagram Generator
        </div>
      </div>

      <div className="flex items-center space-x-4">
        {/* Theme Selector */}
        <div className="flex items-center space-x-2">
          <Palette className="w-4 h-4 text-gray-600" />
          <select
            value={state.theme}
            onChange={(e) => handleThemeChange(e.target.value as any)}
            className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
            <option value="monochrome">Monochrome</option>
          </select>
        </div>

        {/* View Toggle */}
        <button
          onClick={onToggleMarkupEditor}
          className={`flex items-center space-x-2 px-3 py-2 rounded-md transition-colors duration-200 ${
            showMarkupEditor
              ? 'bg-blue-100 text-blue-700'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          {showMarkupEditor ? (
            <>
              <Eye className="w-4 h-4" />
              <span className="text-sm">Canvas Only</span>
            </>
          ) : (
            <>
              <Code className="w-4 h-4" />
              <span className="text-sm">Split View</span>
            </>
          )}
        </button>

        {/* Export Button */}
        <button
          onClick={handleExportSVG}
          className="flex items-center space-x-2 bg-blue-600 text-white px-3 py-2 rounded-md hover:bg-blue-700 transition-colors duration-200"
        >
          <Download className="w-4 h-4" />
          <span className="text-sm">Export SVG</span>
        </button>
      </div>
    </header>
  )
}

export default Header