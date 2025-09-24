import { useState } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import DiagramCanvas from './components/DiagramCanvas'
import MarkupEditor from './components/MarkupEditor'
import { DiagramProvider } from './context/DiagramContext'

function App() {
  const [showMarkupEditor, setShowMarkupEditor] = useState(false)

  return (
    <DiagramProvider>
      <div className="h-screen w-full flex flex-col bg-gray-50">
        <Header 
          showMarkupEditor={showMarkupEditor}
          onToggleMarkupEditor={() => setShowMarkupEditor(!showMarkupEditor)}
        />
        
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          
          <div className="flex-1 flex">
            {showMarkupEditor ? (
              <>
                <div className="w-1/2 border-r border-gray-300">
                  <MarkupEditor />
                </div>
                <div className="w-1/2">
                  <DiagramCanvas />
                </div>
              </>
            ) : (
              <div className="w-full">
                <DiagramCanvas />
              </div>
            )}
          </div>
        </div>
      </div>
    </DiagramProvider>
  )
}

export default App
