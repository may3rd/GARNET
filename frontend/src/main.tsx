import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles/index.css'

// HeroUI v3 needs no app-level provider: styling comes from the imported
// stylesheet and theming from `data-theme` on <html>.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
)
