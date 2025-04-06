import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import RecipeForm from './components/RecipeForm'
import SubstitutedRecipe from './components/SubstitutedRecipe'

function App() {
  const [isSubmitted, setIsSubmitted] = useState(false)

  function submitFunc() {
    console.log("Marking as submitted")
    setIsSubmitted(true)
  }

  function resetSubmitted() {
    console.log("Marking as not submitted")
    setIsSubmitted(false)
  }

  return (
      isSubmitted ?  <SubstitutedRecipe resetFunc={resetSubmitted}/> : <RecipeForm submitFunc={submitFunc}/>
  )
}

export default App
