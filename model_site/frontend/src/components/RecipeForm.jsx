import { useState } from "react";

function RecipeForm(props) {
  const [ingredients, setIngredients] = useState([""]);
  const [restrictions, setRestrictions] = useState([""]); //actually handle these

  const addIngredient = () => {
    setIngredients([...ingredients, ""]);
  };

  const handleChange = (index, value) => {
    const newIngredients = [...ingredients];
    newIngredients[index] = value;
    setIngredients(newIngredients);
  };

  const handleSubmit = () => {
	// TODO: this needs to be updated to send the data to the backend
	  props.submitFunc()
    fetch('/api/processed-recipe', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ingredients: ingredients,
        restrictions: restrictions,
      })
    })
    console.log("Submitted Data:", ingredients);
  };

  return (
    <div className="recipe-form-flex-space">
      <div className="restrictions-container">
	  		<input type="checkbox" value={"option1"}/>
            <input type="checkbox" value={"option2"}/>
            <input type="checkbox" value={"option3"}/>
            <input type="checkbox" value={"option4"}/>
      </div>

      <div className="ingredients-container">
        {ingredients.map((text, index) => (
          <div key={index} className="ingredient-row">
            <input
              value={text}
              onChange={(e) => handleChange(index, e.target.value)}
            />
            {index === ingredients.length - 1 && (
              <button onClick={addIngredient}>+</button>
            )}
          </div>
        ))}
        <button onClick={handleSubmit}>Submit</button>
      </div>

      <div className="instructions-container">
        <p>Instructions go here.</p>
      </div>
    </div>
  );
}

export default RecipeForm
