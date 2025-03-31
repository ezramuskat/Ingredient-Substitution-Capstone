import { useState, useRef, useEffect } from "react";

function RecipeForm(props) {
  const [ingredients, setIngredients] = useState([""]);
  const [restrictions, setRestrictions] = useState([""]);
  const inputRefs = useRef([]); //for QoL stuff so the cursor goes to the new ingredient

  //DOM weirdness if this isn't separate from addIngredient
  useEffect(() => {
    if (inputRefs.current.length > 0) {
      inputRefs.current[inputRefs.current.length - 1]?.focus();
    }
  }, [ingredients]);

  const addIngredient = () => {
    setIngredients([...ingredients, ""]);
  };

  const handleIngredientsChange = (index, value) => {
    const newIngredients = [...ingredients];
    newIngredients[index] = value;
    setIngredients(newIngredients);
  };

  const handleCheckboxChange = (event) => {
    const { value, checked } = event.target;
    if (checked) {
      if (restrictions[0] === "") {
        setRestrictions([value]);
      } else {
        setRestrictions((prev) => [...prev, value]);
      }
    } else {
      setRestrictions((prev) => prev.filter((item) => item !== value));
    }
  };

  const handleKeyDown = (index, event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      if (ingredients[index].trim() !== "") {
        addIngredient();
      }
    }
  };

  const handleSubmit = () => {
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
	  		<label>
          <input type="checkbox" value="vegan" onChange={handleCheckboxChange} />
          Vegan
        </label>
        <label>
          <input type="checkbox" value="vegetarian" onChange={handleCheckboxChange} />
          Vegetarian
        </label>
        <label>
          <input type="checkbox" value="dairy_free" onChange={handleCheckboxChange} />
          Dairy-Free
        </label>
      </div>

      <div className="ingredients-container">
        {ingredients.map((text, index) => (
          <div key={index} className="ingredient-row">
            <input
              ref={(el) => (inputRefs.current[index] = el)}
              value={text}
              onChange={(e) => handleIngredientsChange(index, e.target.value)}
              onKeyDown={(e) => handleKeyDown(index, e)}
            />
            {index === ingredients.length - 1 && (
              <button onClick={addIngredient}>+</button>
            )}
          </div>
        ))}
        <button onClick={handleSubmit}>Submit</button>
      </div>

      <div className="instructions-container">
        <p>To enter a recipe, enter each ingredient in the middle. You can add a new ingredient by hitting the + button, or by hitting "Enter". When you've entered all the ingredients, click submit</p>
      </div>
    </div>
  );
}

export default RecipeForm
