import { useState } from "react";

function RecipeForm(props) {
  const [rawIngredients, setRawIngredients] = useState("");
  const [restrictions, setRestrictions] = useState([""]);

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

  const handleSubmit = async () => {
    const ingredients = rawIngredients
      .split(/[\n,;]+/)
      .map(i => i.trim())
      .filter(i => i.length > 0);

    const response = await fetch('/api/processed-recipe', {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ingredients: ingredients,
        restrictions: restrictions,
      })
    });

    console.log("Submitted Data:", ingredients);
    if (response.ok) {
      const result = await response.json();
      props.submitFunc(result.ingredients);
    }
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
        <textarea
          rows="10"
          cols="50"
          value={rawIngredients}
          onChange={(e) => setRawIngredients(e.target.value)}
          placeholder="Enter ingredients separated by commas, semicolons, or new lines"
        />
        <button onClick={handleSubmit}>Submit</button>
      </div>

      <div className="instructions-container">
        <p>Enter all ingredients into the large box above. You can separate them using commas, semicolons, or new lines. Click submit when you're ready.</p>
      </div>
    </div>
  );
}

export default RecipeForm;
