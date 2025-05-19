import { useState } from "react";

function RecipeForm(props) {
  const [rawIngredients, setRawIngredients] = useState("");
  const [restrictions, setRestrictions] = useState([""]);
  const [modelChoice, setModelChoice] = useState("heuristic");

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
        model: modelChoice,
      })
    });

    console.log("Submitted Data:", ingredients, restrictions, modelChoice);
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
          <input type="checkbox" value="dairy-free" onChange={handleCheckboxChange} />
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
        {/*<label>
					<input
						type="radio"
						name="method"
						value="heuristic"
						checked={modelChoice === "heuristic"}
						onChange={(e) => setModelChoice(e.target.value)}
					/>
					Heuristic
					</label>
					<label>
					<input
						type="radio"
						name="method"
						value="distance"
						checked={modelChoice === "distance"}
						onChange={(e) => setModelChoice(e.target.value)}
					/>
					Distance
					</label>
*/}
      </div>
    </div>
  );
}

export default RecipeForm;
