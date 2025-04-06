import { useEffect, useState } from "react";

function SubstitutedRecipe(props) {
	const [recipe, setRecipe] = useState({
		ingredients: [""],
	});
	
	useEffect(() => {
		fetch("/api/processed-recipe").then((response) => {
			if (response.ok) {
				console.log("Response:", response);
				response.json().then((contents) => {
					console.log("Contents:", contents);
					setRecipe({
						ingredients: contents.ingredients,
					});
				});
			}
		});
		console.log("Recipe Data:", recipe);
	}, []);

	return (
		<div className="recipe-form-flex-space">
			<div className="restrictions-container">
				{/** blank column for formatting purposes, we can potentially put other stuff here later */}
				<button onClick={props.resetFunc}>Enter a new recipe</button>
			</div>

			<div className="ingredients-container">
				{recipe.ingredients.map((text, index) => (
					<div key={index} className="ingredient-row">
						<p>{text}</p>
					</div>
				))}
			</div>

			<div className="instructions-container">
				<p>Instructions go here.</p>
			</div>
		</div>
	);


}

export default SubstitutedRecipe