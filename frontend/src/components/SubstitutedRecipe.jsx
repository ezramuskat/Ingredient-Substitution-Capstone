import { useEffect, useState } from "react";

function SubstitutedRecipe(props) {

	return (
		<div className="recipe-form-flex-space">
			<div className="restrictions-container">
				{/** blank column for formatting purposes, we can potentially put other stuff here later */}
				<button onClick={props.resetFunc}>Enter a new recipe</button>
			</div>

			<div className="ingredients-container">
				{props.ingredients.map((text, index) => (
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