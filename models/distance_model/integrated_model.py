import pandas as pd
import torch.nn as nn
from collections import OrderedDict

from models.distance_model.filtering_model.filtering_model import FilteringModel
from models.distance_model.similar_ingredients.get_similar_ingredients import load_model as get_similar_ingredients_model

class DistanceModel:
    '''
    A class to generate a modified recipe based on dietary restrictions.

    This class substitutes restricted ingredients in a recipe with suitable alternatives
    based on the selected dietary restriction (e.g., vegan, vegetarian, dairy-free).

    Methods
    -------
    generate_substitutes(ingredients, dietary_restrictions):
        Creates a new recipe with substituted ingredients based on dietary restrictions.
    '''

    def __init__(self, hyperparameters=None,
                 filtering_model_training_data_path="/Users/tuvyamacklin/Documents/Repos/Ingredient-Substitution-Capstone/data_preparation/classification_dataset/common_ingredients.csv",
                similar_ingredients_all_ingredients_path="/Users/tuvyamacklin/Documents/Repos/Ingredient-Substitution-Capstone/models/distance_model/similar_ingredients/all_ingredients.json"
                 ):
        '''
        Initializes the DistanceModel class.

        Parameters
        ----------
        hyperparameters : dict, optional
            A dictionary of hyperparameters for the model. If None, default values are used. See below for details.

        filtering_model_training_data_path : str
            The path to the training data for the filtering model.

        similar_ingredients_all_ingredients_path : str
            The path to the all ingredients data for the similar ingredients model.

        Notes
        -----
        The following hyperparameters are used:
        - threshold_1: float
            The threshold for the first filtering step. Default is 0.3.
        - threshold_2: float
            The threshold for the second filtering step. Default is 0.7.
        - num_similar_ingredients: int
            The number of similar ingredients to consider as alternatives. Default is 20.
        '''
        
        # Load the filtering model and set it up
        training_data = pd.read_csv(filtering_model_training_data_path)

        internal_model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(768, 256)),
            ('relu1', nn.LeakyReLU()),
            ('bn1', nn.BatchNorm1d(256)),
            ('fc2', nn.Linear(256, 64)),
            ('dr1', nn.Dropout(0.3)),
            ('relu2', nn.LeakyReLU()),
            ('bn2', nn.BatchNorm1d(64)),
            ('fc3', nn.Linear(64, 4)),
            ('sg1', nn.Sigmoid())
        ]))

        ingredient_column = 'ingredient'
        embedding_model = "facebook/drama-base"

        self.filtering_model = FilteringModel(training_data, ingredient_column, embedding_model, internal_model)

        self.filtering_model.train_model(epochs=20,batch_size=33,val_split=0.2)

        # Load the similar ingredients model
        self.similar_ingredients_model = get_similar_ingredients_model(file_name=similar_ingredients_all_ingredients_path)
    
        # Set the hyperparameters
        if hyperparameters is None:
            self._hyperparameters = {
                'threshold_1': 0.3,
                'threshold_2': 0.7,
                'num_similar_ingredients': 20
            }
        else:
            self._hyperparameters = hyperparameters

    def _flag_violations(self, recipe, dietary_restrictions, threshold=0.5):
        '''
        Flag ingredients in the recipe that violate dietary restrictions.

        Parameters
        -----------
        recipe : list
            List of ingredients in the recipe.

        dietary_restrictions : list
            List of dietary restrictions to check against. (Can be "vegan", "vegetarian", "gluten_free", or "dairy_free")

        threshold : float
            The threshold for the filtering model. Default is 0.5.

        Returns
        --------
        non_violations : list
            List of ingredients that do not violate the dietary restrictions.

        violations : list
            List of ingredients that violate the dietary restrictions.
        '''

        # First run the recipe through the filtering model
        filtered_recipe = self.filtering_model.filter(recipe, threshold=threshold)

        # Then convert it to a dictionary of dictionaries
        recipe_dict = filtered_recipe.to_dict(orient='records')

        # Split the recipe into non-violations and violations
        non_violations = []
        violations = []

        for ingredient in recipe_dict:
            violation = False

            for restriction in dietary_restrictions:
                if ingredient[restriction] == "no":
                    violation = True
                    break

            if violation:
                violations.append(ingredient['ingredient'])
            else:
                non_violations.append(ingredient['ingredient'])
        return non_violations, violations
    
    def _get_similar_ingredients(self, violations):
        '''
        Get similar ingredients for the violations.

        Parameters
        -----------
        violations : list
            List of ingredients that violate the dietary restrictions.

        Returns
        --------
        similar_ingredients : dict
            Dictionary of ingredients and their similar ingredients.
        '''

        # Get similar ingredients for each violation
        similar_ingredients = {}

        for ingredient in violations:
            # Get similar ingredients
            similar = self.similar_ingredients_model(ingredient, top_n=self._hyperparameters['num_similar_ingredients'])
        
            # Store the similar ingredients in the dictionary
            similar_ingredients[ingredient] = similar

        return similar_ingredients
    
    def _pick_substitutes(self, similar_ingredients, dietary_restrictions, num_suggestions = 3):
        '''
        Pick ingredients from the similar ingredients list that do not violate the dietary restrictions.

        Parameters
        -----------
        similar_ingredients : dict
            Dictionary of ingredients and their similar ingredients.

        dietary_restrictions : list
            List of dietary restrictions to check against. (Can be "vegan", "vegetarian", "gluten_free", or "dairy_free")

        num_suggestions : int
            Number of suggestions to return for each violation.

        Returns
        --------
        substitutes : dict
            Dictionary of violations and their substitutes.
        '''

        # Get substitutes for each violation
        substitutes = {}
        
        for ingredient, similar in similar_ingredients.items():
            if len(similar) == 0:
                # Concatenate the ingredient with "(no substitutes found)"
                substitutes[ingredient] = [ingredient + " (no substitutes found)"]
                continue

            # Convert the similar ingredients to a dictionary from the ingredient to the similarity score
            similar_dict = {i[0]: i[1] for i in similar}

            # Get a list of the similar ingredients
            similar_list = list(similar_dict.keys())

            # Check which of them violate the dietary restrictions
            non_violations, _ = self._flag_violations(similar_list, dietary_restrictions, threshold=self._hyperparameters['threshold_2'])

            # Sort the non-violations by their similarity score
            non_violations = sorted(non_violations, key=lambda x: similar_dict[x], reverse=True)

            # Pick the top n suggestions
            substitutes[ingredient] = non_violations[:num_suggestions]

            # If there are no substitutes, add the original ingredient
            if substitutes[ingredient] is None or len(substitutes[ingredient]) == 0:
                substitutes[ingredient] = [ingredient + " (no substitutes found)"]
        return substitutes
    
    def generate_substitutes(self, recipe, dietary_restrictions):
        '''
        Generate substitutes for the recipe based on the dietary restrictions.

        Parameters
        -----------
        recipe : list
            List of ingredients in the recipe.

        dietary_restrictions : list
            List of dietary restrictions to check against. (Can be "vegan", "vegetarian", "gluten_free", or "dairy_free")

        Returns
        --------
        list
            A new recipe with substituted ingredients based on dietary restrictions.
        '''

        # Flag the violations
        print("Recipe: ", recipe)
        non_violations, violations = self._flag_violations(recipe, dietary_restrictions, threshold=self._hyperparameters['threshold_1'])

        # Get similar ingredients for the violations
        similar_ingredients = self._get_similar_ingredients(violations)

        # Print the amounf of similar ingredients found for each violation
        for ingredient, similar in similar_ingredients.items():
            print(f"Similar ingredients for {ingredient}: {len(similar)}")
        # Print the similar ingredients
        for ingredient, similar in similar_ingredients.items():
            print(f"Similar ingredients for {ingredient}: {similar}")

        # Pick substitutes from the similar ingredients list that do not violate the dietary restrictions
        substitutes = self._pick_substitutes(similar_ingredients, dietary_restrictions, num_suggestions=1)

        # Create a new recipe with the substitutes
        new_recipe = []
        for ingredient in recipe:
            if ingredient in substitutes.keys():
                new_recipe.append(substitutes[ingredient][0])
            else:
                new_recipe.append(ingredient)

        return new_recipe
    
    def set_hyperparameters(self, hyperparameters):
        '''
        Set the hyperparameters for the model.

        Parameters
        ----------
        hyperparameters : dict
            A dictionary of hyperparameters for the model. See below for details.

        Notes
        -----
        The following hyperparameters are used:
        - threshold_1 (float): The threshold for the first filtering step. Default is 0.3.
        - threshold_2 (float): The threshold for the second filtering step. Default is 0.7.
        - num_similar_ingredients (int): The number of similar ingredients to consider as alternatives. Default is 20.
        '''

        # Check if the hyperparameters are valid
        if not isinstance(hyperparameters, dict):
            raise ValueError("Hyperparameters must be a dictionary.")
        if 'threshold_1' not in hyperparameters or 'threshold_2' not in hyperparameters or 'num_similar_ingredients' not in hyperparameters:
            raise ValueError("Hyperparameters must contain 'threshold_1', 'threshold_2', and 'num_similar_ingredients' keys.")
        if not isinstance(hyperparameters['threshold_1'], float) or not isinstance(hyperparameters['threshold_2'], float):
            raise ValueError("threshold_1 and threshold_2 must be floats.")
        if not isinstance(hyperparameters['num_similar_ingredients'], int):
            raise ValueError("num_similar_ingredients must be an integer.")
        
        # Make a copy of the hyperparameters
        self._hyperparameters = hyperparameters.copy()

    def get_hyperparameters(self):
        '''
        Get the hyperparameters for the model.

        Returns
        -------
        dict
            A dictionary of hyperparameters for the model. See below for details.

        Notes
        -----
        The following hyperparameters are used:
        - threshold_1 (float): The threshold for the first filtering step. Default is 0.3.
        - threshold_2 (float): The threshold for the second filtering step. Default is 0.7.
        - num_similar_ingredients (int): The number of similar ingredients to consider as alternatives. Default is 20.
        '''
        
        # Return a copy of the hyperparameters
        return self._hyperparameters.copy()