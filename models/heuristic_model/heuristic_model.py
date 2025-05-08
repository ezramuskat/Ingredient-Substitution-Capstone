import json

def load_model(filename = "models/heuristic_model/heuristics.json"):
    '''
    This function loads the heuristics from a JSON file.

    Parameters
    ----------
    filename : str
        The name of the JSON file that contains the heuristics. The default value is "heuristics.json".

    Returns
    -------
    function
        A function that takes in a list of ingredients from a recipe and a dietary restriction, and returns a new list of ingredients that are allowed by the dietary restriction.
    '''

    all_heuristics = {}

    with open(filename, "r") as f:
        all_heuristics = json.load(f)


    def model(ingredients, dietary_restrictions):
        '''
        This function takes in a list of ingredients from a recipe, identifies the ones that are not allowed by the dietary restrictions, and returns a new list of ingredients that are allowed.

        Parameters
        ----------
        ingredients : list[str] 
            A list of ingredients from a recipe.

        dietary_restrictions : list[str]
            A list of strings that specifies the dietary restrictions. Can contain 'vegan', 'vegetarian', or 'dairy-free'.

        Returns
        -------
        list[str]
            A list of ingredients that are allowed by the dietary restrictions.

        '''

        # Check if the dietary restrictions are valid
        for restriction in dietary_restrictions:
            if restriction not in ['vegan', 'vegetarian', 'dairy-free']:
                raise Exception("Invalid dietary restriction found. Please choose from 'vegan', 'vegetarian', or 'dairy-free'.")

        # Load up the heuristics with the dietary restrictions specified by the user
        heuristics = {}
        if 'vegan' in dietary_restrictions:
            heuristics.update(all_heuristics['vegan'])
        if 'vegetarian' in dietary_restrictions:
            heuristics.update(all_heuristics['vegetarian'])
        if 'dairy-free' in dietary_restrictions:
            heuristics.update(all_heuristics['dairy-free'])

        new_recipe = []
        for ingredient in ingredients:

            is_ingredient_problematic = False

            for problem, solution in heuristics.items():
                if problem.lower() in ingredient.lower():
                    new_recipe.append(solution)
                    is_ingredient_problematic = True
                    break
                
            if not is_ingredient_problematic:
                new_recipe.append(ingredient)

        return new_recipe
    
    return model

if __name__ == "__main__":
    # Try loading the model and testing it
    ingredients = ["milk", "butter", "brown eggs", "flour", "sugar", "salt", "baking powder", "chocolate chips", "ground beef"]
    dietary_restrictions = ["dairy-free", "vegetarian"]

    model = load_model()
    new_recipe = model(ingredients, dietary_restrictions)

    print(new_recipe)