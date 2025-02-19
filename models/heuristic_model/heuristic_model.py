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


    def model(ingredients, dietary_restriction):
        '''
        This function takes in a list of ingredients from a recipe, identifies the ones that are not allowed by the dietary restriction, and returns a new list of ingredients that are allowed.

        Parameters
        ----------
        ingredients : list
            A list of ingredients from a recipe.

        dietary_restriction : str
            A string that specifies the dietary restriction. Can be 'vegan', 'vegetarian', or 'dairy-free'.

        Returns
        -------
        list
            A list of ingredients that are allowed by the dietary restriction.

        '''

        # Check if the dietary restriction is valid
        if dietary_restriction not in ['vegan', 'vegetarian', 'dairy-free']:
            return "Invalid dietary restriction. Please choose from 'vegan', 'vegetarian', or 'dairy-free'."

        heuristics = None

        if dietary_restriction == 'vegan':
            heuristics = all_heuristics['vegan']
        elif dietary_restriction == 'vegetarian':
            heuristics = all_heuristics['vegetarian']
        elif dietary_restriction == 'dairy-free':
            heuristics = all_heuristics['dairy-free']

        new_recipe = []
        for ingredient in ingredients:
            if ingredient.lower() in heuristics:
                new_recipe.append(heuristics[ingredient.lower()])
            else:
                new_recipe.append(ingredient)

        return new_recipe
    
    return model

if __name__ == "__main__":
    # Try loading the model and testing it
    ingredients = ["milk", "butter", "eggs", "flour", "sugar", "salt", "baking powder", "chocolate chips"]
    dietary_restriction = "vegan"

    model = load_model()
    new_recipe = model(ingredients, dietary_restriction)

    print(new_recipe)