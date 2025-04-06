import json

def load_model(filename = "/Users/ezramuskat/YU_Documents/COM/capstone_stuff/const_main/Ingredient-Substitution-Capstone/models/heuristic_model/heuristics.json"):
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
        This function takes in a list of ingredients from a recipe, identifies the ones that are not allowed by the dietary restriction, and returns a new list of ingredients that are allowed.

        Parameters
        ----------
        ingredients : list
            A list of ingredients from a recipe.

        dietary_restrictions : list[str]
            A string that specifies the dietary restriction. Can be 'vegan', 'vegetarian', or 'dairy-free'.

        Returns
        -------
        list
            A list of ingredients that are allowed by the dietary restriction.

        '''

        # Check if the dietary restriction is valid
        for restriction in dietary_restrictions:
            if restriction not in ['vegan', 'vegetarian', 'dairy-free']:
                return "Invalid dietary restriction found. Please choose from 'vegan', 'vegetarian', or 'dairy-free'."

        heuristics = {}

        if 'vegan' in dietary_restrictions:
            heuristics.update(all_heuristics['vegan'])
        if 'vegetarian' in dietary_restrictions:
            heuristics.update(all_heuristics['vegetarian'])
        if 'dairy-free' in dietary_restrictions:
            heuristics.update(all_heuristics['dairy-free'])

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
    dietary_restrictions = ["vegan"]

    model = load_model()
    new_recipe = model(ingredients, dietary_restrictions)

    print(new_recipe)