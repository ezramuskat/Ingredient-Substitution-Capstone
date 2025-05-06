# Ingredient Substitution
BS"D

Here there will be a brief overview of the project, including its purpose and how it is meant to be used on a high level.

## Getting Started for Users
This will reference the website ezra made

## Getting Started for Developers
This will explain how to use the various modules we created. It will explain how to use the following:
- Setup the environment
- Full Models
    - Distance model
    - Heuristic Model
- Other Functionality
    - Similar Ingredients functionality
    - Filtering model
    - Eval framework
- How to start up the website
- Datasets we created:
    - Ingredient dataset
    - Recipe dataset
    - Classification dataset (common ingredients)

### Setting up the Environment
To set up the environment, run the `setup.sh` script in the root directory of the project. This will create a virtual environment and install all the necessary dependencies.

**Important Note:** It is recommended that you make the imports from a file that is in the root directory of the project, and not in any of the subdirectories. Alternatively you can add the root directory to your PYTHONPATH. This will ensure that the imports work correctly.

### Full Models
The Heuristic and Distance models are the two main models we created. They take a recipe and dietary restrictions and return a new recipe with substitutions for the ingredients that are not allowed.

To use the **Heuristic model**, import and call the function `load_model` from `models.heuristic_model.heuristic_model`. The load model function takes a file path as a parameter which should point to the "heuristics.json" file in `models/heuristic_model/`. This file contains the heuristics that the model uses to make substitutions. The function returns a function which applies the heuristics to a recipe. The options for dietary restrictions are 'vegan', 'vegetarian' and 'dairy-free.

```python
from models.heuristic_model.heuristic_model import load_model

heuristic_model = load_model("models/heuristic_model/heuristics.json")

recipe = ["pasta", "tomato sauce", "cheese"]
dietary_restriction = "dairy-free"

substituted_recipe = heuristic_model(recipe, dietary_restriction)
```

To use the **Distance Model**, import the `DistanceModel` class from `models.distance_model.integrated_model` and instantiate it. To perform inference, call `generate_substitutes` function on a recipe and, this time, on a list of dietary restrictions. The function will return a new recipe with substitutions for the ingredients that are not allowed.

```python
from models.distance_model.integrated_model import DistanceModel

distance_model = DistanceModel()

recipe = ["pasta", "tomato sauce", "cheese"]
dietary_restriction = ["dairy-free", "vegan"]

substituted_recipe = distance_model.generate_substitutes(recipe, dietary_restriction)
```

**Note:** The distance model relies on some files within the `models/distance_model/` directory. These are "`models/distance_model/similar_ingredeitns/all_ingredients.json` and `data_preparation/classification_dataset/common_ingredients.csv`. You may have to manually pass the paths to these files when instantiating the `DistanceModel` class.

### Other Functionality
#### Similar Ingredients Functionality
One of the features we developed in the process of building the models was a function that returns a list of similar ingredients. This is used for finding substitutes for ingredients that are not allowed in a recipe.

To use this functionality alone, import the `load_model` function from `models.distance_model.similar_ingredients.get_similar_ingredients`. The function takes an ingredient as a parameter and returns a list of similar ingredients. As with the distance model, this function relies on the `all_ingredients.json` file. You may have to manually pass the path to this file when calling the function.

```python
from models.distance_model.similar_ingredients.get_similar_ingredients import load_model

similar_ingredients_model = load_model("models/distance_model/similar_ingredeitns/all_ingredients.json")

similar_ingredients = similar_ingredients_model("pasta")
```

#### Filtering Model
The filtering model is a model that identifies if ingredients violate the dietary restrictions we worked with (vegan, vegetarian, dairy-free, gluten-free). It is used to filter out ingredients that are not allowed in a recipe.



## Presentation

## About Us
### Students
**Name:** Solomon Firfer  
**Bio:** <Fill in later>  
**LinkedIn:** <Fill in later>  

**Name:** Tuvya Macklin  
**Bio:** <Fill in later>  
**LinkedIn:** <Fill in later>  


**Name:** Ezra Muskat  
**Bio:** <Fill in later>  
**LinkedIn:** <Fill in later>  

### Supervisor
**Name:** Professor Dave Feltenberger  
**Bio:** <Fill in later>  
**LinkedIn:** <Fill in later>  