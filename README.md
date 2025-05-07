# Ingredient Substitution
BS"D

__*Here there will be a brief overview of the project, including its purpose and how it is meant to be used on a high level.*__

## Getting Started for Users
To use our project, you can go to the website [here](https://ingredient-substitution-capstone.onrender.com). The website allows you to input a recipe and dietary restrictions, and it will return a new recipe with substitutions for the ingredients that are not allowed.

## Getting Started for Developers
This will explain how to use the various modules we created. It will explain how to use the following:
- Setup the environment
- Full Models
    - Distance model
    - Heuristic Model
- Other Functionality
    - Similar Ingredients functionality
    - Filtering model - NEED TO FILL IN
    - Eval framework
- How to start up the website
- Datasets we created:
    - Ingredient dataset
    - Recipe dataset
    - Classification dataset (common ingredients)

### Setting up the Environment
To set up the environment, run the `setup.sh` script in the root directory of the project. This will create a virtual environment and install all the necessary dependencies. Note that you need to have Python 3.11 installed on your machine.

> **Important Note:** It is recommended that you make the imports from a file that is in the root directory of the project, and not in any of the subdirectories. Alternatively you can add the root directory to your PYTHONPATH. This will ensure that the imports work correctly.

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

> **Note:** The distance model relies on some files within the `models/distance_model/` directory. These are "`models/distance_model/similar_ingredeitns/all_ingredients.json` and `data_preparation/classification_dataset/common_ingredients.csv`. You may have to manually pass the paths to these files when instantiating the `DistanceModel` class.

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

**_FILL IN LATER_**

#### Eval Framework
The eval framework is a framework that allows you to evaluate the performance of the models. It provides four metrics to evaluate the models: IoU, F1, Precision, and Recall. This can be used to evaluate models that address our problem of ingredient substitution.

To use the eval framework, import the `eval.py` file and load the `eval()` function. You will need to set the paths to the datasets used via the `RECIPES_PATH` and `PAIRS_PATH` variables. The function takes a model, a dataset, and a list of dietary restrictions as parameters. It returns the evaluation metrics for the model on the dataset.

The datasets can be downloaded from [this](https://huggingface.co/datasets/lishuyang/recipepairs/tree/main) link. The `RECIPES_PATH` variable should point to the `recipes.parquet` file and the `INGREDIENTS_PATH` variable should point to the `pairs.parquet` file.

You also need to encapsulate your model in a class that has a `generate` method. This method should take a recipe and a list of dietary restrictions as parameters and return a new recipe with substitutions for the ingredients that are not allowed.

```python
from eval import eval, RECIPES_PATH, PAIRS_PATH

PAIRS_PATH = "data/recipes.parquet"
RECIPES_PATH = "data/pairs.parquet"

class MyModel:
    def __init__(self):
        pass

    def generate(self, recipe, dietary_restriction):
        # Your model code here
        return substituted_recipe

model = MyModel()
dietary_restriction = ["dairy-free", "vegan"]

print(eval(model, RECIPES_PATH, PAIRS_PATH, dietary_restriction))
```

### Starting the Website
__*FILL IN LATER*__

### Datasets
We created three datasets for the project. The first dataset is a list of ingredients and their properties. The second dataset is a list of recipes and their ingredients. The third dataset is a list of common ingredients that are used in recipes.

The **Classification dataset** is a list of ingredients and what dietary restrictions they violate. It can be found at `data_preparation/classification_dataset/common_ingredients_1000.csv`.

The **Ingredients dataset** is a list of words that are ingredients in recipes. It can be found at `models/distance_model/similar_ingredients/all_ingredients.json`.

The **Recipes dataset** is a collection of recipes in the form of lists of ingredients. There are two versions of this in the `models/distance_model/similar_ingredients/data` directory, named `dataset_1.json` and `dataset_2.json`. They contain different recipes. It is reccommended to use the second dataset first, as it is more precise.

> **Note:** To load this dataset into a pandas dataframe, you can use the following code: `pd.read_json(filepath, orient='table')`.

## Presentation

## About Us
### Students
**Name:** Solomon Firfer  
**Bio:** <Fill in later>  
**[LinkedIn](https://www.linkedin.com/in/solomon-firfer-b9a20b265/s)**

**Name:** Tuvya Macklin  
**Bio:** <Fill in later>  
**[LinkedIn](https://www.linkedin.com/in/tuvyamacklin/)**


**Name:** Ezra Muskat  
**Bio:** <Fill in later>  
**[LinkedIn](https://www.linkedin.com/in/ezra-muskat-b6695721a/)**

### Supervisor
**Name:** Professor Dave Feltenberger  
**Bio:** Professor Feltenberger is a former Engineering Director at Google and currently serves as a Director at Meta.  
**[LinkedIn](https://www.linkedin.com/in/davefeltenberger/)**