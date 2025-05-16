# Ingredient Substitution
BS"D

This project provides dietary-aware ingredient substitutions using NLP-based models.

Beyond this functionality, we also created various general purpose models and functionality that can be used for other projects relating to ingredient substitution and recipe generation.

## Table of Contents
- [Ingredient Substitution](#ingredient-substitution)
  - [Table of Contents](#table-of-contents)
  - [Getting Started for Users](#getting-started-for-users)
  - [Getting Started for Developers](#getting-started-for-developers)
    - [Setting up the Environment](#setting-up-the-environment)
    - [Full Models](#full-models)
    - [Other Functionality](#other-functionality)
      - [Similar Ingredients Functionality](#similar-ingredients-functionality)
      - [Filtering Model](#filtering-model)
      - [Eval Framework](#eval-framework)
    - [Starting the Website Locally](#starting-the-website-locally)
    - [Datasets](#datasets)
  - [Presentation](#presentation)
  - [About Us](#about-us)
    - [Students](#students)
    - [Supervisor](#supervisor)

## Getting Started for Users
To use our project, you can go to the website [here](https://ingredient-substitution-capstone.onrender.com). The website allows you to input a recipe and dietary restrictions, and it will return a new recipe with substitutions for the ingredients that are not allowed.

## Getting Started for Developers
This will explain how to use the various models and tools we created.

### Setting up the Environment
To set up the environment, run the `setup.sh` script in the root directory of the project. This will create a virtual environment and install all the necessary dependencies. Note that you need to have Python 3.11 installed on your machine.

> **Important Note:** It is recommended that you make the imports from a file that is in the root directory of the project, and not in any of the subdirectories. Alternatively you can add the root directory to your PYTHONPATH. This will ensure that the imports work correctly.

### Full Models
The Heuristic and Distance models are the two main models we created. They take a recipe and dietary restrictions and return a new recipe with substitutions for the ingredients that are not allowed.

To use the **Heuristic model**, import and call the function `load_heuristic_model` from the `models` module. The load function takes a file path as a parameter which should point to the "heuristics.json" file in `models/heuristic_model/`. This file contains the heuristics that the model uses to make substitutions. The function returns a function which applies the heuristics to a recipe. The options for dietary restrictions are 'vegan', 'vegetarian' and 'dairy-free'.

```python
from models import load_heuristic_model

heuristic_model = load_heuristic_model("models/heuristic_model/heuristics.json")

recipe = ["pasta", "tomato sauce", "cheese"]
dietary_restrictions = ["dairy-free", "vegan"]

substituted_recipe = heuristic_model(recipe, dietary_restriction)
```

To use the **Distance Model**, import the `DistanceModel` class from `models` module and instantiate it. To perform inference, call `generate_substitutes` function on a recipe and a list of dietary restrictions. (The distance model can also work with the gluten free diet.) The function will return a new recipe with substitutions for the ingredients that are not allowed.

```python
from models import DistanceModel

distance_model = DistanceModel()

recipe = ["pasta", "tomato sauce", "cheese"]
dietary_restriction = ["dairy-free", "vegan"]

substituted_recipe = distance_model.generate_substitutes(recipe, dietary_restriction)
```

> **Note:** The distance model relies on some files within the `models/distance_model/` directory. These are `models/distance_model/similar_ingredients/all_ingredients.json` and `data_preparation/classification_dataset/common_ingredients.csv`. You may have to manually pass the paths to these files when instantiating the `DistanceModel` class, if the class is run from a directory other than the project root.

### Other Functionality
#### Similar Ingredients Functionality
One of the features we developed in the process of building the models was a function that returns a list of similar ingredients. This is used for finding substitutes for ingredients that are not allowed in a recipe.

To use this functionality alone, import the `load_similar_ingredients_model` function from `models.distance_model` module. The function takes an ingredient as a parameter and returns a list of similar ingredients. As with the distance model, this function relies on the `all_ingredients.json` file. You may have to manually pass the path to this file when calling the function.

```python
from models.distance_model import load_similar_ingredients_model

path = "models/distance_model/similar_ingredients/all_ingredients.json"
model = load_similar_ingredients_model(path)

similar_ingredients = model("pasta")
```

#### Filtering Model
The filtering model is a model that identifies if ingredients violate the dietary restrictions we worked with (vegan, vegetarian, dairy-free, gluten-free). It is used to filter out ingredients that are not allowed in a recipe.

To use this model, import the `FilteringModel` class from the `models.distance_model` module and instantiate it. The model can be instantiated in two main ways:
1. By passing a path to a pre-trained model file. We have trained one which can be found at `models/distance_model/filtering_model/filtering_model.pt`.

```python
from models.distance_model import FilteringModel

model_path = "models/distance_model/filtering_model/filtering_model.pt"
filtering_model = FilteringModel(default_model_path=model_path)
```

2. By training the model from scratch using the `train` method. To use this method, load the training data from the `data_preparation/classification_dataset/common_ingredients_1000.csv` file as a pandas DataFrame, and pass it to the constructor of the `FilteringModel` class. Then call the `train` method on the instance of the class.

```python
import pandas as pd
from models.distance_model import FilteringModel

data_path = "data_preparation/classification_dataset/common_ingredients_1000.csv"
df = pd.read_csv(data_path)
filtering_model = FilteringModel(references=df)
filtering_model.train()
```

To use the model, call the `filter` method on an instance of the `FilteringModel` class. The method takes a list of ingredients and returns a dataframe specifying which ingredients are allowed and which are not, based on the dietary restrictions.

```python
ingredients = ["pasta", "tomato sauce", "cheese"]

filtering_model.filter(ingredients)
```

#### Eval Framework
The eval framework is a framework that allows you to evaluate the performance of the models. It provides four metrics to evaluate the models: IoU, F1, Precision, and Recall. This can be used to evaluate models that address our problem of ingredient substitution.

To use the eval framework, import the `eval()` function from the `eval.py` module. You will need to set the paths to the datasets used via the `RECIPES_PATH` and `PAIRS_PATH` variables. The function takes a model, a dataset, and a list of dietary restrictions as parameters. It returns the evaluation metrics for the model on the dataset.

The datasets can be downloaded from [this](https://huggingface.co/datasets/lishuyang/recipepairs/tree/main) link. The `RECIPES_PATH` variable should point to the `recipes.parquet` file and the `INGREDIENTS_PATH` variable should point to the `pairs.parquet` file.

You also need to encapsulate your model in a class that has a `generate` method. This method should take a recipe and a list of dietary restrictions as parameters and return a new recipe with substitutions for the ingredients that are not allowed.

```python
from eval import eval, RECIPES_PATH, PAIRS_PATH

PAIRS_PATH = "data/pairs.parquet"
RECIPES_PATH = "data/recipes.parquet"

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

### Starting the Website Locally
To run the website locally, execute the `startup_dev.sh` script. This will start up the website which can be accessed at `http://localhost:5173`.

### Datasets
We created three datasets for the project. The first dataset is a list of ingredients and their properties. The second dataset is a list of recipes and their ingredients. The third dataset is a list of common ingredients that are used in recipes.

The **Classification dataset** is a list of ingredients and what dietary restrictions they violate. It can be found at `data_preparation/classification_dataset/common_ingredients_1000.csv`.

The **Ingredients dataset** is a list of words that are ingredients in recipes. It can be found at `models/distance_model/similar_ingredients/all_ingredients.json`.

The **Recipes dataset** is a collection of recipes in the form of lists of ingredients. There are two versions of this in the `models/distance_model/similar_ingredients/data` directory, named `dataset_1.json` and `dataset_2.json`. They contain different recipes. It is recommended to use the second dataset first, as it is more precise.

> **Note:** To load this dataset into a pandas dataframe, you can use the following code: `pd.read_json(filepath, orient='table')`.

## Presentation
(Insert link to the presentation here)

## About Us
### Students
**Name:** Solomon Firfer  
**Bio:** <Fill in later>  
**[LinkedIn](https://www.linkedin.com/in/solomon-firfer-b9a20b265/s)**

**Name:** Tuvya Macklin  
**Bio:** Tuvya Macklin is a graduate of Yeshiva University with a degree in Computer Science specializing in AI, and has conducted research at Bar-Ilan University. He is currently seeking a job and is preparing to begin a Master's in Computer Science at Georgia Tech.  
**[LinkedIn](https://www.linkedin.com/in/tuvyamacklin/)**


**Name:** Ezra Muskat  
**Bio:** <Fill in later>  
**[LinkedIn](https://www.linkedin.com/in/ezra-muskat-b6695721a/)**

### Supervisor
**Name:** Professor Dave Feltenberger  
**Bio:** Professor Feltenberger is a former Engineering Director at Google and currently serves as a Director at Meta.  
**[LinkedIn](https://www.linkedin.com/in/davefeltenberger/)**