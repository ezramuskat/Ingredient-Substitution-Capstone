# Ingredient Substitution
BS"D

Here there will be a brief overview of the project, including its purpose and how it is meant to be used on a high level.

## Getting Started
### For Users
This will reference the website ezra made

### For Developers
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

#### Setting up the Environment
To set up the environment, run the `setup.sh` script in the root directory of the project. This will create a virtual environment and install all the necessary dependencies.

#### Full Models
The Heuristic and Distance models are the two main models we created. They take a recipe and dietary restrictions and return a new recipe with substitutions for the ingredients that are not allowed.

**Important Note:** It is recommended that you make the imports from the root directory of the project, and not in any of the subdirectories. Alternatively you can add the root directory to your PYTHONPATH. This will ensure that the imports work correctly.

To use the Heuristic model, import and call the function `load_model` from `models.heuristic_model.heuristic_model`. The load model function takes a file path as a parameter which should point to the "heuristics.json" file in `models/heuristic_model/`. This file contains the heuristics that the model uses to make substitutions. The function returns a function which applies the heuristics to a recipe.

```python
from models.heuristic_model.heuristic_model import load_model

heuristic_model = load_model("models/heuristic_model/heuristics.json")

recipe = ["pasta", "tomato sauce", "cheese"]
dietary_restriction = "dairy-free"

substituted_recipe = heuristic_model(recipe, dietary_restriction)
```

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