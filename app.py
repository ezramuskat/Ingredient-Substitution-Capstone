from flask import Flask, render_template, request, flash
from models.heuristic_model import heuristic_model

app = Flask(__name__, template_folder='../frontend')

processed_recipe = []
#restrictions = []



def process_recipe(ingredients, restrictions, model="heuristic"):
      #TODO: actually incorporate model(s) here
      if model == "heuristic":
            print("heuristic")
            #TODO: this is a temporary workaround to handle the fact that the heuristic model can currently only handle one restriction at a time
            #restriction = restrictions[0]
            new_recipe = heuristic_model.load_model()(ingredients, restrictions)
            if isinstance(new_recipe, list):
                  return new_recipe
            else:
                  #TODO: better error handling here
                  print("error in model run:")
                  print(new_recipe)
      else:
            #TODO: 500 error here?
            print("unknown model type loaded")
      return ingredients


@app.route('/processed-recipe', methods=['GET', 'POST'])
def get_processed_recipe():
      global processed_recipe
      # TODO: grab results from model here
      if request.method == 'POST':
            contents = request.get_json()
            print(contents)
            processed_recipe = process_recipe(contents['ingredients'], contents['restrictions'])
            return {}
      else:
            print(processed_recipe)
            return {
                  'ingredients': processed_recipe
            }
      
      
