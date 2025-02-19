from flask import Flask, render_template, request, flash

app = Flask(__name__, template_folder='../frontend')

processed_recipe = []
#restrictions = []

def process_recipe(ingredients, restrictions):
      #TODO: actually incorporate model(s) here
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
      
      
