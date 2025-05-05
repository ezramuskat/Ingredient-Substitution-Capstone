from flask import Flask, send_from_directory, request, session
import os
import logging
import sys
from models.heuristic_model import heuristic_model

IS_PRODUCTION = os.getenv('FLASK_ENV') == 'production'

app = Flask(__name__, static_folder='frontend/dist' if IS_PRODUCTION else None)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key")

#logging stuff
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

#processed_recipe = []
#restrictions = []



def process_recipe(ingredients, restrictions, model="heuristic"):
      #TODO: actually incorporate model(s) here
      if model == "heuristic":
            print("heuristic")
            #TODO: this is a temporary workaround to handle the fact that the heuristic model can currently only handle one restriction at a time
            #restriction = restrictions[0]
            new_recipe = heuristic_model.load_model(filename="models/heuristic_model/heuristics.json")(ingredients, restrictions)
            if isinstance(new_recipe, list):
                  return new_recipe
            else:
                  #TODO: better error handling here
                  app.logger.info("error in model run:")
                  app.logger.info(new_recipe)
      else:
            #TODO: 500 error here?
            app.logger.info("unknown model type loaded")
      return ingredients


@app.route('/api/processed-recipe', methods=['GET', 'POST'])
def get_processed_recipe():
      global processed_recipe
      # TODO: grab results from model here
      if request.method == 'POST':
            contents = request.get_json()
            app.logger.info(contents)
            processed_recipe = process_recipe(contents['ingredients'], contents['restrictions'])
            session['processed_recipe'] = processed_recipe
            return {}
      else:
            processed_recipe = session.get('processed_recipe', [])
            app.logger.info(processed_recipe)
            return {
                  'ingredients': processed_recipe
            }

if IS_PRODUCTION:
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_react(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)