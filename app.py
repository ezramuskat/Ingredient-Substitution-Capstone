from flask import Flask, send_from_directory, request, session
import os
import logging
import sys
from models.heuristic_model import heuristic_model

IS_PRODUCTION = os.getenv('FLASK_ENV') == 'production'

app = Flask(__name__, static_folder='frontend/dist' if IS_PRODUCTION else None)

#logging stuff
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)



def process_recipe(ingredients, restrictions, model="heuristic"):
      if model == "heuristic":
            print("heuristic")
            new_recipe = heuristic_model.load_model(filename="models/heuristic_model/heuristics.json")(ingredients, restrictions)
            if isinstance(new_recipe, list):
                  print("new recipe:")
                  print(new_recipe)
                  return new_recipe
            else:
                  app.logger.info("error in model run:")
                  app.logger.info(new_recipe)
      else:
            app.logger.info("unknown model type loaded")
      return ingredients


@app.route('/api/processed-recipe', methods=['GET', 'POST'])
def get_processed_recipe():
      global processed_recipe
      if request.method == 'POST':
            contents = request.get_json()
            app.logger.info(contents)
            processed_recipe = process_recipe(contents['ingredients'], contents['restrictions'])
            #session['processed_recipe'] = processed_recipe
            return {'ingredients': processed_recipe}, 200

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