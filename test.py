# This document will test if you can run the integrated distance model from another directory
from models.distance_model.integrated_model import DistanceModel

def test_distance_model():
    recipe = ["beef", "onion", "garlic", "salt", "pepper", "cheese", "lettuce", "tomato", "bun"]
    restrictions = ["vegan", "gluten_free"]

    # Create an instance of the DistanceModel
    model = DistanceModel()

    # Call the predict method with the recipe and restrictions
    output = model.generate_substitutes(recipe, restrictions)

    print("Output:", output)

if __name__ == "__main__":
    test_distance_model()