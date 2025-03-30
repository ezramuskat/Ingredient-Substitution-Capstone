import json
from gensim import downloader

def load_model(file_name = "models/distance_model/similar_ingredients/all_ingredients.json"):
    '''
    This function loads the similar ingredients model. It uses Google News word2vec embeddings to find similar ingredients. Note that it should take a few seconds to load the model.

    Parameters
    ----------
    file_name : str
        The name of the JSON file that contains valid ingredients. The default value is the absolute path to "all_ingredients.json".

    Returns
    -------
    function
        A function that takes in an ingredient and returns a list of similar ingredients.
        
    Notes
    -----
    The file all_ingredients.json contains a list of all valid ingredients. This is needed because the word embeddings may suggest words that are not food items.

    '''

    all_ingredients = []
    with open(file_name, "r") as f:
        all_ingredients = json.load(f)

    embedding_model = downloader.load("word2vec-google-news-300")

    def model(ingredient, top_n = 10):
        '''
        This function takes in an ingredient and returns a list of similar ingredients.

        Parameters
        ----------
        ingredient : str
            The ingredient for which similar ingredients are to be found.

        top_n : int
            The number of similar ingredients to return. The default value is 10.

        Returns
        -------
        list
            A list of similar ingredients.
        '''
        
        # Increase the top_n to compensate for the filtering
        top_n = top_n * 10

        suggestions = embedding_model.most_similar(ingredient, topn=top_n)
        
        filtered_suggestions = []

        for suggestion, score in suggestions:
            if suggestion.lower().replace(' ', '_') in all_ingredients:
                filtered_suggestions.append((suggestion, score))
        
        # Remove the underscores from the suggestions
        filtered_suggestions = [(suggestion.replace('_', ' '), score) for suggestion, score in filtered_suggestions]

        # Return the top_n suggestions
        top_n = top_n // 10
        return filtered_suggestions[:top_n]
    
    return model


# Try using the model
if __name__ == "__main__":
    model = load_model()
    
    print(model("ground beef"))