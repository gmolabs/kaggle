import re

def preprocess_recipe(recipe):
    recipe = [ingredient.lower() for ingredient in recipe]
    recipe = [re.sub(r'-', ' ',   ingredient)  for ingredient in recipe]
    recipe = [re.sub(r'[^a-zA-Z0-9 ]', '',   ingredient)  for ingredient in recipe]
    derivative_ingredients = []
    for ingredient in recipe:
        words = ingredient.split()
        for start_index in range(len(words) - 1):
            for end_index in range(start_index + 1, len(words)):
                derivative_ingredients.append(' '.join(words[start_index: end_index + 1]))
    return recipe + derivative_ingredients
