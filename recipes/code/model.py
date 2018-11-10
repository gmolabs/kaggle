from data import make_dataset
from collections import Counter

train, test, validate = make_dataset()

#get count of cuisines
cuisineCounter = Counter(recipe['cuisine'] for recipe in train)
print("~~~~~~~~~~~~~~CUISINES~~~~~~~~~~~~~~~~~~~~~~")
print("{} cuisines found...".format(len(cuisineCounter)))
for k, v in cuisineCounter.most_common():
    print("{}:  {} recipes".format(k, v))

#get count of ingredients
total = [recipe['ingredients'] for recipe in train]
total = [cat for sublist in total for cat in sublist] #flatten the list
ingredientCounter = Counter(total)
print("~~~~~~~~~~~~~~INGREDIENTS~~~~~~~~~~~~~~~~~~~~~~")
print("{} ingredients found...".format(len(ingredientCounter)))
for k, v in ingredientCounter.most_common():
    print("{}:  {}".format(k, v))
