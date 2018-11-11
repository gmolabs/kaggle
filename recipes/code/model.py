from data import make_dataset
from numpy import array
from collections import Counter
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


ingredientCutoff = 200
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

print("~~~~~~~~~~~~~~TOP INGREDIENTS~~~~~~~~~~~~~~~~~~~~~~")
print("Showing the top {} most common ingredients...".format(ingredientCutoff))
topIngredientsList = list(ingredientCounter.most_common())[:ingredientCutoff]
for k, v in topIngredientsList:
    print("{}:  {} uses".format(k, v))

#prep for one-hot encoding: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
#integer encode top ingredients
ingredientLabels = [item[0] for item in topIngredientsList] #list comprehension to extract labels and leave count
#print(ingredientLabels)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(ingredientLabels)
#print(integer_encoded)
#one-hot encode labels
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#print(onehot_encoded)


#create recipe vectors of 1hot-encoded ingredients (binary vecs with a 1 at the index of each ingredient)
