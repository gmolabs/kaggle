from data import make_dataset
from numpy import array
from collections import Counter
from sklearn import datasets
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from utils import indexInList


ingredientCutoff = 20
train, test, validate = make_dataset()

#get count of cuisines
#cuisineCounter = Counter(recipe['cuisine'] for recipe in train)
#print("~~~~~~~~~~~~~~CUISINES~~~~~~~~~~~~~~~~~~~~~~")
#print("{} cuisines found...".format(len(cuisineCounter)))
#for k, v in cuisineCounter.most_common():
#    print("{}:  {} recipes".format(k, v))

#get count of ingredients
total = [recipe['ingredients'] for recipe in train]
total = [cat for sublist in total for cat in sublist] #flatten the list
ingredientCounter = Counter(total)
#print("~~~~~~~~~~~~~~INGREDIENTS~~~~~~~~~~~~~~~~~~~~~~")
#print("{} ingredients found...".format(len(ingredientCounter)))
#for k, v in ingredientCounter.most_common():
#    print("{}:  {}".format(k, v))

#print("~~~~~~~~~~~~~~TOP INGREDIENTS~~~~~~~~~~~~~~~~~~~~~~")
#print("Showing the top {} most common ingredients...".format(ingredientCutoff))
topIngredientsCount = list(ingredientCounter.most_common())[:ingredientCutoff]
ingredientNames = list()
for k, _ in topIngredientsCount:
#    print("{}:  {} uses".format(k, v))
    ingredientNames.append(k)
#print(ingredientNames)

#create recipe vectors of 1hot-encoded ingredients (binary vecs with a 1 at the index of each ingredient)
#onehotIngredients = []
#for index, value in enumerate(ingredientNames):
#	encodedIngredient = [0 for _ in range(ingredientCutoff)]
#	encodedIngredient[index] = 1
#	onehotIngredients.append(encodedIngredient)
#print(onehotIngredients)

onehotRecipes = list()
recipes = [recipe['ingredients'] for recipe in train]
for recipe in recipes:
#    print(recipe)
    onehotRecipe = [0 for _ in range(ingredientCutoff)]
    for ingredient in recipe:
        i = indexInList(ingredient, ingredientNames)
        if(i>=0):
            onehotRecipe[i] = 1
    onehotRecipes.append(onehotRecipe)
#print(onehotRecipes)

onehotValidate = list()
recipes = [recipe['ingredients'] for recipe in validate]
for recipe in recipes:
#    print(recipe)
    onehotVal = [0 for _ in range(ingredientCutoff)]
    for ingredient in recipe:
        i = indexInList(ingredient, ingredientNames)
        if(i>=0):
            onehotVal[i] = 1
    onehotValidate.append(onehotVal)

#get labels for classifier
cuisineLabels = list(recipe["cuisine"] for recipe in train)
expected = list(recipe["cuisine"] for recipe in validate)

#setup classifier and fit data
classifier = svm.SVC(gamma='scale')
classifier.fit(onehotRecipes, cuisineLabels)

#predict cuisine from onehotRecipes
#print("predition should be:")
#print(expected)
predicted = classifier.predict(onehotValidate)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
