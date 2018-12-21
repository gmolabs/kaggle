import json
from random import shuffle
from math import floor
from utils import indexInList
from sklearn.externals import joblib
from collections import Counter
from numpy import array
from keras.utils import to_categorical

def onehotEncodeRecipes(source, labels):
    #print('Onehot encode recipes data...')
    encoded = list()
    for elem in source:
    #    print(recipe)
        binaryVec = [0 for _ in range(len(labels))]
        for item in elem:
            i = indexInList(item, labels)
            if(i>=0):
                binaryVec[i] = 1
        encoded.append(binaryVec)
    return encoded

def getIngredients(dataset, skip, cutoff, min_occurences):
    ingredient_total = [recipe['ingredients'] for recipe in dataset]
    ingredient_total = [cat for sublist in ingredient_total for cat in sublist] #flatten the list
    ingredientCounter = Counter(ingredient_total)
    topIngredientsCount = list(ingredientCounter.most_common())[skip:cutoff]
    ingredientLabels = list()
    for k, i in topIngredientsCount:
        #exclude unique items
        if(i>min_occurences):
            ingredientLabels.append(k)
    return ingredientLabels
