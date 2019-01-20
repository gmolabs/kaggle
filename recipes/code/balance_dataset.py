import json
from random import shuffle
from math import floor
from utils import indexInList
from collections import Counter
from numpy import array

with open('../data/validation.json') as data_file:
    print("balancing dataset")
    output =[]
    data = json.load(data_file)
    recipeCounter = Counter(recipe['cuisine'] for recipe in data)
    max = recipeCounter.most_common()[0][1]
    cuisines = list(recipeCounter.keys())
    splitJSONList = []
    for cuisine in cuisines:
        splitJSONList.append([])
    for recipe in data:
        recipeCuisine = recipe['cuisine']
        cuisinesIndex = cuisines.index(recipeCuisine)
        splitJSONList[cuisinesIndex].append(recipe)
    for cuisine in splitJSONList:
        while len(cuisine) < max:
            cuisine.extend(cuisine)
        dif = len(cuisine)-max
        del(cuisine[:dif])
        output.extend(cuisine)
    shuffle(output)
    with open('../data/balanced.json', 'w') as fp:
        json.dump(output, fp)
