import json
from random import shuffle
from math import floor
from utils import indexInList
from sklearn.externals import joblib
from collections import Counter
from numpy import array
from keras.utils import to_categorical

# Dataset Preparation
#return tuples of numpy arrays for train, test: ([x_train, y_train]),  ([x_test, y_test])
def loadData(ingredientSkip=0, ingredientCutoff=1000, train_path='../data/train.json', test_path='../data/test.json', validation_portion=0.1):
    with open(train_path) as trainJSON, open(test_path) as testJSON:
        train = json.load(trainJSON)
        #randomize training set
        shuffle(train)
        #get ingredients as a list of strings
        ingredients = getIngredientLabels(train, ingredientCutoff)
        #create list of recipes to encode
        recipes = [recipe['ingredients'] for recipe in train]
        #encode recipes
        encodedRecipes = onehotEncode(recipes, ingredients)
        #get labels for recipes
        labels = list(recipe["cuisine"] for recipe in train)
        cuisines = set(labels)
        print(cuisines)
        print(len(cuisines))
        #turn cuisines back into a list for onehotEncoding
        cuisines = list(cuisines)
        #convert cuisine names (strings) to index (int) in cuisines set
        encodedLabels = onehotCuisineEncode(labels, cuisines)
        #split around validation_length as index
        validation_length = int(float(len(train))*validation_portion)
        x_train = array(encodedRecipes[validation_length:])
        y_train = array(encodedLabels[validation_length:])
        x_test = array(encodedRecipes[:validation_length])
        y_test = array(encodedLabels[:validation_length])
        return((x_train, y_train), (x_test, y_test))


def encodeLabels(labels, cuisines):
    encoded = list()
    for l in labels:
        encoded.append(indexInList(l, cuisines))
    return encoded

# Takes train and test paths, returns two lists
def make_dataset(train_path='../data/train.json', test_path='../data/test.json', validation_portion=0.1):
    print ('Make Dataset from JSON ... ')
    with open(train_path) as trainJSON, open(test_path) as testJSON:
        train = json.load(trainJSON)
        test = json.load(testJSON)
        #randomize training set
        shuffle(train)
        #create a randomly selected validation set of size "validation_portion percent" of the test set
        validation_length = int(float(len(train))*validation_portion)
        validate = list(train)
        validate = validate[:validation_length]
        train = train[validation_length:]
        return train, test, validate


#the following function returns a list of vectors, each of which represents a recipe with ingredients one-hot encoded.
#example return value with two elems in source, 6 items recognized
#[[0,1,1,0,1,1], [1, 1, 0, 0, 0, 0]]
#
#source: a list of lists (list of recipes)
#labels: labels for each encoded item (ingredients labels)


def onehotEncode(source, labels):
    #print('Onehot encode source data...')
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


def onehotPickle(recipesToPickle, labels, dest):
    encoded = onehotEncode(recipesToPickle, labels)
    joblib.dump(encoded, dest)
    return encoded


def getIngredientLabels(dataset, cutoff):
    total = [recipe['ingredients'] for recipe in dataset]
    total = [cat for sublist in total for cat in sublist] #flatten the list
    ingredientCounter = Counter(total)
    topIngredientsCount = list(ingredientCounter.most_common())[:cutoff]
    ingredientLabels = list()
    for k, i in topIngredientsCount:
        #exclude unique items
        if(i>1):
            ingredientLabels.append(k)
    return ingredientLabels


def getIngredients(dataset, skip, cutoff):
    total = [recipe['ingredients'] for recipe in dataset]
    total = [cat for sublist in total for cat in sublist] #flatten the list
    ingredientCounter = Counter(total)
    topIngredientsCount = list(ingredientCounter.most_common())[skip:cutoff]
    ingredientLabels = list()
    for k, i in topIngredientsCount:
        #exclude unique items
        if(i>1):
            ingredientLabels.append(k)
    return ingredientLabels
