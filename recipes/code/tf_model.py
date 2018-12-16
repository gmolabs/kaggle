import json
from keras.utils import to_categorical
from sklearn import preprocessing

from utils import indexInList
from collections import Counter
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import array
from random import shuffle
from data import getIngredients, onehotEncodeRecipes

#CONSTANTS
INGREDIENT_CUTOFF = 1000
VALIDATION_PORTION = .1

print(keras.__version__)

le = preprocessing.LabelEncoder()
recipes = []

x_train = []
y_train = []
x_test = []
y_test = []
print("Loading Data...")
#(x_train, y_train), (x_test, y_test) = loadData() #return tuples of numpy arrays

#prepare data for model
with open('../data/train.json') as dataJSON:
    data = json.load(dataJSON)
    shuffle(data)
    recipes = [recipe['ingredients'] for recipe in data]
    labels = le.fit_transform([recipe["cuisine"] for recipe in data])
    print(labels)
    #labels = labels.reshape(len(labels), 1) #doesn't seem to change output of to_categorical
    #encodedLabels = to_categorical(labels)
    ingredients = getIngredients(data, INGREDIENT_CUTOFF)
    #print(le.inverse_transform([1]))
    encodedRecipes = onehotEncodeRecipes(recipes, ingredients)
    validation_length = int(float(len(data))*VALIDATION_PORTION)
    x_train = np.array(encodedRecipes[validation_length:])
    y_train = np.array(labels[validation_length:])
    x_test = np.array(encodedRecipes[:validation_length])
    y_test = np.array(labels[:validation_length])

model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.softmax) #there are 20 cuisines, final category
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5, batch_size=32)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)

predictions = model.predict(x_test)

print("Ingredients for first recipe: {}".format(recipes[0]))

print("Predictions for first recipe: {}".format(predictions[0]))

print("Predicted cuisine: {}".format(le.classes_[np.argmax(predictions[0])]))
print("Actual cuisine: {}".format(le.classes_[y_test[0]]))
