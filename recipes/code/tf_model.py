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
import re

def preprocess_recipe(recipe):
    recipe = [ingredient.lower() for ingredient in recipe]
    recipe = [re.sub(r'-', ' ',   ingredient)  for ingredient in recipe]
    recipe = [re.sub(r'[^a-zA-Z0-9 ]', '',   ingredient)  for ingredient in recipe]
    return recipe

#CONSTANTS

INGREDIENT_CUTOFF = 10000
INGREDIENT_SKIP = 0
VALIDATION_PORTION = .1
N_RECIPES_TO_GUESS = 1
N_EPOCHS = 10
BATCH_SIZE = 32
DROPOUT_RATE = .9
N_HIDDEN_SIZE = 1028
MIN_OCCURENCES = 1 #minimum number of times an ingredient must appear to be included

#print(keras.__version__)
print("*** CONSTANTS:")
print("****** Ingredient cutoff: {}".format(INGREDIENT_CUTOFF))
print("****** Ingredient skip: {}".format(INGREDIENT_SKIP))
print("****** Min Occurences to include an ingredient: {}".format(MIN_OCCURENCES))
print("****** Epochs: {}".format(N_EPOCHS))
print("****** Batch Size: {}".format(BATCH_SIZE))
print("****** Dropout rate: {}".format(DROPOUT_RATE))
print("****** Hidden Layer Size: {}".format(N_HIDDEN_SIZE))

le = preprocessing.LabelEncoder()
recipes = []

x_train = []
y_train = []
x_test = []
y_test = []
print("Loading Data...")

#prepare data for model
with open('../data/train.json') as dataJSON:
    data = json.load(dataJSON)
    shuffle(data)
    recipes = [recipe['ingredients'] for recipe in data]
    labels = le.fit_transform([recipe["cuisine"] for recipe in data])
    #labels = labels.reshape(len(labels), 1) #doesn't seem to change output of to_categorical
    #encodedLabels = to_categorical(labels)
    ingredients = getIngredients(data, INGREDIENT_SKIP, INGREDIENT_CUTOFF, MIN_OCCURENCES)
    #print(le.inverse_transform([1]))
    encodedRecipes = onehotEncodeRecipes(recipes, ingredients)
    validation_length = int(float(len(data))*VALIDATION_PORTION)
    test_length = 2*int(float(len(data))*VALIDATION_PORTION)

    x_train = np.array(encodedRecipes[test_length:])
    y_train = np.array(labels[test_length:])
    x_validation = np.array(encodedRecipes[validation_length:test_length])
    y_validation = np.array(labels[validation_length:test_length])
    x_test = np.array(encodedRecipes[:validation_length])
    y_test = np.array(labels[:validation_length])


model = keras.Sequential()
model.add(keras.layers.Dense(N_HIDDEN_SIZE, activation=tf.nn.relu, input_shape=x_train[0].shape))
model.add(keras.layers.Dropout(DROPOUT_RATE))
model.add(keras.layers.Dense(len(le.classes_), activation=tf.nn.softmax))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=N_EPOCHS, batch_size=BATCH_SIZE)
test_loss, test_acc = model.evaluate(x_test, y_test)
model.save("../tf_models/model_{}.h5".format(test_acc))


print('Test accuracy:', test_acc)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


predictions = model.predict(x_test)

for j, recipe in enumerate(recipes):
    if j>=N_RECIPES_TO_GUESS:
        break
    print("Ingredients: {}".format(recipes[j]))

    print("Predictions:")
    for i, p in enumerate(predictions[j]):
        confidence = float(p*100.)
        if(confidence>=0.1):
            print("{}: {:.1f}%".format(le.classes_[i], confidence))
    print("Predicted cuisine: {}: {:.1f}%".format(le.classes_[np.argmax(predictions[j])], 100.*predictions[j][np.argmax(predictions[j])]))
    print("Actual cuisine: {}".format(le.classes_[y_test[j]]))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# Recreate the exact same model, including weights and optimizer.
new_model = keras.models.load_model('../tf_models/model_{}.h5'.format(test_acc))
new_model.summary()

new_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
loss, acc = new_model.evaluate(x_test, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
