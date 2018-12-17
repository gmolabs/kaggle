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
from data_utils import preprocess_recipe

#CONSTANTS
INGREDIENT_CUTOFF = 4955 # 10093
VALIDATION_PORTION = .1

#print(keras.__version__)

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
    # for recipe in data:
    #     recipe['ingredients'] = preprocess_recipe(recipe['ingredients'])
    recipes = [recipe['ingredients'] for recipe in data]
    labels = le.fit_transform([recipe["cuisine"] for recipe in data])
    #labels = labels.reshape(len(labels), 1) #doesn't seem to change output of to_categorical
    #encodedLabels = to_categorical(labels)
    ingredients = getIngredients(data, INGREDIENT_CUTOFF)
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

model.add(keras.layers.Dense(512, input_dim=INGREDIENT_CUTOFF))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.7))
model.add(keras.layers.Dense(20))
model.add(keras.layers.Activation('softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=3, batch_size=32)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


predictions = model.predict(x_test)

for j, recipe in enumerate(recipes):
    if j>=0:
        break
    print("Ingredients: {}".format(recipes[j]))

    print("Predictions:")
    for i, p in enumerate(predictions[j]):
        confidence = float(p*100.)
        if(confidence>=0.1):
            print("{}: {:.1f}%".format(le.classes_[i], confidence))
    print("Predicted cuisine: {}: {:.1f}%".format(le.classes_[np.argmax(predictions[j])], 100.*predictions[j][np.argmax(predictions[j])]))
    print("Actual cuisine: {}".format(le.classes_[y_test[j]]))
    print("actual cuisine index: {}".format(y_test[j]))
    print(le.classes_)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
