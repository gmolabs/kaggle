#personal utilities
from utils import indexInList
#from data
from data import make_dataset, onehotEncode, onehotPickle, getIngredientLabels, loadData
#python libraries
from collections import Counter
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#print(tf.__version__)
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print("Loading Data...")
(x_train, y_train), (x_test, y_test) = loadData() #return tuples of numpy arrays
print("Data Loaded.")
