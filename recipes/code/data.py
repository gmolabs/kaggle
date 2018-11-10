import json
from random import sample
from math import floor

# Dataset Preparation
# Takes train and test paths, returns two lists
def make_dataset(train_path='../data/train.json', test_path='../data/test.json', validation_portion=0.1):
    print ('Make Dataset from JSON ... ')
    with open(train_path) as trainJSON, open(test_path) as testJSON:
        train = json.load(trainJSON)
        test = json.load(testJSON)
        #create a randomly selected validation set of size "validation_portion percent" of the test set
        validation_length = int(float(len(train))*validation_portion)
        validate = sample(train,validation_length)
        return train, test, validate
