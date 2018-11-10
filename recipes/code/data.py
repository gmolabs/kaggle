import json
from random import shuffle
from math import floor

# Dataset Preparation
# Takes train and test paths, returns two lists
def make_dataset(train_path='../data/train.json', test_path='../data/test.json', validation_portion=0.0001):
    print ('Make Dataset from JSON ... ')
    with open(train_path) as trainJSON, open(test_path) as testJSON:
        train = json.load(trainJSON)
        test = json.load(testJSON)
        shuffle(train)
        #create a randomly selected validation set of size "validation_portion percent" of the test set
        validation_length = int(float(len(train))*validation_portion)
        validate = list(train)
        validate = validate[:validation_length]
        train = train[validation_length:]
        return train, test, validate
