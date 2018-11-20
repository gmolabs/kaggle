import json
from random import shuffle
from math import floor
from utils import indexInList


# Dataset Preparation
# Takes train and test paths, returns two lists
def make_dataset(train_path='../data/train.json', test_path='../data/test.json', validation_portion=0.001):
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
    print('Onehot encode source data')
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
