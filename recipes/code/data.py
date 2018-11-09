import json

# Dataset Preparation
# Takes train and test paths, returns two lists
def make_dataset(train_path='../data/train.json', test_path='../data/test.json'):
    print ('Make Dataset from JSON ... ')
    with open(train_path) as trainJSON, open(test_path) as testJSON:
        return json.load(trainJSON), json.load(testJSON)
