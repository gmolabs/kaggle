import json

# Dataset Preparation
print ("Make Dataset from JSON ... ")

# Takes train and test paths, returns two lists
def make_dataset(train_path, test_path):
    return json.load(open(train_path)), json.load(open(test_path))
