# Recipe ML project to-do list

https://www.kaggle.com/c/whats-cooking-kernels-only

## Data wrangling
- ~~download and parse (data.py) list of JSON~~
- ~~make_dataset should return train, validate, test~~
- ~~validate is a randomly selected 10% of train.json, pre-shuffled [validation_portion=0.1 could be another default arg] (see python random library)~~
- ~~histogram of ingredients (from collections import Counter)~~
- ~~count of recipes in each cuisine, count of cuisines~~
- refactor one-hot encoding as function in data.py
- explore ingredients data to look for low-hanging fruit cleaning (example: are there ingredients that differ only by punctuation or capitalization?)

## Modeling
- ~~truncate list to top 20-200 ingredients~~
- ~~one-hot encode recipes~~
- ~~create list of encoded recipes and labels for train data~~
- ~~run and fit classifier~~
- print predictions and percent correct
- print summary for a portion of training data equivalent in size to the validation set (to better monitor overfitting)
- find the point at which additional ingredients are no longer helping the model
- try three different non-neural-net sklearn models to determine what performs best. Each model should be called via a short function in model.py
- (option a) Try an sklearn MLP regressor
- (option b) Install Tensorflow and get more ToDos

## Presentation
- task a
- task b
- task c
