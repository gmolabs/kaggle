# Recipe ML project to-do list

https://www.kaggle.com/c/whats-cooking-kernels-only

## Data wrangling
- ~~download and parse (data.py) list of JSON~~
- ~~make_dataset should return train, validate, test~~
- ~~validate is a randomly selected 10% of train.json, pre-shuffled [validation_portion=0.1 could be another default arg] (see python random library)~~
- ~~histogram of ingredients (from collections import Counter)~~
- ~~count of ingredients: 6496...4779 non-unique ingredients~~
- ~~count of recipes in each cuisine, count of cuisines~~
- ~~refactor one-hot encoding as function in data.py~~
- explore ingredients data to look for low-hanging fruit cleaning (example: are there ingredients that differ only by punctuation or capitalization?)
- add saving and loading of one-hot-encoded data

## Modeling
- ~~truncate list to top 20-200 ingredients~~
- ~~one-hot encode recipes~~
- ~~create list of encoded recipes and labels for train data~~
- ~~run and fit classifier~~
- ~~print predictions and percent correct~~
- print summary for training data (to better monitor overfitting)
- ~~find the point at which additional ingredients are no longer helping the model (400 ingredients: 66.6 correct, 800: 64% 2000: 66.7, 200: .612) 400 seems to be the sweet spot~~
- ~~try three different non-neural-net sklearn models to determine what performs best. Each model should be called via a short function in model.py

linear kernal svc: Correct: 2611
Incorrect: 1366
Success rate: 0.6565250188584361

random forest (much faster than svc):
Correct: 1015
Incorrect: 2962
Success rate: 0.25521750062861454

adaboost:
Correct: 2046
Incorrect: 1931
Success rate: 0.5144581342720643

RandomForestClassifier(max_depth=20, n_estimators=200, max_features='sqrt'):
Correct: 2216
Incorrect: 1761
Success rate: 0.5572039225546894

GradientBoostingClassifier:
Correct: 2564
Incorrect: 1413
Success rate: 0.6447070656273574

GradientBoostingClassifierTuned:
Correct: 2738
Incorrect: 1239
Success rate: 0.6884586371636913~~

- ~~Write a function which, given an (unpickled) model, an encoder, a list of ingredients (unencoded), returns the model prediction, which should be in a form of a sorted list like `[('Italian', 0.78), ('British', 0.12), ... , ('Indian':0.001)]`~~
- (option a) Try an sklearn MLP regressor
- (option b) Install Tensorflow and get more ToDos

## Presentation
- task a
- task b
- task c

## Feature Imaginarium
- modify the sklearn functions to have a save parameter which pickles the model once it's been trained. These models to save should also have probability set to True, to give more interesting info when used in the Presentation tasks.
- Figure out how to pickle and unpickle one-hot feature encoders, since the model pickling isn't too useful without knowing the associated encoder. One solution is for the models to be passed unencoded data and the parameters with which to encode, then they can create the one-hot encoding and save it alongside the model itself. (One-hot cuisine encoding will be the same every time and can be done from scratch every time.)
- Write a function which unpickles and returns any of the sklearn models plus an encoder, given a path to a pickled model and a path to an encoder.
