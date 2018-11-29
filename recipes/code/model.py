from data import make_dataset, onehotEncode, onehotPickle, getIngredientLabels
#from numpy import array
from sklearn import datasets
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from collections import Counter

#personal utilities
from utils import indexInList
#classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#predictions
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier


classifierIndex = 7
ingredientCutoff = 1000 #6496 unique ingredients, 4779 non-unique ingredients

names = ["Nearest_Neighbors", "Linear_SVM", "RBF_SVM", "Scale_SVC", "Gaussia_Process",
         "Decision_Tree", "Random_Forest", "MLP_Classifier", "AdaBoost",
         "Naive_Bayes", "QDA", "Gradient_Boosting_Classifier", "GradientBoosting_Classifier_Tuned"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    SVC(gamma='scale', probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=20, n_estimators=200, max_features='sqrt'),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    GradientBoostingClassifier(),
    GradientBoostingClassifier(n_estimators=200, learning_rate=.2, max_depth=5, max_features='sqrt')]

print('Using classifier: {}_{}'.format(names[classifierIndex], ingredientCutoff))
savedModelPath = '../models/{}_{}.joblib'.format(names[classifierIndex], ingredientCutoff)

train, test, validate = make_dataset()
ingredientLabels = getIngredientLabels(train, ingredientCutoff)
cuisineLabels = set(Counter(recipe['cuisine'] for recipe in train))



#Takes an untrained classifier, a training set, file destination for pickled model, and file destination for pickled encoded recpe[s]
#Returns the trained model and encoded training set
def trainModel(model=classifiers[classifierIndex], trainingSet=train, pickledModelPath=savedModelPath):
    trainingRecipes = [recipe['ingredients'] for recipe in trainingSet]
    encodedRecipes = onehotEncode(trainingRecipes, ingredientLabels)
    cuisineLabels = list(recipe["cuisine"] for recipe in trainingSet)
    print("Training model...")
    model.fit(encodedRecipes, cuisineLabels)
    joblib.dump(model, pickledModelPath)
    return model

def loadModel(pickledModelPath=savedModelPath):
    print("Loading model...")
    trainedModel = joblib.load(pickledModelPath)
    return trainedModel

def validateModel(trainedModel, validationSet=validate, labels=ingredientLabels):
    print("Validating model...")
    validationRecipes = [recipe['ingredients'] for recipe in validationSet]
    encodedValidationRecipes = onehotEncode(validationRecipes, labels)
    print("making predictions for validation...")
    predictedCuisines = trainedModel.predict(encodedValidationRecipes)
    actualCuisines = list(recipe["cuisine"] for recipe in validationSet)
    print("Validation set results:")
    print("************************************")

    print("Classification report for classifier %s:\n%s\n"
      % (trainedModel, metrics.classification_report(actualCuisines, predictedCuisines)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(actualCuisines, predictedCuisines))
    correctGuesses = 0
    for index,prediction in enumerate(predictedCuisines):
#   print("Actual/Predicted: {}/{}".format(expected[index], prediction))
        if(actualCuisines[index]==prediction):
            correctGuesses+=1
    print("Correct: {}".format(correctGuesses))
    print("Incorrect: {}".format((len(predictedCuisines)-correctGuesses)))
    print("Success rate: {}".format(correctGuesses/len(predictedCuisines)))

def predictCuisine(trainedModel, recipe, iLabels=ingredientLabels):
    print("Predicting Cuisine...")
    encodedPredictionRecipe = onehotEncode(recipe, iLabels)
    #print(encodedPredictionRecipe)
    #cuisineLabels = list(recipe["cuisine"] for recipe in validate)
    #onehotRecipes = onehotEncode(recipe, ingredientLabels)
    #print(onehotRecipe)
    print("testing recipe: {}".format(recipe))
    #predictions = trainedModel.predict_proba(encodedPredictionRecipe)[:,1]
    return(m.predict_proba(encodedPredictionRecipe))

m = trainModel()
#m = loadModel()
validateModel(m, validate)
#recipesToPredict = [recipe['ingredients'] for recipe in validate[0]]

predictionRecipe = [["tortilla", "ground beef", "cheddar cheese", "oil", "cumin", "salt", "serrano chiles", "sour cream", "cilantro"]]
#predictionRecipe = [["eggplant", "cumin", "ginger", "onion"]]
print("Possible cuisines: {}".format(cuisineLabels))
predicted_cuisine = predictCuisine(m, predictionRecipe)
results = list()
cl = list(cuisineLabels)
for i, p in enumerate(predicted_cuisine[0]):
    results.append([cl[i], p])
results.sort(key=lambda tup: tup[1], reverse=True)


print(results)


#for p, index in predicted_cuisine[0]:
#    print('{}: {}%'.format(cuisineLabels[index]), p)

#get count of ingredients
#total = [recipe['ingredients'] for recipe in train]
#total = [cat for sublist in total for cat in sublist] #flatten the list
#ingredientCounter = Counter(total)
#print("~~~~~~~~~~~~~~INGREDIENTS~~~~~~~~~~~~~~~~~~~~~~")
#print("{} ingredients found...".format(len(ingredientCounter)))
#ingredientCounter = {k:v for k,v in ingredientCounter.iteritems() if v > 1}
#print("{} non-unique ingredients found...".format(len(ingredientCounter)))
#for k, v in ingredientCounter.most_common():
#    print("{}:  {}".format(k, v))

#print("~~~~~~~~~~~~~~TOP INGREDIENTS~~~~~~~~~~~~~~~~~~~~~~")
#print("Showing the top {} most common ingredients...".format(ingredientCutoff))
#topIngredientsCount = list(ingredientCounter.most_common())[:ingredientCutoff]
#ingredientLabels = list()
#for k, i in topIngredientsCount:
#    print("{}:  {} uses".format(k, v))
    #exclude unique items
#    if(i>1):
#        ingredientLabels.append(k)
#print("{} non-unique ingredients".format(len(ingredientLabels)))
#4779 non-unique ingredients




#get count of cuisines
#cuisineCounter = Counter(recipe['cuisine'] for recipe in train)
#print("~~~~~~~~~~~~~~CUISINES~~~~~~~~~~~~~~~~~~~~~~")
#print("{} cuisines found...".format(len(cuisineCounter)))
#for k, v in cuisineCounter.most_common():
#    print("{}:  {} recipes".format(k, v))



#create recipe vectors of 1hot-encoded ingredients (binary vecs with a 1 at the index of each ingredient)
#onehotIngredients = []
#for index, value in enumerate(ingredientLabels):
#	encodedIngredient = [0 for _ in range(ingredientCutoff)]
#	encodedIngredient[index] = 1
#	onehotIngredients.append(encodedIngredient)
#print(onehotIngredients)

#ingredientLabels = getIngredientLabels(train)
#cuisineLabels = list(recipe["cuisine"] for recipe in train)


#trainingRecipes = [recipe['ingredients'] for recipe in train]
#onehotTrainingRecipes = onehotEncode(trainingRecipes, ingredientLabels)



#validationRecipes = [recipe['ingredients'] for recipe in validate]
#onehotValidationRecipes = onehotEncode(validationRecipes, ingredientLabels)

#onehotTrainingPickle = onehotPickle(validationRecipes, ingredientLabels, "onhotTrainingRecipes.joblib")
#onehotVaidationPickle = onehotPickle(validationRecipes, ingredientLabels, "onhotValidationRecipes.joblib")
#print(joblib.load("onhotValidationRecipes.joblib"))


#get labels for classifier
#cuisineLabels = list(recipe["cuisine"] for recipe in train)
#expected = list(recipe["cuisine"] for recipe in validate)

#setup classifier and fit data
#classifier = svm.SVC(gamma='scale')

#classifier = classifiers[classifierIndex]
#classifier.fit(onehotTrainingRecipes, cuisineLabels)

#save model to file
#joblib.dump(classifier, 'pickledClassifier.joblib')

#load pickled classifiers
#classifier = joblib.load('pickledClassifier.joblib')

#predict cuisine from onehotRecipes
#print("predition should be:")
#print(expected)

#predicted = classifier.predict(onehotValidationRecipes)
#print("Validation set results:")
#print("************************************")

#print("Classification report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(expected, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

#correctGuesses = 0
#for index,prediction in enumerate(predicted):
#    print("Actual/Predicted: {}/{}".format(expected[index], prediction))
#    if(expected[index]==prediction):
#        correctGuesses+=1
#print("Correct: {}".format(correctGuesses))
#print("Incorrect: {}".format((len(predicted)-correctGuesses)))
#print("Success rate: {}".format(correctGuesses/len(predicted)))


#portion = classifier.predict(onehotPortion)
#print("Checking portion of train data for overfitting. Report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(expected, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


#predict cuisine from onehotRecipes
#print("predition should be:")
#print(expected)



#predicted = classifier.predict(onehotTrainingRecipes)
#print("Training set (Overfit reference) results:")
#print("*****************************************")

#print("Classification report for classifier %s:\n%s\n"
#      % (classifier, metrics.classification_report(cuisineLabels, predicted)))
#print("Confusion matrix:\n%s" % metrics.confusion_matrix(cuisineLabels, predicted))

#correctGuesses = 0
#for index,prediction in enumerate(predicted):
#    print("Actual/Predicted: {}/{}".format(expected[index], prediction))
#    if(cuisineLabels[index]==prediction):
#        correctGuesses+=1
#print("Correct: {}".format(correctGuesses))
#print("Incorrect: {}".format((len(predicted)-correctGuesses)))
#print("Success rate: {}".format(correctGuesses/len(predicted)))
