# Import numerical and data processing libraries

import numpy as np

import pandas as pd



# Import helpers that make it easy to do cross-validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# Import machine learning models

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



# Import visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline



# Import a method in order to make deep copies

from copy import deepcopy



# Import an other usefull libraries

import itertools



# Set the paths for inputs and outputs

local = 0

if(local == 0):

    inputPath = "../input/"

    outputPath = "../output/"

else:

    inputPath = "data/"

    outputPath = "data/"
# This creates a pandas dataframe and assigns it to the titanic variable

titanicOrigTrainDS = pd.read_csv(inputPath + "train.csv")

titanicTrainDS = deepcopy(titanicOrigTrainDS)



titanicOrigTestDS = pd.read_csv(inputPath + "test.csv")

titanicTestDS = deepcopy(titanicOrigTestDS)



# Print the first five rows of the dataframe

titanicTrainDS.head(5)
# What is the percentage of survival by class (1st, 2nd, 3rd)?

titanicTrainDS[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean() 



# We find a big variability. The first class passengers had definetely more chances to survive. 

# This means that "Pclass" is an important feature.
# What is the percentage of survival by sex?

titanicTrainDS[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()



# We find a huge variability. Woman had more chances to survive. 

# This is definitely an important feature.
# What is the percentage of survival according to the port of embarkation

titanicTrainDS[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
# What is the percentage of survival according to the number of siblings?

titanicTrainDS[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
# What is the percentage of survival according to the number of parents?

titanicTrainDS[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
# What is the percentage of survival according to the age (grouped)?

interval = 10

TempV = round(titanicTrainDS["Age"]//interval)*interval

titanicTrainDS["AgeIntervalMin"] = TempV

titanicTrainDS["AgeIntervalMax"] = TempV + interval

titanicTrainDS[["AgeIntervalMin", "AgeIntervalMax", "Survived"]].groupby(["AgeIntervalMin"], as_index=False).mean()
# What is the percentage of survival according to the fare (grouped)?

interval = 25

TempV = round(titanicTrainDS["Fare"]//interval)*interval

titanicTrainDS["FareIntervalMin"] = TempV

titanicTrainDS["FareIntervalMax"] = TempV + interval

titanicTrainDS[["FareIntervalMin", "FareIntervalMax", "Survived"]].groupby(["FareIntervalMin"], as_index=False).mean()
titanicDSs = [titanicTrainDS, titanicTestDS]
# lenght of the dataframe

len(titanicTrainDS)
# Summary on the dataframe

titanicTrainDS.describe()
# lenght of the dataframe

len(titanicTestDS)
# Summary on the dataframe

titanicTestDS.describe()
titanicTrainDS["AgeEmptyOrNot"] =  titanicTrainDS["Age"].apply(lambda x: 1 if x>=0  else 0)

titanicTrainDS[['Embarked', 'AgeEmptyOrNot']].groupby(['Embarked'], as_index=False).mean() 
titanicTrainDS[['Embarked', 'Age']].groupby(['Embarked'], as_index=False).mean() 
# Fill missing values with the median value

for dataset in titanicDSs:

    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())
# What are the values for this column?

for dataset in titanicDSs:

    print(dataset["Sex"].unique())
# Convert to numerical values

for dataset in titanicDSs:

    dataset.loc[dataset["Sex"] == "male", "Sex"] = 0

    dataset.loc[dataset["Sex"] == "female", "Sex"] = 1
# What are the values for this column?

for dataset in titanicDSs:

    print(dataset["Embarked"].unique())
# Fill missing values with most frequent value

mostFrequentOccurrence = titanicTrainDS["Embarked"].dropna().mode()[0]

titanicTrainDS["Embarked"] = titanicTrainDS["Embarked"].fillna(mostFrequentOccurrence)



# Convert to numerical values

for dataset in titanicDSs:

    dataset.loc[dataset["Embarked"] == "S", "Embarked"] = 0

    dataset.loc[dataset["Embarked"] == "C", "Embarked"] = 1

    dataset.loc[dataset["Embarked"] == "Q", "Embarked"] = 2
titanicTestDS["Fare"] = titanicTestDS["Fare"].fillna(titanicTestDS["Fare"].median())
# The columns that can be used in the prediction

predictorsAll = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] 



# Create all combinations of predictors

predictorCombinations = [] # all combination of predictord

for index in range(1, len(predictorsAll)+1):

    for subset in itertools.combinations(predictorsAll, index):

         predictorCombinations.append(list(subset))  

            

#predictorCombinations
# Function: Evaluate one algorithm type (and return n fitted algorithms)

# -input

    # predictorsDs: the dataset projected to the predictors of interest

    # targetDs: the target or label vector of interest (the column "Survived" in our work)

    # algModel: the "template" or model of the algorithm to apply

    # nbFK: the number of cross validation folders

# -output

    # algs: nbKF fitted algorithms 

    # accuracy: the evaluation of the accuracy

def binClassifModel_kf(predictorsDs, targetDs, algModel, nbKF):

    # List of algorithms

    algs = []

    

    # Generate cross-validation folds for the titanic data set

    # It returns the row indices corresponding to train and test

    # We set random_state to ensure we get the same splits every time we run this

    kf = KFold(nbKF, random_state=1)



    # List of predictions

    predictions = []



    for trainIndexes, testIndexes in kf.split(predictorsDs):

        # The predictors we're using to train the algorithm  

        # Note how we only take the rows in the train folds

        predictorsTrainDs = (predictorsDs.iloc[trainIndexes,:])

        # The target we're using to train the algorithm

        train_target = targetDs.iloc[trainIndexes]

        

        # Initialize our algorithm class

        alg = deepcopy(algModel)

        # Training the algorithm using the predictors and target

        alg.fit(predictorsTrainDs, train_target)

        algs.append(alg)

        

        # We can now make predictions on the test fold

        thisSlitpredictions = alg.predict(predictorsDs.iloc[testIndexes,:])

        predictions.append(thisSlitpredictions)





    # The predictions are in three separate NumPy arrays  

    # Concatenate them into a single array, along the axis 0 (the only 1 axis) 

    predictions = np.concatenate(predictions, axis=0)



    # Map predictions to outcomes (the only possible outcomes are 1 and 0)

    predictions[predictions > .5] = 1

    predictions[predictions <=.5] = 0

    accuracy = len(predictions[predictions == targetDs]) / len(predictions)

    

    # return the multiple algoriths and the accuracy

    return [algs, accuracy]
# Helper that return the indexed of the sorted list

def sort_list(myList):

    return sorted(range(len(myList)), key=lambda i:myList[i])



# Function: Run multiple evaluations for one algorithm type (one for each combination of predictors)

# -input

    # algModel: the "template" or model of the algorithm to apply

    # nbFK: the number of cross validation folders

# -output

    # {}

def getAccuracy_forEachPredictor(algModel, nbKF):

    accuracyList = []

    

    # For each combination of predictors

    for combination in predictorCombinations:

        result = binClassifModel_kf(titanicTrainDS[combination], titanicTrainDS["Survived"], algModel, nbKF)

        accuracy = result[1]

        accuracyList.append(accuracy)



    # Sort the accuracies

    accuracySortedList = sort_list(accuracyList)



    # Diplay the best combinations

    for i in range(-5, 0):

        print(predictorCombinations[accuracySortedList[i]], ": ", accuracyList[accuracySortedList[i]])

    #for elementIndex in sort_list(accuracyList1):

    #    print(predictorCombinations[elementIndex], ": ", accuracyList1[elementIndex])

        

    print("--------------------------------------------------")



    # Display the accuracy corresponding to combination that uses all the predictors

    lastIndex = len(predictorCombinations)-1

    print(predictorCombinations[lastIndex], ":", accuracyList[lastIndex])
algModel = LinearRegression(fit_intercept=True, normalize=True)

getAccuracy_forEachPredictor(algModel, 5)
algModel = LogisticRegression()

getAccuracy_forEachPredictor(algModel, 5)
algModel = GaussianNB()

getAccuracy_forEachPredictor(algModel, 5)
algModel = KNeighborsClassifier(n_neighbors=5)

getAccuracy_forEachPredictor(algModel, 5)
algModel = DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=2)

getAccuracy_forEachPredictor(algModel, 5)
algModel = RandomForestClassifier(n_estimators=100, min_samples_split=4, min_samples_leaf=2)

getAccuracy_forEachPredictor(algModel, 5)
# Run again the model with the tuned parameters on the dataset using the best combination of predictors

algModel = RandomForestClassifier(n_estimators=100, min_samples_split=4, min_samples_leaf=2)

predictors = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']

result = binClassifModel_kf(titanicTrainDS[predictors], titanicTrainDS["Survived"], algModel, 5)

algList = result[0] # the set of algorithms



predictionsList = []

for alg in algList:

    predictions = alg.predict(titanicTestDS[predictors])

    predictionsList.append(predictions)    



# There are different preditions, we take the mean (a voting-like system)

predictionsFinal = np.mean(predictionsList, axis=0)



# Map predictions to outcomes (the only possible outcomes are 1 and 0)

predictionsFinal[predictionsFinal > .5] = 1

predictionsFinal[predictionsFinal <=.5] = 0



# Cast as int

predictionsFinal = predictionsFinal.astype(int)
# Create a new dataset with only the id and the target column

submission = pd.DataFrame({

        "PassengerId": titanicTestDS["PassengerId"],

        "Survived": predictionsFinal

    })



#submission.to_csv(outputPath + 'submission.csv', index=False)