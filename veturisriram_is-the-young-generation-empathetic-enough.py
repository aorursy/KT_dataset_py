# Import statements
import copy
import time
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC, SVR
from IPython.display import display
from scipy.stats.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import VarianceThreshold, chi2, RFE, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, GridSearchCV

# Import the datasets (responses and columns).
responsesData = pd.read_csv('../input/responses.csv')
columnsData = pd.read_csv('../input/columns.csv')

print("Datasets loaded!")
print("Shape of the data : {0} and {1}".format(responsesData.shape, columnsData.shape))
# Some styling..
pd.set_option('display.max_columns',150)
pd.set_option('display.max_rows',1010)
plt.style.use('bmh')
print("Responses Data")
responsesData.head(n=3)
print("Columns Data")
columnsData.head(n=3)
# Pre processing the data set provided
def preprocessingDataset(dataset):

    # Define imp from Imputer class for missing values
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    #### Preprocessing the Dataset
    music = dataset.iloc[:, 0:19]
    movies = dataset.iloc[:, 19:31]
    phobias = dataset.iloc[:, 63:73]
    interests = dataset.iloc[:, 31:63]
    health = dataset.iloc[:, 73:76]
    personal = dataset.iloc[:, 76:133]
    information = dataset.iloc[:, 140:150]
    expenditure = dataset.iloc[:, 133:140]

    """
    print(music)
    print(movies)
    print(phobias)
    print(interests)
    print(health)
    print(personal)
    print(information)
    print(spendings)
    """
    # Processing the personal
    for x in personal["Lying"]:
        if x == "never":
            personal.replace(x, 1.0, inplace=True)
        elif x == "only to avoid hurting someone":
            personal.replace(x, 2.0, inplace=True)
        elif x == "sometimes":
            personal.replace(x, 3.0, inplace=True)
        elif x == "everytime it suits me":
            personal.replace(x, 4.0, inplace=True)
        elif x == "Nan":
            personal.replace(x, np.nan, inplace=True)
        elif x == "nan":
            personal.replace(x, np.nan, inplace=True)

    for x in personal["Punctuality"]:
        if x == "i am often early":
            personal.replace(x, 3.0, inplace=True)
        elif x == "i am always on time":
            personal.replace(x, 2.0, inplace=True)
        elif x == "i am often running late":
            personal.replace(x, 1.0, inplace=True)
        elif x == "Nan":
            personal.replace(x, np.nan, inplace=True)
        elif x == "nan":
            personal.replace(x, np.nan, inplace=True)

    for x in personal["Internet usage"]:
        if x == "most of the day":
            personal.replace(x, 4.0, inplace=True)
        elif x == "few hours a day":
            personal.replace(x, 3.0, inplace=True)
        elif x == "less than an hour a day":
            personal.replace(x, 2.0, inplace=True)
        elif x == "no time at all":
            personal.replace(x, 1.0, inplace=True)
        elif x == "Nan":
            personal.replace(x, np.nan, inplace=True)
        elif x == "nan":
            personal.replace(x, np.nan, inplace=True)

    # Replace strings with numpy NaNs
    personal = personal.replace("NaN", np.nan)
    personal = personal.replace("nan", np.nan)

    # Replace missing values with most frequent values
    imp.fit(personal)
    personal_data = imp.transform(personal)

    d = personal_data[:, :]
    ind = []
    for x in range(len(personal_data)):
        ind.append(x)
    c = personal.columns.tolist()
    personal = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing the health
    for x in health["Smoking"]:
        if x == "current smoker":
            health.replace(x, 1.0, inplace=True)
        elif x == "former smoker":
            health.replace(x, 2.0, inplace=True)
        elif x == "tried smoking":
            health.replace(x, 3.0, inplace=True)
        elif x == "never smoked":
            health.replace(x, 4.0, inplace=True)
        elif x == "Nan":
            health.replace(x, np.nan, inplace=True)
        elif x == "nan":
            health.replace(x, np.nan, inplace=True)

    for x in health["Alcohol"]:
        if x == "drink a lot":
            health.replace(x, 1.0, inplace=True)
        elif x == "social drinker":
            health.replace(x, 2.0, inplace=True)
        elif x == "never":
            health.replace(x, 3.0, inplace=True)
        elif x == "Nan":
            health.replace(x, np.nan, inplace=True)
        elif x == "nan":
            health.replace(x, np.nan, inplace=True)

    # Replace strings with numpy NaNs
    health = health.replace("NaN", np.nan)
    health = health.replace("nan", np.nan)

    # Replace missing values with most frequent values
    imp.fit(health)
    healthData = imp.transform(health)
    d = healthData[:, :]
    ind = []
    for x in range(len(healthData)):
        ind.append(x)
    c = health.columns.tolist()
    health = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing the information
    for x in information["Gender"]:
        if x == "female":
            information.replace(x, 2.0, inplace=True)
        elif x == "male":
            information.replace(x, 1.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["Left - right handed"]:
        if x == "right handed":
            information.replace(x, 1.0, inplace=True)
        elif x == "left handed":
            information.replace(x, 2.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["Education"]:
        if x == "doctorate degree":
            information.replace(x, 6.0, inplace=True)
        elif x == "masters degree":
            information.replace(x, 5.0, inplace=True)
        elif x == "college/bachelor degree":
            information.replace(x, 4.0, inplace=True)
        elif x == "secondary school":
            information.replace(x, 3.0, inplace=True)
        elif x == "primary school":
            information.replace(x, 2.0, inplace=True)
        elif x == "currently a primary school pupil":
            information.replace(x, 1.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["Only child"]:
        if x == "yes":
            information.replace(x, 1.0, inplace=True)
        elif x == "no":
            information.replace(x, 2.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["Village - town"]:
        if x == "village":
            information["Village - town"].replace(x, 1.0, inplace=True)
        elif x == "city":
            information["Village - town"].replace(x, 2.0, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    for x in information["House - block of flats"]:
        if x == "block of flats":
            information["House - block of flats"].replace(x, 1, inplace=True)
        elif x == "house/bungalow":
            information["House - block of flats"].replace(x, 2, inplace=True)
        elif x == "Nan":
            information.replace(x, np.nan, inplace=True)
        elif x == "nan":
            information.replace(x, np.nan, inplace=True)

    information = information.replace("nan", np.nan)
    information = information.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(information)
    informationData = imp.transform(information)
    d = informationData[:, :]
    ind = []
    for x in range(len(informationData)):
        ind.append(x)
    c = information.columns.tolist()
    information = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing music
    music = music.replace("nan", np.nan)
    music = music.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(music)
    musicData = imp.transform(music)
    d = musicData[:, :]
    ind = []
    for x in range(len(musicData)):
        ind.append(x)
    c = music.columns.tolist()
    music = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing movies
    movies = movies.replace("nan", np.nan)
    movies = movies.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(movies)
    moviesData = imp.transform(movies)
    d = moviesData[:, :]
    ind = []
    for x in range(len(moviesData)):
        ind.append(x)
    c = movies.columns.tolist()
    movies = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing phobias
    phobias = phobias.replace("nan", np.nan)
    phobias = phobias.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(phobias)
    phobiasData = imp.transform(phobias)
    d = phobiasData[:, :]
    ind = []
    for x in range(len(phobiasData)):
        ind.append(x)
    c = phobias.columns.tolist()
    phobias = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing interests
    interests = interests.replace("nan", np.nan)
    interests = interests.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(interests)
    interestsData = imp.transform(interests)
    d = interestsData[:, :]
    ind = []
    for x in range(len(interestsData)):
        ind.append(x)
    c = interests.columns.tolist()
    interests = pd.DataFrame(data=d, index=ind, columns=c)

    # Processing spendings
    expenditure = expenditure.replace("nan", np.nan)
    expenditure = expenditure.replace("NaN", np.nan)

    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    imp.fit(expenditure)
    expenditureData = imp.transform(expenditure)
    d = expenditureData[:, :]
    ind = []
    for x in range(len(expenditureData)):
        ind.append(x)
    c = expenditure.columns.tolist()
    expenditure = pd.DataFrame(data=d, index=ind, columns=c)

    # Joining all the processed sections
    joinedDatasets = music.join(movies.join(phobias.join(interests.join(health.join(personal.join(information.join(expenditure)))))))

    return joinedDatasets
# Collect the dataset with missing values filled with the most frequent entry in the column. 

print("Preprocessing data might take some time..")
print("1) Missing values are being handled!")
print("2) Categorical entries are getting converted to numeric values!")
print("3) Dummy variables are being handled!")
filledData = preprocessingDataset(responsesData)
print("Done preprocessing data!")
filledData.head(n=3)
# Scale the dataset.
def scalingDataset(dataset):
    # Scaling the dataset
    scaler = StandardScaler()
    scaledDataarray = scaler.fit_transform(dataset)
    if type(dataset) is np.ndarray:
        return scaledDataarray
    else:
        d = scaledDataarray[:, :]
        ind = []
        for x in range(len(dataset)):
            ind.append(x)
        c = dataset.columns.tolist()
        scaledData = pd.DataFrame(data=d, index=ind, columns=c)
        return scaledData

scaledData = scalingDataset(filledData)
scaledData.head(n=3)
relations = filledData.corr()
# The line of code below is used from a question posted in StackOverflow
# Link : https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
multicollinearity = (relations.where(np.triu(np.ones(relations.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
# Top positive correlations are:
multicollinearity.head(n=15)
# Top negative correlations are:
multicollinearity.tail(n=15)
filledData.describe()
def correlationFigure(featureVariablesMain, targetVariable):
    # Calculate correlation
    #print(featureVariablesMain.columns)
    #print(targetVariable.values)
    def correlationCalculation(targetVariable, featureVariables, features):
        columns = [] # For maintaining the feature names
        values = [] # For maintaining the corr values of features with "Empathy"

        # Traverse through all the input features
        for x in features:
            if x is not None:
                columns.append(x) # Append the column name
                # Calculate the correlation
                c = np.corrcoef(featureVariables[x], featureVariables[targetVariable])
                absC = abs(c) # Absolute value because important values might miss
                values.append(absC[0,1])

        corrValues = pd.DataFrame()
        dataDict = {'features': columns, 'correlation_values': values}
        corrValues = pd.DataFrame(dataDict)
        # Sort the value by correlation values
        sortedCorrValues = corrValues.sort_values(by="correlation_values")

        # Plot the graph to show the features with their correlation values
        figure, ax = plt.subplots(figsize=(15, 45), squeeze=True)
        ax.set_title("Correlation Coefficients of Features")
        sns.barplot(x=sortedCorrValues.correlation_values, y=sortedCorrValues['features'], ax=ax)
        ax.set_ylabel("-----------Corr Coefficients--------->")


        plt.show()

        return sortedCorrValues

    # Make a list of columns
    columns = []
    for x in featureVariablesMain.columns:
        columns.append(x)
    # Remove "Empathy" from df
    columns.remove(targetVariable)

    # Compute correlations
    correlations = correlationCalculation(targetVariable, featureVariablesMain, columns)
    return correlations
# Plotting the correlations with respect to "Empathy" variable
target = "Empathy"
targetVariable = filledData['Empathy'].to_frame()
corrData = correlationFigure(scaledData, target)
importantFeatures = corrData.sort_values(by="correlation_values", ascending=True).tail(20)
# 20 most important feature with their correlation values
importantFeatures
finalColumnsList = []
for x in importantFeatures['features']:
    finalColumnsList.append(x)

df = pd.DataFrame() # Final prepared dataset for modelling
df = filledData[finalColumnsList[0]].to_frame()
for x in range(1, len(finalColumnsList)):
    df = df.join(filledData[finalColumnsList[x]].to_frame())
xTrain, xTest, yTrain, yTest = train_test_split(df, targetVariable, test_size=0.2, random_state=0)
xTrain = xTrain.sort_index()
xTest = xTest.sort_index()
yTrain = yTrain.sort_index()
yTest = yTest.sort_index()
# Decision Tree Modelling
def dt(xTrain, yTrain, xTest, yTest):
    print("Hyperparameter Tuning!")
    gridClassifier = DecisionTreeClassifier()
    depthList = [1,3,5,10]
    parameters = {'max_depth':depthList}
    gridSearch = GridSearchCV(estimator=gridClassifier,
                              param_grid=parameters,
                              scoring="accuracy",
                              cv=10,
                              n_jobs=5)
    gridSearch.fit(xTrain, yTrain.values.ravel())
    bestAccuracyMLP = gridSearch.best_score_
    bestParametersMLP = gridSearch.best_params_

    print("The best parameters for Decision Tree model are :\n{}\n".format(bestParametersMLP))
    dtclassifier = DecisionTreeClassifier(max_depth=3)
    dtclassifier.fit(xTrain, yTrain.values.ravel())
    yPredictiondtTest = dtclassifier.predict(xTest)
    yPredictiondtTrain = dtclassifier.predict(xTrain)
    print("Decision Tree Evaluations :\n")
    print("Training Accuracy => {}".format(accuracy_score(yTrain, yPredictiondtTrain) * 100))
    print("Testing Accuracy => {}\n".format(accuracy_score(yTest, yPredictiondtTest) * 100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTest, yPredictiondtTest)))
    print("Classification Summary => \n{}\n".format(classification_report(yTest, yPredictiondtTest)))
    plt.scatter(yTest, yPredictiondtTest)

dt(xTrain, yTrain, xTest, yTest)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

# Below function is taken from the official documentation of "sklearn" 
# Link : http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = DecisionTreeClassifier(max_depth=3)
plot_learning_curve(estimator, title, xTrain, yTrain, cv=cv, n_jobs=4)
# KNN Modelling
def knn(xTrain, yTrain, xTest, yTest):
    print("Hyperparameter Tuning!")
    gridClassifier = KNeighborsClassifier()
    nearestNeighbors = [1, 3, 5, 10]
    parameters = {'n_neighbors': nearestNeighbors}
    gridSearch = GridSearchCV(estimator=gridClassifier,
                              param_grid=parameters,
                              scoring="accuracy",
                              cv=10,
                              n_jobs=5)
    gridSearch.fit(xTrain, yTrain.values.ravel())
    bestAccuracyMLP = gridSearch.best_score_
    bestParametersMLP = gridSearch.best_params_

    print("The best parameters for KNN model are :\n{}\n".format(bestParametersMLP))
    knnClassifier = KNeighborsClassifier(n_neighbors=1)
    knnClassifier.fit(xTrain, yTrain.values.ravel())
    yPredKNNTest = knnClassifier.predict(xTest)
    yPredKNNTrain = knnClassifier.predict(xTrain)
    print("KNN Evaluation :\n")
    print("Training Accuracy => {}".format(accuracy_score(yTrain, yPredKNNTrain) * 100))
    print("Testing Accuracy => {}\n".format(accuracy_score(yTest, yPredKNNTest) * 100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTest, yPredKNNTest)))
    print("Classification Summary => \n{}\n".format(classification_report(yTest, yPredKNNTest)))
    plt.scatter(yTest, yPredKNNTest)

knn(xTrain, yTrain, xTest, yTest)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

# Below function is taken from the official documentation of "sklearn" 
# Link : http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = KNeighborsClassifier(n_neighbors=1)
plot_learning_curve(estimator, title, xTrain, yTrain, cv=cv, n_jobs=4)
# Logistics Regression Modelling
def logisticRegression(xTrain, yTrain, xTest, yTest):
    logRegClassifier = LogisticRegression(multi_class='ovr', random_state=0, C=3)
    logRegClassifier.fit(xTrain, yTrain.values.ravel())
    yPredLogTest = logRegClassifier.predict(xTest)
    yPredLogTrain = logRegClassifier.predict(xTrain)
    print("Logistic Regression Evaluation :\n")
    print("Testing Accuracy => {}".format(accuracy_score(yTest, yPredLogTest) * 100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTest, yPredLogTest)))
    print("Classification Summary => \n{}\n".format(classification_report(yTest, yPredLogTest)))
    plt.scatter(yTest, yPredLogTest)


logisticRegression(xTrain, yTrain, xTest, yTest)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

# Below function is taken from the official documentation of "sklearn" 
# Link : http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = LogisticRegression(multi_class='ovr', random_state=0, C=3)
plot_learning_curve(estimator, title, xTrain, yTrain, cv=cv, n_jobs=4)
def kBestLogReg(filledData):
    bestDF = filledData.drop(columns=['Empathy'], axis=1)
    targetVariable = filledData['Empathy'].to_frame()
    selected = SelectKBest(score_func=f_classif, k=20)
    selectedFit = selected.fit(bestDF, targetVariable.values.ravel())
    selectedFitTransform = selectedFit.transform(bestDF)

    xTrainBestK, xTestBestK, yTrainBestK, yTestBestK = train_test_split(
        scalingDataset(selectedFitTransform),
        targetVariable,
        test_size=0.2,
        random_state=0)
    print("Cross Validating for best parameters..")
    print("This might take some time..\n")
    lr = LogisticRegression(multi_class='ovr')
    cList = [10, 100, 1000, 10000]
    solverList = ['lbfgs', 'sag', 'saga', 'newton-cg']
    maxIterList = [100, 1000, 10000]
    parameters = {'C': cList, 'solver': solverList, 'max_iter': maxIterList}
    gridSearch = GridSearchCV(estimator=lr,
                              param_grid=parameters,
                              scoring="accuracy",
                              cv=10,
                              n_jobs=4)
    gridSearch.fit(xTrainBestK, yTrainBestK.values.ravel())
    bestAccuracyLogBestK = gridSearch.best_score_
    bestParametersLogBestK = gridSearch.best_params_
    print("The best parameters for Logistic Regression model are :\n{}\n".format(bestParametersLogBestK))
    # Best parameters : C:10, maxiter:100, solver:sag
    lr = LogisticRegression(C=10, max_iter=100, solver='lbfgs', multi_class='ovr', random_state=1)
    lr.fit(xTrainBestK, yTrainBestK.values.ravel())
    yPredLogBestTest = lr.predict(xTestBestK)
    bestKLogAcc = accuracy_score(yTestBestK, yPredLogBestTest)
    print("Logistic Regression using SelectKBest Method Evaluations :\n")
    print("Training Accuracy : {}".format(bestAccuracyLogBestK*100))
    print("Testing Accuracy  : {}\n".format(bestKLogAcc*100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTestBestK, yPredLogBestTest)))
    print("Classification Summary => \n{}\n".format(classification_report(yTestBestK, yPredLogBestTest)))
    plt.scatter(yTestBestK, yPredLogBestTest)

kBestLogReg(filledData)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

# Below function is taken from the official documentation of "sklearn" 
# Link : http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = LogisticRegression(C=10, max_iter=100, solver='lbfgs', multi_class='ovr', random_state=1)
plot_learning_curve(estimator, title, xTrain, yTrain, cv=cv, n_jobs=4)
def mlp(df, scaledData, targetVariable):

    # For MLP, we have to use scaled data
    scaledDF = scalingDataset(df)
    xTrain, xTest, yTrain, yTest = train_test_split(scaledData, targetVariable, test_size=0.2, random_state=0, shuffle=False)
    xTrain = xTrain.sort_index()
    xTest = xTest.sort_index()
    yTrain = yTrain.sort_index()
    yTest = yTest.sort_index()
    
    # You can run the below commented code to cross validate for the best parameters
    # print("SIT BACK AND RELAX! CROSS VALIDATION WOULD TAKE SOME TIME...")
    # mlpClass = MLPClassifier(random_state=0)
    # iterList = [100, 500]
    # hiddenLayerList = [(100, 100), (100, 200)]
    # parameters = {'alpha': 10.0 ** -np.arange(1, 7), 'max_iter': iterList, 'hidden_layer_sizes': hiddenLayerList}
    # gridSearch = GridSearchCV(estimator=mlpClass,
    #                           param_grid=parameters,
    #                           scoring="accuracy",
    #                           cv=10,
    #                           n_jobs=10)
    # gridSearch.fit(xTrain, yTrain.values.ravel())
    # bestAccuracyMLP = gridSearch.best_score_
    # bestParametersMLP = gridSearch.best_params_
    # print("The best parameters for MLP model are :\n{}\n".format(bestParametersMLP))

    mlpClass = MLPClassifier(hidden_layer_sizes=(100, 200), alpha=0.1, max_iter=500, random_state=0)
    mlpClass.fit(xTrain, yTrain.values.ravel())
    yPredMLP = mlpClass.predict(xTest)
    print("Multilayer Perceptron Evaluation :\n")
    # print("Training Accuracy => {}".format(bestAccuracyMLP*100))
    print("Testing Accuracy => {}\n".format(accuracy_score(yTest, yPredMLP)*100))
    print("Confusion Matrix => \n{}\n".format(confusion_matrix(yTest, yPredMLP)))
    print("Classification Summary => \n{}\n".format(classification_report(yTest, yPredMLP)))
    plt.scatter(yTest, yPredMLP)

mlp(df, scaledData, targetVariable)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

# Below function is taken from the official documentation of "sklearn" 
# Link : http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=10,
                        n_jobs=10, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


title = "Learning Curves"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
xTrain, xTest, yTrain, yTest = train_test_split(scaledData, targetVariable, test_size=0.2, random_state=0, shuffle=False)
xTrain = xTrain.sort_index()
xTest = xTest.sort_index()
yTrain = yTrain.sort_index()
yTest = yTest.sort_index()

estimator = MLPClassifier(hidden_layer_sizes=(100, 200), alpha=0.1, max_iter=500)
plot_learning_curve(estimator, title, xTrain, yTrain, cv=10, n_jobs=10)