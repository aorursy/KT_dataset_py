import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
rawData = pd.read_csv("../input/weatherAUS.csv")
rawData.head()
rawData.shape
rawData.isnull().sum()
workingData = rawData.drop(['Evaporation', 'Location', 'Date', 'Sunshine', 'Cloud9am', 'Cloud3pm','RISK_MM', 'RainTomorrow'], axis=1).copy()
Y = rawData['RainTomorrow'].copy()
#We will take this opportunity to also map our catagorical values.
Y = Y.map({'Yes':1, 'No':0})
Y = pd.DataFrame(Y, columns = ['RainTomorrow'])
workingData['RainToday'] = workingData['RainToday'].map({'Yes':1, 'No':0})
workingData.isnull().sum()
workingData = workingData.drop(['Pressure9am', 'Pressure3pm','WindDir9am'], axis = 1)

#Fill the missing data with numerical averages.
meanMatrix = workingData.drop(['WindDir3pm', 'RainToday', 'WindGustDir'], axis=1).copy()
meanWorkingData = meanMatrix.fillna(meanMatrix.mean())
#We will replace the observations that don't have a reading for the 'RainToday' feature with the most popular observation for that feature.
rawData['RainToday'].value_counts()
#So we will replace all missing RainToday valuse with 'No':0
meanWorkingData['RainToday'] = workingData['RainToday'].fillna(0)
#Brief investigation.
meanWorkingData.head(10)
#Ensure all Nulls are gone
meanWorkingData.isnull().sum()
from sklearn import preprocessing
xScaled = preprocessing.scale(meanWorkingData.drop(['RainToday'], axis = 1))



#xScaled['RainToday'] = meanWorkingData['RainToday'].values
xScaled = pd.DataFrame(xScaled, columns=meanWorkingData.drop(['RainToday'], axis =1).columns)
xScaled['RainToday'] = meanWorkingData['RainToday']
#We will briefly visualise the correlation matrix to investigate if there are (m)any relationships. We can expect there to be temperature relationships, since those values will remain more or less close to one another, similar with wind speeds, since they should not change drastically in a daily observation.
sns.heatmap(data = xScaled.corr(), cmap = 'Blues')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

nFeatures = [1, 3, 5, 7, 9, 11]
xSelectors = []
xFeatures = []

for feature in nFeatures:
    xSelectionModel = SelectKBest(f_classif, k=feature).fit(xScaled, Y['RainTomorrow'])
    xSelectors.append(xSelectionModel)
    
for selector in xSelectors:
    featureList = xScaled.columns[selector.get_support()]
    xFeatures.append(featureList)
from sklearn.base import clone as skClone

def produceModels ( xElements, yElements, featureList, modelObject ):
    output = []
    for featureset in featureList:
        print('Generating model.')
        #Filter the xElements to the appropriate values in the xSelector
        xFilteredElements = xElements[featureset]
        print(xFilteredElements.shape)
        classifierModel = skClone(modelObject)
        classifierModel = classifierModel.fit(xFilteredElements, yElements['RainTomorrow'])
        output.append(classifierModel)
        print('Done.')
        #print('Coefficients: {0}'.format(classifierModel.coef_))
    return output

def compareModels ( xElements, yElements, featureList, modelList, showCoeffs = False):
    print('Comparing models.')
    iterator = 0
    for model in modelList:
        #Filter the xElements to the appropriate values in the xSelector
        xFilteredElements = xElements[featureList[iterator]]
        if showCoeffs:
            print(model.coef_)
        print('Comparing model with {0} features:'.format(nFeatures[iterator]))
        print('Accuracy : {0}'.format(model.score(xFilteredElements, yElements)))
        iterator += 1
    return

######## Is there a good way of comparing the predicting features similar to an adjusted r-squared or p-value for each of the predictors? Please let me know if you can think of a good guide on how to do this for future reference. Thanks.
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xScaled, Y, test_size = 0.2, random_state = 28122018)
from sklearn.linear_model import LogisticRegression
logisticRegressor = LogisticRegression(random_state = 28122018, solver='lbfgs')
logisticModels = produceModels(xTrain, yTrain, xFeatures, logisticRegressor)
compareModels(xTest, yTest, xFeatures, logisticModels)
from sklearn import tree
decisionTreeClassifier = tree.DecisionTreeClassifier(random_state = 28122018)
treeModels = produceModels(xTrain, yTrain, xFeatures, decisionTreeClassifier)
compareModels(xTest, yTest, xFeatures, treeModels)
from sklearn.neighbors import KNeighborsClassifier as KNC
#Using 5 neighbors
print('KNN (5)')
KNNClassifier5 = KNC(n_neighbors=5)
KNNModels5 = produceModels(xTrain, yTrain, xFeatures, KNNClassifier5)
compareModels(xTest, yTest, xFeatures, KNNModels5)

#Using 7 neighbors
print('KNN (7)')
KNNClassifier7 = KNC(n_neighbors=7)
KNNModels7 = produceModels(xTrain, yTrain, xFeatures, KNNClassifier7)
compareModels(xTest, yTest, xFeatures, KNNModels7)

#Using 10 neighbors
print('KNN (10)')
KNNClassifier10 = KNC(n_neighbors=10)
KNNModels10 = produceModels(xTrain, yTrain, xFeatures, KNNClassifier10)
compareModels(xTest, yTest, xFeatures, KNNModels10)
