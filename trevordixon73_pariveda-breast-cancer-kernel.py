import numpy as np
import pandas as pd
import sklearn as skl
import sklearn.decomposition as dcmp
import sklearn.preprocessing as prpc
import seaborn as sns
import matplotlib.pyplot as plt
# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#Read in data, drop unnamed column, convert diagnoses to binary
allData = pd.read_csv('../input/data.csv')
allData = allData.drop(columns=["Unnamed: 32"])
allData.loc[ allData['diagnosis']  == 'M', 'diagnosis'] = 1
allData.loc[ allData['diagnosis']  == 'B', 'diagnosis'] = 0

#use skl to perform standardization scaling on the columns other than id and diagnosis
standardScaler = prpc.StandardScaler()
colsToScale = allData.drop(columns = ["id","diagnosis"])
x = colsToScale.values
x_scaled = pd.DataFrame(standardScaler.fit_transform(x), columns = colsToScale.columns)
scaledData = pd.concat([allData.loc[:,["id","diagnosis"]], x_scaled], axis=1)
scaledData.head()
#graph diagnosis histogram for each column
for column in list(scaledData.drop(columns=["id","diagnosis"]).columns.values):
    g = sns.FacetGrid(scaledData, col='diagnosis')
    g.map(plt.hist, column, bins=10)
#shuffle data and separate into training/testing
shuffledData = skl.utils.shuffle(scaledData, random_state=0)
trainData = pd.DataFrame(shuffledData.iloc[:400])
testData = pd.DataFrame(shuffledData.iloc[400:])
dataList = [trainData, testData]
allData.shape, trainData.shape, testData.shape
trainData = trainData.loc[:,['id','diagnosis','radius_mean', 'area_mean','concave points_worst', 'texture_mean']]
testData = testData.loc[:,['id','diagnosis','radius_mean', 'area_mean','concave points_worst', 'texture_mean']]
#examine diagnosis distributions
allNeg = allData.loc[allData['diagnosis'] == 0]
allPos = allData.loc[allData['diagnosis'] == 1]
trainNeg = trainData.loc[trainData['diagnosis'] == 0]
trainPos = trainData.loc[trainData['diagnosis'] == 1]
testNeg = testData.loc[testData['diagnosis'] == 0]
testPos = testData.loc[testData['diagnosis'] == 1]
allPosRate = len(allPos)/len(allNeg)
trainPosRate = len(trainPos)/len(trainNeg)
testPosRate = len(testPos)/len(testNeg)
print("All % Pos:", allPosRate, "\nTrain % Pos:", trainPosRate, "\nTest % Pos:", testPosRate)
train_input = trainData.drop(columns=["id","diagnosis"])
train_output = trainData.loc[:,'diagnosis']
test_input = testData.drop(columns=["id","diagnosis"])
test_output = testData.loc[:,"diagnosis"]
train_input.shape, train_output.shape, test_input.shape, test_output.shape
logreg = LogisticRegression()
logreg.fit(train_input, train_output)
acc_log = round(logreg.score(test_input, test_output) * 100, 2)
acc_log
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_input, train_output)
acc_knn = round(knn.score(test_input, test_output) * 100, 2)
acc_knn
test_predictions = logreg.predict(test_input)
ans = np.vstack((testData.loc[:,"id"],test_predictions, test_output))
ans = ans.transpose()
answer = pd.DataFrame(np.array(ans), columns=["id","predicted","actual"])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(answer)