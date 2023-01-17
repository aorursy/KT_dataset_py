import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

df = pd.read_csv("../input/pulsar_stars.csv")
df.info()
df.head()
x_data = df.drop(["target_class"], axis = 1)

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

y = df.target_class.values



compareScore = []
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

def initializeWeightBias(dimension):

    w = np.full((dimension,1), 0.01)

    b = 0.0

    return w, b



def sigmoidFunc(z):

    y_head = 1 / (1 + np.exp(-z))

    return y_head

    
def fwPropagation(w, b, x_train, y_train):

    #forward

    z = np.dot(w.T, x_train.T) + b

    y_head = sigmoidFunc(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss)) / x_train.shape[0]

    

    #backward

    deriWeight = (np.dot(x_train.T,((y_head-y_train.T).T)))/x_train.shape[0] 

    deriBias = np.sum(y_head-y_train.T)/x_train.shape[0]

    graDescents = {"deriWeight" : deriWeight, "deriBias" : deriBias}

    

    return graDescents, cost

    
def update(iterNumber, w, b, x_train, y_train, learningRate) :

    costList = []

    index = []

    for i in range(iterNumber + 1):

        graDescents, cost = fwPropagation(w, b, x_train, y_train)

        w = w - learningRate*graDescents["deriWeight"]

        b = b - learningRate*graDescents["deriBias"]

        

        if i % 10 == 0:

            costList.append(cost)

            index.append(i)

            print("Cost after {} iteration = {}".format(i, cost))

            

    parameters = {"weight" : w, "bias" : b}

    

    return parameters, costList, index     
def plotGraph(index, costList):

    plt.plot(index, costList)

    plt.ylabel("Cost")

    plt.show()

    
def predict(w, b, x_test):

    z = np.dot(w.T, x_test.T) + b

    y_head = sigmoidFunc(z)

    yPrediction = np.zeros((1,x_test.shape[0]))

    

    for i in range(y_head.shape[1]):

        if y_head[0,i] <= 0.5:

            yPrediction[0,i] = 0

        else:

            yPrediction[0,i] = 1

            

    return yPrediction
def logisticRegression(x_train, y_train, x_test, y_test, iterNumber, learningRate):

    dimension = x_train.shape[1]

    w, b = initializeWeightBias(dimension)

    parameters, costList, index = update(iterNumber, w, b, x_train, y_train, learningRate)

    

    predictionTest = predict(parameters["weight"], parameters["bias"], x_test)

    

    #printing errors

    print("Test accuracy: {} %".format(100 - np.mean(np.abs(predictionTest - y_test)) * 100))

    

    plotGraph(index, costList)

    

logisticRegression(x_train, y_train, x_test, y_test, iterNumber = 30, learningRate = 0.5)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train, y_train)



lrScore = lr.score(x_test, y_test) * 100

compareScore.append(lrScore)



print("Test accuracy: {} %".format(lrScore))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)



knn.fit(x_train, y_train)

print("Test accuracy: {}%".format(knn.score(x_test,y_test)*100))

scoreList = []

n = 15

for i in range(1,n):

    knn2 = KNeighborsClassifier(n_neighbors=i)

    knn2.fit(x_train, y_train)

    scoreList.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,n), scoreList)

plt.ylabel("Accuracy rate")

plt.show()
knn = KNeighborsClassifier(n_neighbors=8)



knn.fit(x_train, y_train)



knnScore = knn.score(x_test,y_test)*100

compareScore.append(knnScore)



print("Test accuracy: {}%".format(knnScore))
from sklearn.svm import SVC

svm = SVC(random_state=42, gamma = "scale")



svm.fit(x_train, y_train)



svmScore = svm.score(x_test,y_test)*100

compareScore.append(svmScore)



print("Test accuracy: {}%".format(svmScore))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()



nb.fit(x_train, y_train)



nbScore = nb.score(x_test,y_test)*100

compareScore.append(nbScore)



print("Test accuracy: {}%".format(nbScore))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42, max_depth=4)



dt.fit(x_train, y_train)



dtScore = dt.score(x_test, y_test)*100

compareScore.append(dtScore)



print("Test accuracy: {}%".format(dtScore))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42, n_estimators=10)



rf.fit(x_train, y_train)



rfScore = rf.score(x_test, y_test)*100

compareScore.append(rfScore)



print("Test accuracy: {}%".format(rfScore))
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



algoList = ["LogisticRegression", "KNN", "SVM", "NaiveBayes", "DecisionTree", "RandomForest"]

comparison = {"Models" : algoList, "Accuracy" : compareScore}

dfComparison = pd.DataFrame(comparison)



newIndex = (dfComparison.Accuracy.sort_values(ascending = False)).index.values

sorted_dfComparison = dfComparison.reindex(newIndex)





data = [go.Bar(

               x = sorted_dfComparison.Models,

               y = sorted_dfComparison.Accuracy,

               name = "Scores of Models",

               marker = dict(color = "rgba(116,173,209,0.8)",

                             line=dict(color='rgb(0,0,0)',width=1.0)))]



layout = go.Layout(xaxis= dict(title= 'Models',ticklen= 5,zeroline= False))



fig = go.Figure(data = data, layout = layout)



iplot(fig)