# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from math import *



df = None

def load_data(file_name):

    global df

    df = pd.read_csv(file_name)

    #assigning numbers to classes

    mapping = {c: i for i, c in enumerate(list(df))}

    Y = df["quality"]

    X = df.drop("quality", axis=1)

    return (X,Y)



def probability(Y, true):

    if Y.count()==0:

        return 0

    mean = 5

    if true == 0:

        return Y[Y<mean].count()/Y.count()

    else:

        return Y[Y>=mean].count()/Y.count()

    

    

def entropy(Y):

    if probability(Y,0)==0:

        value0 = 0

    else:

        value0 = -probability(Y,0)*log(probability(Y,0),2)

    

    if probability(Y,1)==0:

        value1 = 0

    else:

        value1 = -probability(Y,1)*log(probability(Y,1),2)    

                                       

    return value0+value1





def entropyFeature(X,Y,feature):

    

    mean = columnMean[feature]

    #feature value < mean

    entropy0 = entropy(Y.loc[X[feature][X[feature]<mean].index])

    #feature value >= mean

    entropy1 = entropy(Y.loc[X[feature][X[feature]>=mean].index])

    

    p0 = X[feature][X[feature]<mean].count()/X[feature].count()

    p1 = X[feature][X[feature]>=mean].count()/X[feature].count()

    

    return p0*entropy0 + p1*entropy1

    

def informationGain(entropySource, entropyFeature):

    return entropySource - entropyFeature

adj = {}        

root = 1

columnMean ={}

def recur(X, Y):

    global root

    maxValue = -1

    rootFeature = None

    entropySource = entropy(Y)

    assert Y.count()==len(X.index)

    currRoot = root

    root+=1

    if entropySource == 0:

        if probability(Y,1)==1:

            return (currRoot,"YES")

        else:

            return (currRoot,"NO")

    if Y.count()==0 or len(X.columns)==0:

        return (currRoot,"NO")

    

    for feature in list(X.head(0)):

        currValue = informationGain(entropySource, entropyFeature(X,Y,feature))

        if currValue > maxValue:

            maxValue = currValue

            rootFeature = feature

    

    mean = columnMean[rootFeature]

    

    child0X = X[X[rootFeature]<mean]

    child1X = X[X[rootFeature]>=mean]

    

    child0X = child0X.drop(rootFeature,axis=1)

    child1X = child1X.drop(rootFeature,axis=1)

    

    child0Y = Y.loc[child0X.index]

    child1Y = Y.loc[child1X.index]

    

    

    currRoot=(currRoot,rootFeature)

    adj[currRoot]=[]



    child0 = recur(child0X, child0Y)

    child1 = recur(child1X, child1Y)

    

    adj[currRoot].append(child0)

    adj[currRoot].append(child1)

    

    return currRoot

   

    

Y_train: pd.DataFrame

X_train: pd.DataFrame

ok = None

rootFeature = None    



         

def dfs(root, X):

    feature = root[1]

    if feature=="YES":

        return 1

    elif feature == "NO":

        return 0

    mean = columnMean[feature]

   

    if X[feature] < mean:

        return dfs(adj[root][0],X)

    else:

        return dfs(adj[root][1],X)

    



def train(X,Y):

    for feature in list(X.head(0)):

        print('Feature - ' + str(feature) + ', entropy - ' + str(entropyFeature(X,Y,feature)))

    rootFeature = recur(X,Y)

    return rootFeature



def test(X, Y, root):

    

    num=0

    den=0

    cnt2=0

    cnt3=0

    for index, row in X.iterrows():

    

        res = dfs(root, row)

        if Y[index]<5:

            desired=0

        else:

            desired=1

        if res==desired:

            num+=1

        den+=1

        if res==1:

            cnt2+=1

        else:

            cnt3+=1

            

        print (row)

        print(f'Output: {res} desired: {desired}')

    

    print('accuracy - ' + str(num/den))



def main():

    global X_train

    global Y_train

    global ok

    X, Y = load_data('../input/winequality-red.csv')

    

    #gets a random 80% of the entire set

    X_train = X.sample(frac=0.8, random_state=1)

    Y_train = Y.loc[X_train.index]

    

    for column in X_train:

        columnMean[column] = X_train[column].mean()

    

    

    #gets the left out portion of the dataset

    X_test = X.loc[~X.index.isin(X_train.index)]

    Y_test = Y.loc[X_test.index]

    

    root = train(X_train,Y_train)

    

    test(X_test, Y_test, root)

    return

    

    

    return

    dfs(1)

    

    

if __name__ == '__main__':

    main()



    
import numpy as np

a=np.arange(10)

np.where(a<5)[0].shape[0]

list(df.head(0))

245/(245+75)
print(X_train['fixed acidity'].mean())

print(Y_train[Y_train>5].count())

print (Y_test)



X_train
