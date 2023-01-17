import pandas as pd
import math
from statistics import mode 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_test=pd.read_csv('/kaggle/input/titanic/test.csv')
data_train=pd.read_csv('/kaggle/input/titanic/train.csv')
y_train=data_train[["Survived"]]
submission = data_test[["PassengerId"]]
for i in ('PassengerId','Survived','Name','Ticket'):
    del data_train[i]
for i in ('PassengerId','Name','Ticket'):
    del data_test[i]
data_train
categCol=data_train.select_dtypes(exclude=['float64','int']).columns
contCol=data_train.select_dtypes(exclude=['object']).columns
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
data_train[contCol]=imp.fit_transform(data_train[contCol])
data_test[contCol]=imp.transform(data_test[contCol])
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_train[categCol]=imp.fit_transform(data_train[categCol])
data_test[categCol]=imp.transform(data_test[categCol])
X=data_train
enc = OneHotEncoder(handle_unknown='ignore')
binaryValues=enc.fit_transform(X[categCol]).toarray()
newCol=list(enc.get_feature_names(categCol))
categData = pd.DataFrame(binaryValues,columns=newCol,index=X.index)
X=data_train[contCol].merge(categData,left_index=True,right_index=True)
scaler = StandardScaler()
X=pd.DataFrame(data=scaler.fit_transform(X),columns=X.columns)
X
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
neight = KNeighborsClassifier()
length=int(math.sqrt(X.shape[0]))
parameters={'n_neighbors':([i for i in range(2,100)])}
clf = GridSearchCV(neight, parameters,verbose=0,cv=5)
clf.fit(X,y_train)
predictions=clf.predict(X)
knn=clf.best_estimator_
print(knn)
from sklearn.tree import DecisionTreeClassifier
decreaseList=[]
decreaseList.append(0)
for i in range(0,20):
    decreaseList.append(float(decreaseList[i]+0.05))
print(decreaseList)
param_grid = {'min_samples_split': range(2,30),'min_impurity_split':[None,0.45,0.46,0.465,0.466,0.467,0.4668,0.4669,0.47,0.471,0.49,0.5,0.52,0.55,0.57,0.6,0.7,0.8],'min_impurity_decrease':decreaseList}
clf = DecisionTreeClassifier(criterion='entropy',random_state=42)
grid_search2 = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 5)
grid_search2.fit(X,y_train)
print(grid_search2.score(X,y_train))
tree=grid_search2.best_estimator_
print(tree)
from sklearn.naive_bayes import GaussianNB
class NaiveBayes(GaussianNB):
    trainDict=None
    outCol=None
    y_train=None
    X_train=None
    gausscianNaiveBayes=None
    predictions=None
    TestInd=None
    
    def CategTraining(self,X_train,y_train,outCol):
        trainCount=y_train[outCol[0]].value_counts().to_dict()
        col1=list(X_train.columns)
        outputList=y_train[outCol[0]].value_counts().keys().tolist()
        trainDict=dict([(key, []) for key in col1])
        for i in range(0,len(col1)):
            inputList=X_train[col1[i]].value_counts().keys().tolist()
            proxy=dict([(key, dict([(keys, []) for keys in outputList])) for key in inputList])
            trainDict[col1[i]]=proxy
        for i in trainDict.keys():
            for j in trainDict[i].keys():
                for k in trainDict[i][j].keys():
                    num=(X_train.loc[(X_train[i]==j) &  (y_train[outCol[0]]==k)]).shape[0]
                    den=trainCount[k]
                    prob=num/den
                    trainDict[i][j][k]=prob
        self.trainDict=trainDict
        self.outCol=outCol
        self.y_train=y_train
        self.X_train=X_train
    
    def CategTesting(self,X_test):
        y_train=self.y_train
        outCol=self.outCol
        trainDict=self.trainDict
        trainProb=(y_train[outCol[0]].value_counts()/y_train.shape[0]).to_dict()
        outputList=y_train[outCol[0]].value_counts().keys().tolist()
        testDict1=dict([(key,dict([(keys, []) for keys in outputList])) for key in X_test.index])
        for i in testDict1.keys():
            for j in testDict1[i].keys():
                prob=1
                l=0
                for k in trainDict.keys():
                    try:
                        prob=trainDict[k][X_test.loc[i][l]][j]*prob
                    except Exception as e:
                        pass
                    l=l+1
                testDict1[i][j]=prob*trainProb[j]
        return testDict1
    
    
    def fit(self, X, y):
        data=pd.DataFrame(data=scaler.inverse_transform(X),columns=X.columns)
        contData=data[contCol]
        categData=pd.DataFrame(data=enc.inverse_transform(data[newCol]),columns=categCol)
        X=pd.merge(categData,contData,left_index=True, right_index=True)
        X_cont=X.select_dtypes(exclude='object')
        self.gausscianNaiveBayes=super().fit( X_cont, y)
        self.y_train=pd.DataFrame(data=y,columns=["Survived"])
        outCol=y_train.columns.tolist()
        y=y_train
        X_Categ=X.select_dtypes(include='object')
        self.CategTraining(X_Categ,y,outCol)
        return self
    
    def predict(self,X):
        data=pd.DataFrame(data=scaler.inverse_transform(X),columns=X.columns)
        contData=data[contCol]
        categData=pd.DataFrame(data=enc.inverse_transform(data[newCol]),columns=categCol)
        X=pd.merge(categData,contData,left_index=True, right_index=True)
        X_Categ=X.select_dtypes(include='object')
        CategtestDict=self.CategTesting(X_Categ)
        X_Cont=X.select_dtypes(exclude='object')
        outCol=self.outCol
        ContpredictProb=super().predict_proba(X_Cont)
        y_train=self.y_train
        posOp=y_train[outCol[0]].value_counts().keys().tolist()
        testInd=X.index.tolist()
        FinalPrediction=[]
        r=0
        c=0
        ContTestDict=dict([(key,dict([(keys, []) for keys in posOp])) for key in X.index])
        for i in ContTestDict.keys():
            c=0
            for j in ContTestDict[i].keys():
                ContTestDict[i][j]=ContpredictProb[r][c]
                c=c+1
            r=r+1
        size=X.shape[0]
        for i in ContTestDict.keys():
            maxi=0
            pr=''
            p=0
            for j in posOp:
                p=ContTestDict[i][j]*CategtestDict[i][j]
                if(p>maxi):
                    maxi=p
                    pr=j
            FinalPrediction.append(pr)
        return FinalPrediction
    
    def score(self, X, y):
        data=pd.DataFrame(data=scaler.inverse_transform(X),columns=X.columns)
        contData=data[contCol]
        categData=pd.DataFrame(data=enc.inverse_transform(data[newCol]),columns=categCol)
        X=pd.merge(categData,contData,left_index=True, right_index=True)
        X_Categ=X.select_dtypes(include='object')
        CategtestDict=self.CategTesting(X_Categ)
        X_Cont=X.select_dtypes(exclude='object')
        outCol=self.outCol
        ContpredictProb=super().predict_proba(X_Cont)
        y_train=self.y_train
        posOp=y_train[outCol[0]].value_counts().keys().tolist()
        testInd=X.index.tolist()
        FinalPrediction=[]
        r=0
        c=0
        ContTestDict=dict([(key,dict([(keys, []) for keys in posOp])) for key in X.index])
        for i in ContTestDict.keys():
            c=0
            for j in ContTestDict[i].keys():
                ContTestDict[i][j]=ContpredictProb[r][c]
                c=c+1
            r=r+1
        size=X.shape[0]
        for i in ContTestDict.keys():
            maxi=0
            pr=''
            p=0
            for j in posOp:
                p=ContTestDict[i][j]*CategtestDict[i][j]
                if(p>maxi):
                    maxi=p
                    pr=j
            FinalPrediction.append(pr)
        from sklearn import metrics
        return metrics.accuracy_score(y,FinalPrediction)   
naiveBayes=NaiveBayes()
naiveBayes.fit(X,y_train)
naiveBayes.score(X,y_train)
X_test=data_test
binaryValues=enc.transform(X_test[categCol]).toarray()
newCol=list(enc.get_feature_names(categCol))
categData = pd.DataFrame(binaryValues,columns=newCol,index=X_test.index)
X_test=data_test[contCol].merge(categData,left_index=True,right_index=True)
scaler = StandardScaler()
X_test=pd.DataFrame(data=scaler.fit_transform(X_test),columns=X_test.columns)
X_test
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
X_train=X
eclf1 = VotingClassifier(estimators=[('knn', knn), ('tree',tree), ('nb',naiveBayes)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
predictions=eclf1.predict(X_test)
output=pd.DataFrame(predictions)
output.to_csv('output')
submit = pd.DataFrame(list(zip(submission["PassengerId"], predictions)), columns = ["PassengerId", "Survived"])
submit["Survived"] = submit["Survived"].map({
    1.0 : 1,
    0.0 : 0,
})
submit.to_csv("submit.csv", index = False)
submit
