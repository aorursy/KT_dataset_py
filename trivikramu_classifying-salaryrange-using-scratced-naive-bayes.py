import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time
from sklearn.naive_bayes import GaussianNB,CategoricalNB
import numpy as np
from sklearn import metrics
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/adults/adult.data",header=None,names=['Age','Workplace','fnlwgt','education','education num','marital-stauts','occupation','relationship','race','sex','captial gain','capital loss','hours per week','native country','Salaray'])
df
def removeQuestionMark(outCol,inpCol):
    for i in inpCol:
        repl=X[i].value_counts().keys().tolist()[0]
        X[i]=X[i].replace(to_replace=' ?',value=repl)
    rep=y[outCol[0]].value_counts().keys().tolist()[0]
    y[outCol[0]]=y[outCol[0]].replace(to_replace=' ?',value=rep)
    
def Splitting(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return [X_train,y_train,X_test,y_test]

def CategTraining(X_train,y_train,outCol):
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
    return trainDict

def CategTesting(outCol,trainDict,y_train,y_test,X_test):
    trainProb=(y_train[outCol[0]].value_counts()/y_train.shape[0]).to_dict()
    outputList=y_train[outCol[0]].value_counts().keys().tolist()
    testDict1=dict([(key,dict([(keys, []) for keys in outputList])) for key in y_test.index])
    for i in testDict1.keys():
        for j in testDict1[i].keys():
            prob=1
            l=0
            for k in trainDict.keys():
                prob=trainDict[k][X_test.loc[i][l]][j]*prob
                l=l+1
            testDict1[i][j]=prob*trainProb[j]
    return testDict1

def CategPredict(testDict,y_test):
    size=X_test.shape[0]
    predict=dict((key,[]) for key in y_test.index)
    for i in predict.keys():
        maxi=0
        l=''
        for j in testDict[i].keys():
            if(testDict[i][j]>maxi):
                maxi=testDict[i][j]
                l=j
        predict[i]=l
    accuracy=0
    count=0
    for i in predict.keys():
        if(y_test.loc[i][0]==predict[i]):
            count=1+count
    accuracy=count*100/size
    print(accuracy)

def Probop(y_test,posOp):
    Probop=dict([(key,[]) for key in posOp])
    for i in Probop.keys():
        Probop[i]=(y_test['Salaray'].value_counts()[i])/y_test.shape[0]
    return Probop

def ContTraining(X_train,y_train):
    NumericDict=dict([(key,dict([(keys, []) for keys in posOp])) for key in X_train.columns])
    Inplist=dict([(key,dict([(keys, []) for keys in posOp])) for key in X_train.columns])
    for i in NumericDict.keys():
        for j in NumericDict[i].keys():
            Inplist[i][j]=(y_train.loc[y_train['Salaray']==j]).index.tolist()
    for i in NumericDict.keys():
        for j in NumericDict[i].keys():
            count=len(Inplist[i][j])
            su=0
            for k in range(0,count):
                su+=(X_train.loc[Inplist[i][j][k]])[i]
            NumericDict[i][j]=[su/count]
    for i in Inplist.keys():
        for j in Inplist[i].keys():
            count=len(Inplist[i][j])
            diff=0
            for k in range(0,count):
                diff+=((X_train.loc[Inplist[i][j][k]])[i]-NumericDict[i][j][0])**2
            NumericDict[i][j].append(diff/count)
    return NumericDict    

def ContTesting(X_test,posOp,NumericDict):
    testInd=X_test.index
    prediction=dict([(key,dict([(keys,dict([(key,[]) for key in posOp])) for keys in testInd])) for key in X_test.columns])
    for i in prediction.keys():
        for j in posOp:
            den=math.sqrt(2*math.pi*(NumericDict[i][j][1]**2))
            count=len(prediction[i])
            for k in range(0,count):
                num=math.exp(-(X_test.loc[testInd[k]][i]-NumericDict[i][j][0]/(2*math.pow(NumericDict[i][j][1],2))))
                prediction[i][testInd[k]][j]=num/den
    Predict=dict([(key,dict([(keys, []) for keys in posOp])) for key in y_test.index])
    for i in Predict.keys():
        for j in posOp:
            prob=1
            for k in prediction.keys():
                if(prediction[k][i][j]!=0):
                    prob=prob*prediction[k][i][j]
            Predict[i][j]=prob
    return Predict

def ContPredict(prediction,ProbOp,y_test):
    testInd=y_test.index
    FinalPrediction=dict([(keys,[]) for keys in testInd])
    for i in testInd:
        maxi=0
        l=''
        for j in posOp:
            prob=1
            for k in prediction.keys():
                if(prediction[k][i][j]!=0):
                    prob=prob*prediction[k][i][j]
            prob=prob*ProbOp[j]
            if(prob>maxi):
                maxi=prob
                l=j
        FinalPrediction[i]=l 
    acc=0
    count=0
    length=len(testInd)
    for i in range(0,length):
            if(FinalPrediction[testInd[i]]==y_test.iloc[i,0]):
                count+=1
    acc=count*100/y_test.shape[0]
    print(acc)
    
def Predict(testDict,prediction,ProbOp,y_test):
    testInd=y_test.index
    FinalPrediction=dict([(keys,[]) for keys in testInd])
    for i in FinalPrediction.keys():
        maxi=0
        pr=''
        p=0
        for j in posOp:
            p=prediction[i][j]*testDict[i][j]
            if(p>maxi):
                maxi=p
                pr=j
        FinalPrediction[i]=pr
    acc=0
    count=0
    length=len(testInd)
    for i in range(0,length):
            if(FinalPrediction[testInd[i]]==y_test.iloc[i,0]):
                count+=1
    acc=count*100/y_test.shape[0]
    return acc
X=df[['Workplace','education','race','sex','marital-stauts','occupation','relationship','native country','Age','fnlwgt','education num','captial gain','capital loss','hours per week']]
y=df[['Salaray']]
[x_train,y_train,x_test,y_test]=Splitting(X,y)
inpCol=list(X.columns)
outCol=list(y.columns)
removeQuestionMark(outCol,inpCol)
X_train=x_train[['Workplace','education','race','sex','marital-stauts','occupation','relationship','native country']]
X_test=x_test[['Workplace','education','race','sex','marital-stauts','occupation','relationship','native country']]
trainDict=CategTraining(X_train,y_train,outCol)
testDict=CategTesting(outCol,trainDict,y_train,y_test,X_test)
X_train=x_train[['Age','fnlwgt','education num','captial gain','capital loss','hours per week']]
X_test=x_test[['Age','fnlwgt','education num','captial gain','capital loss','hours per week']]
posOp=y_train['Salaray'].value_counts().keys().tolist()
size=y_train.shape[0]
ProbOp=Probop(y_test,posOp)
NumericDict=ContTraining(X_train,y_train)
prediction=ContTesting(X_test,posOp,NumericDict)
print("The accuracy obtained from scratched Naive Bayes is "+str(Predict(testDict,prediction,ProbOp,y_test)))
X=df[['Workplace','education','race','sex','marital-stauts','occupation','relationship','native country','Age','fnlwgt','education num','captial gain','capital loss','hours per week']]
y=df[['Salaray']]
[x_train,y_train,x_test,y_test]=Splitting(X,y)
inpCol=list(X.columns)
outCol=list(y.columns)
removeQuestionMark(outCol,inpCol)
[x_train,y_train,x_test,y_test]=Splitting(X,y)
X_train=x_train[['Workplace','education','race','sex','marital-stauts','occupation','relationship','native country']]
X_test=x_test[['Workplace','education','race','sex','marital-stauts','occupation','relationship','native country']]
trainDict=CategTraining(X_train,y_train,outCol)
testDict=CategTesting(outCol,trainDict,y_train,y_test,X_test)
X_train=x_train[['Age','fnlwgt','education num','captial gain','capital loss','hours per week']]
X_test=x_test[['Age','fnlwgt','education num','captial gain','capital loss','hours per week']]
posOp=y_train['Salaray'].value_counts().keys().tolist()
size=y_train.shape[0]
ProbOp=Probop(y_test,posOp)
clf = GaussianNB()
clf.fit(X_train,y_train)
s=clf.predict_proba(X_test)
r=0
c=0
prediction=dict([(key,dict([(keys, []) for keys in posOp])) for key in y_test.index])
for i in prediction.keys():
    c=0
    for j in prediction[i].keys():
        prediction[i][j]=s[r][c]
        c=c+1
    r=r+1
print("The accuracy obtained after using sklearn of Gaussian Naive Bayes and scratched categNaive Bayes is "+str(Predict(testDict,prediction,ProbOp,y_test)))