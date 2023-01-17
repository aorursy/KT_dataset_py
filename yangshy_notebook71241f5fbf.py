# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from numpy import *

import operator

import csv

def toInt(array):

    array=mat(array)

    m,n=shape(array)

    newArray=zeros((m,n))

    for i in xrange(m):

        for j in xrange(n):

                newArray[i,j]=int(array[i,j])

    return newArray

    

def nomalizing(array):

    m,n=shape(array)

    for i in xrange(m):

        for j in xrange(n):

            if array[i,j]!=0:

                array[i,j]=1

    return array

    

def loadTrainData():

    l=[]

    print('done')

    with open('train.csv') as file:

        lines=csv.reader(file)

        for line in lines:

            l.append(line) #42001*785

    l.remove(l[0])

    l=array(l)

    label=l[:,0]

    data=l[:,1:]

    return nomalizing(toInt(data)),toInt(label)  #label 1*42000  data 42000*784

    #return data,label

    

def loadTestData():

    l=[]

    print('done')

    with open('test.csv') as file:

        lines=csv.reader(file)

        for line in lines:

            l.append(line)

     #28001*784

    l.remove(l[0])

    data=array(l)

    return nomalizing(toInt(data))  #  data 28000*784





#dataSet:m*n   labels:m*1  inX:1*n

def classify(inX, dataSet, labels, k):

    inX=mat(inX)

    dataSet=mat(dataSet)

    labels=mat(labels)

    dataSetSize = dataSet.shape[0]                  

    diffMat = tile(inX, (dataSetSize,1)) - dataSet   

    sqDiffMat = array(diffMat)**2

    sqDistances = sqDiffMat.sum(axis=1)                  

    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()            

    classCount={} 

    print('done')

    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i],0]

        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]



def saveResult(result):

    with open('result.csv','wb') as myFile:    

        myWriter=csv.writer(myFile)

        print('done')

        for i in result:

            tmp=[]

            tmp.append(i)

            myWriter.writerow(tmp)

        



def handwritingClassTest():

    trainData,trainLabel=loadTrainData()

    testData=loadTestData()

    m,n=shape(testData)

    resultList=[]

    print('done')

    for i in range(m):

        classifierResult = classify(testData[i], trainData, trainLabel.transpose(), 5)

        resultList.append(classifierResult)

    saveResult(resultList)