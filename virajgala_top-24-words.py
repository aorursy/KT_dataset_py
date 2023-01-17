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
import os

print(os.listdir("../input"))
from __future__ import print_function

import sys

import re

import numpy as np 

from numpy import dot

from numpy.linalg import norm
!pip install pyspark
from pyspark import SparkContext
def stringVector(x): 

    returnVal= str (x[0]) 

    for j in x[1]:

        returnVal += ','+ str(j) 

    return returnVal
    sc = SparkContext(appName="Assig6")
data = "../input/20-lines-datasettxt" 
corpus = sc.textFile(data)
numberOfDocs = corpus.count()

numberOfDocs
validLines = corpus.filter(lambda x : 'id=' in x and 'url=' in x)

V=validLines.count()

V
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]),regex.sub(' ', x[1]).lower().split()))
allWords= keyAndListOfWords.flatMap(lambda x: [x[0],[(data, 1) for data in x[1]]])
allWords= keyAndListOfWords.flatMap(lambda x: [(data, 1) for data in x[1]])
keyval= allWords.groupByKey()
wordsGrouped = allWords.groupByKey()
allWords.map(lambda x , y: (x , 1)).reduceByKey(lambda a,b: (a + b))
wordsGrouped = allWords.groupByKey()
allCounts = wordsGrouped.mapValues(sum).map(lambda x: (x[1],x[0])).sortByKey(False)
topWords = allCounts.values().take(24)
topWords
twentyK = sc.parallelize(range(24))
dictionary = twentyK.map(lambda x: (topWords[x], x))
sc.stop()
