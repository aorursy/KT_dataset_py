#customary imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#series when data is an array
series1 = pd.Series(np.random.randn(5), index= ["a", "b", "c", "d", "e"])
series1
series1[2]
series1.index
#series when data is a dictionary
d = {'a': 10, 'b': 11, 'c':12}
series2 = pd.Series(d)
series2
#series when data is a scalar value
series3 = pd.Series(data = [5, 2,3] , index = ['a','b', 'c'])
series3
#operations on series 
series1
series3[series3.index == "b"]
#to fetch a value, first argument is always index either the ones you specified like "a", "b", "c" or numeric ones assigned by python starting from 0,1, ...
series3[2]
#to get more than 1 value, put all the indexes in a square bracket
series3[[1,2]]
#add values to a series by dict
series2["d"] = 4
series2
series3["u"] = 12
series3
#check if an index exists in a series
'u' in series3.index
#to check if a value exists in a series
12 in series3.values
for x in series3.values:
    print (x)
np.multiply(series3,2)
np.zeros(5)
data = np.zeros((2,), dtype=[('A', 'i4'),('B', 'f4'),('C', 'a10')])
data
test_dataset = pd.read_csv("../input/yelp_user.csv")
test_dataset.head()
#add a new column to the dataset using assign . it returns a copy of the data, leaving the original DataFrame untouched.
test_dataset.assign(perc_of_funny_reviews = test_dataset['useful']/test_dataset['review_count']).head()
#We can also pass a function, instead of the actual values. for example if we want the computation done only on a subset of data
#the query must go in 1 bracket!
test_dataset.query('review_count > 1').assign(perc_of_funny_reviews = lambda x: x['useful']/x['review_count']).head()
#filtering happens first and then the plotting!
test_dataset.query('review_count > 2000').assign(perc_of_funny_reviews = lambda x: x['useful']/x['review_count']).plot(kind='scatter', x='review_count', y='useful')