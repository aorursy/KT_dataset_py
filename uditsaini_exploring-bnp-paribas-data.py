%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

# Any results you write to the current directory are saved as output.
#train shape
print(train.shape)
#test shape
print(test.shape)
%matplotlib inline
#target varibale 
train.target.value_counts().plot.bar()
#percantage of null values
float(len(train[pd.isnull(train)]))/float((train.shape[1])*train.shape[0])
#null value percantages
nullvalues=[float((train[col].isnull().sum()))/len(train[col]) for col in train.columns.values]
percentagenull=list(zip(train.columns.values,nullvalues))
nullplot=pd.DataFrame(data=percentagenull,columns=["varname","percantage_null"])
nullplot=nullplot.set_index("varname")
nullplot.plot.bar(figsize =(23,5),title="percentage of null values per feature")
#duplicate row in train set
train.shape[0]-train.drop_duplicates().shape[0]
#constent feature count
uniquecount=[train[col].nunique() for col in train.columns.values]
uniquecount=pd.DataFrame(data=list(zip(train.columns.values,uniquecount)),columns=["var","unique_count"])
unique_count=uniquecount[uniquecount.unique_count==1]
print("constent features count = {} ".format(unique_count.shape[0]))

#seprating numeric and charcter features
train_numr =train.select_dtypes(include=[np.number])
train_char =train.select_dtypes(include=[np.object])
print("Numerical column count : {}".format(train_numr.shape[1]))
print("Character column count : {}".format(train_char.shape[1]))
#lest look at charcter features
for col in  train_char:
    print(col+" : " +str(train_char[col].unique()[:10]))
%matplotlib inline
import random
df = train_numr.loc[np.random.choice(train_numr.index, 25000, replace=False)]
plt.matshow(df.corr())
