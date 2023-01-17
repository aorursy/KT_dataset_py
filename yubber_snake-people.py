# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# lc4311 is cwacc



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as skl

import matplotlib.pyplot as plt # prereq for sns

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/housing.csv")



# loc: locate by names

# iloc: locate by index. needs #



data.loc[data["total_bedrooms"].isna()] # squish missing vals ØwØ 



# ocean_proximity needs to be numerical 



data.groupby("ocean_proximity").describe()["median_house_value"] # from this we know location matters - island houses are most pricey cuz p r i v a c y 



data.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1, c = "median_house_value") # alpha is opt, measures pepel. c shows house pricing in shade, opt. 



# ~data["key"].isna() == isn't na
from sklearn.impute import SimpleImputer # for every missing value replace it. but is dumdum only checks 1 col



data_num = data.drop("ocean_proximity", axis=1)

data_cat = data["ocean_proximity"] # imputing this is dumb ur dumb heckin dum aaaaaAAAAå´ø˚µß√®øßˆƒµ©ªœ∑´ÅÍÎ◊˝Á‡ˇ 



ornageman = SimpleImputer(strategy = "median") # replace with median data (435 in this case)



ornageman.fit(data_num) # training time



ornageman.transform(data_num)



transdata_num = pd.DataFrame(ornageman.transform(data_num),columns=data_num.columns)



print(transdata_num.isna().sum()) # all 0 -> yay is ocmplet



data_cat.unique() # lists cat types
from sklearn.linear_model import LinearRegression # to predict categorical use Naïve Bayes



lr_model = LinearRegression() # hello world



lr_model.fit(transdata_num.drop("median_house_value",axis=1),transdata_num["median_house_value"]) # predicts MHV so drops it 



lr_model.predict(transdata_num.drop("median_house_value",axis=1))
data_cat
# wanna cry? maybe piss your pants? perhaps shit and ç¨µ?



from sklearn.preprocessing import OneHotEncoder as OHA # no this was never intended to be risqué



encoder = OHA()



encoder.fit(pd.DataFrame(data_cat))

transval = encoder.transform(pd.DataFrame(data_cat)) # returns a sparse matrix, a matrix with mostly 0s – in every row/col there is only 1 true 



data_cate = pd.DataFrame(transval.toarray(),columns=encoder.categories_)



data_cate
dataowo = pd.concat([transdata_num,data_cate],axis=1)



dataowo
from sklearn.tree import DecisionTreeRegressor as DTR



lm = LinearRegression()



targ = dataowo["median_house_value"]

predictors = dataowo.drop("median_house_value",axis=1)



lm.fit(predictors,targ)



preds = lm.predict(predictors)



np.sqrt(((preds-targ)**2).mean()) # root mean square error (wtf) - abs bad
tree = DTR()



tree.fit(predictors,targ)



trepid = tree.predict(predictors)



np.sqrt(((trepid-targ)**2).mean()) # oh mai gäAAAAÃ



# a decision tree partitions data and asks if/else. vary fast.

# e.g. x between(a,b) ? x is ‡ : x is £

# it can only determine stuff inside partitioned regions, cannot predict unknown data outside partitions.