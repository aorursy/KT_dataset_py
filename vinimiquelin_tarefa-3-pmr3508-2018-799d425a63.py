import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../input/train.csv')
df
df.shape
df["population"].value_counts().plot(kind="hist")
df["population"].mean()
df["households"].value_counts().plot(kind="hist")
df["households"].mean()
df["total_rooms"].value_counts().plot(kind="hist")
df["total_rooms"].mean()
df["total_bedrooms"].value_counts().plot(kind="hist")
df["total_bedrooms"].mean()
tdf = pd.read_csv('../input/test.csv')
tdf
tdf.shape
Xdf = df[["latitude", "longitude", "median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]]
Ydf = df.median_house_value

Xtdf = tdf[["latitude", "longitude", "median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]]
from sklearn.neighbors import KNeighborsRegressor
KNR = KNeighborsRegressor(n_neighbors=20)
KNR.fit(Xdf, Ydf)
from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(KNR, Xdf, Ydf, cv=20)
scores1
np.mean(scores1)
from sklearn import linear_model
LR = linear_model.Lasso(alpha=1.0)
LR.fit(Xdf, Ydf)
from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(LR, Xdf, Ydf, cv=20)
scores2
np.mean(scores2)
from sklearn.linear_model import Ridge
RR = Ridge(alpha=1.0)
RR.fit(Xdf, Ydf)
from sklearn.model_selection import cross_val_score
scores3 = cross_val_score(RR, Xdf, Ydf, cv=20)
scores3
np.mean(scores3)
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor(max_depth=10)
DTR.fit(Xdf, Ydf)
from sklearn.model_selection import cross_val_score
scores4 = cross_val_score(DTR, Xdf, Ydf, cv=20)
scores4
np.mean(scores4)
predicted_KNR = KNR.predict(Xtdf)
predicted_KNR
KNRdf = pd.DataFrame(index=tdf.Id,columns=['median_house_value'])
KNRdf['median_house_value'] = predicted_KNR
KNRdf
KNRdf.shape
predicted_LR = LR.predict(Xtdf)
predicted_LR
LRdf = pd.DataFrame(index=tdf.Id,columns=['median_house_value'])
LRdf['median_house_value'] = predicted_LR
LRdf
LRdf.shape
num = LRdf._get_numeric_data()
num[num < 0] = -num
predicted_RR = RR.predict(Xtdf)
predicted_RR
RRdf = pd.DataFrame(index=tdf.Id,columns=['median_house_value'])
RRdf['median_house_value'] = predicted_RR
RRdf
RRdf.shape
num = RRdf._get_numeric_data()
num[num < 0] = -num
predicted_DTR = DTR.predict(Xtdf)
predicted_DTR
DTRdf = pd.DataFrame(index=tdf.Id,columns=['median_house_value'])
DTRdf['median_house_value'] = predicted_DTR
DTRdf
KNRdf.to_csv('myPredT3_KNR.csv')
LRdf.to_csv('myPredT3_LR.csv')
RRdf.to_csv('myPredT3_RR.csv')
DTRdf.to_csv('myPredT3_DTR.csv')