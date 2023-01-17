import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
import keras
%matplotlib inline
#u_cols = ['id', '', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('wine reviews.csv')
users
newDf = users.loc[0:100,['brand','categories','dateAdded','reviews.title','reviews.rating']]
newDf
newDf.dropna()
newDf
#newDf['log_review_ratings']=np.log(newDf['reviews.rating'].values)
plt.hist(newDf['reviews.rating'].values, bins= 100, color='red')
plt.xlabel('reviews_rating')
plt.ylabel('Number of reviews')
plt.show()
import sklearn
from sklearn.model_selection import train_test_split
feature_names = list(newDf.columns)
y = newDf['reviews.rating']
Xtr,Xtv,Ytr,Ytv = train_test_split(newDf[feature_names].values, y, test_size=0.2, random_state= 1999)
import xgboost as xgb
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(Xtr, Ytr)
# make predictions for test data
y_pred = model.predict(Xtv)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(Ytv, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



