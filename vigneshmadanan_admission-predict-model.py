import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
pred=pd.read_csv("../input/Admission_Predict.csv")
# Splitting dependent and independent values to arrays 
X = pred.iloc[:, :-1].values
y = pred.iloc[:, 8].values
pred=pred.drop("Serial No.",axis=1) 
plt.plot(pred) 
# GRE Scores in 300 to 350 Range
# TOEFL scores in 100-120 range
# University Rating SOP=1-5 LOR=1-5 CGPA=5-10 Research 0-1
# Data Visualisation
# In the graph, we see high correlation between Research, LOP, SOP, Uni Rating
pred.corr()
sns.heatmap(pred.corr())

# Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
len(X_train),len(y_train),len(X_test),len(y_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred
coeff=regressor.coef_
inter=regressor.intercept_
coeff,inter
#Mean Square Error and R2 score
mean_squared_error(y_test, y_pred),r2_score(y_test,y_pred)
