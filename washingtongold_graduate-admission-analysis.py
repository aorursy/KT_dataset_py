import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

data.head()
data = data.drop('Serial No.',axis=1)

def admit(chance):

    if chance > 0.5:

        return 1

    else:

        return 0

data['Admit'] = data['Chance of Admit '].apply(admit)

data.head()
from sklearn.model_selection import train_test_split

X = data.drop(['Admit','Chance of Admit '],axis=1)

y = data['Chance of Admit ']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor



lr = LinearRegression()

lr.fit(X_train, y_train)



rr = Ridge(alpha=0.01)

rr.fit(X_train, y_train)



rr100 = Ridge(alpha=100) #  comparison with alpha value

rr100.fit(X_train, y_train)



lasso = Lasso()

lasso.fit(X_train,y_train)



dtr = DecisionTreeRegressor()

dtr.fit(X_train,y_train)



mlpr = MLPRegressor()

mlpr.fit(X_train,y_train)



from sklearn.metrics import mean_absolute_error as mae

print("LR")

print(mae(y_test,lr.predict(X_test)))

print("RR")

print(mae(y_test,rr.predict(X_test)))

print("RR100")

print(mae(y_test,rr100.predict(X_test)))

print("LASSO")

print(mae(y_test,lasso.predict(X_test)))

print("Decision Tree Regression")

print(mae(y_test,dtr.predict(X_test)))

print("Neural Network Regression")

print(mae(y_test,mlpr.predict(X_test)))
import shap



explainer = shap.TreeExplainer(dtr)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rr, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
from pdpbox import pdp, info_plots

import matplotlib.pyplot as plt



base_features = data.columns.values.tolist()

base_features.remove('Chance of Admit ')

base_features.remove('Admit')



for column in data.columns.drop(['Chance of Admit ','Admit']):

    

    feat_name = column

    pdp_dist = pdp.pdp_isolate(model=dtr, dataset=X_test, model_features=base_features, feature=feat_name)



    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()
columns = data.columns.tolist()

columns.remove('Chance of Admit ')

columns.remove('Admit')
starting_index = 1

for column in columns:

    for index in range(starting_index,7):

        feat_list = [column,columns[index]]

        inter1  =  pdp.pdp_interact(model=dtr, dataset=X_test, model_features=base_features, features=feat_list)

        pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=feat_list, plot_type='contour')

        plt.show()

    starting_index += 1
data.head(1)
import seaborn as sns

starting_index = 1

for column in columns:

    for index in range(starting_index,7):

        sns.scatterplot(data[column],data[columns[index]])

        sns.lmplot(column,columns[index],data = data)

        plt.show()

    starting_index += 1
for column in columns:

    sns.lmplot(column,'Chance of Admit ',data = data)

    plt.show()