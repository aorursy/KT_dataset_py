import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')
data.head()
data.drop('id',axis=1,inplace=True)
def get_year(date):

    return int(str(date)[:4])

data['year'] = data['date'].apply(get_year)

data.drop('date',axis=1,inplace=True)
data.head()
from sklearn.model_selection import train_test_split

X = data.drop('price',axis=1)

y = data['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
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

#lr, rr, rr100, lasso

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

perm = PermutationImportance(dtr, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
from pdpbox import pdp, info_plots

import matplotlib.pyplot as plt



base_features = data.columns.values.tolist()

base_features.remove('price')



feat_name = 'sqft_living'

pdp_dist = pdp.pdp_isolate(model=dtr, dataset=X_test, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()



feat_name = 'grade'

pdp_dist = pdp.pdp_isolate(model=dtr, dataset=X_test, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
inter1  =  pdp.pdp_interact(model=dtr, dataset=X_test, model_features=base_features, 

                            features=['sqft_living', 'grade'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['sqft_living','grade'], plot_type='contour')

plt.show()
inter1  =  pdp.pdp_interact(model=dtr, dataset=X_test, model_features=base_features, 

                            features=['lat', 'long'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['lat','long'], plot_type='contour')

plt.show()
import folium

from folium.plugins import HeatMap



# Create basic Folium crime map

crime_map = folium.Map(location=[47.5112,-122.257], 

                       tiles = "Stamen Terrain",

                      zoom_start = 9)



# Add data for heatmp 

data_heatmap = data[['lat','long','price']]

data_heatmap = data.dropna(axis=0, subset=['lat','long','price'])

data_heatmap = [[row['lat'],row['long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10, 

        gradient = {.35: 'blue',.55: 'purple',.68:'lime',.78:'red'}).add_to(crime_map)

# Plot!

crime_map
for column in data.columns.drop('price'):

    feat_name = column

    pdp_dist = pdp.pdp_isolate(model=dtr, dataset=X_test, model_features=base_features, 

                               feature=feat_name)



    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()
data.head()
inter1  =  pdp.pdp_interact(model=dtr, dataset=X_test, model_features=base_features, 

                            features=['sqft_living', 'sqft_lot'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['sqft_living','sqft_lot'], 

                      plot_type='contour')

plt.show()
def heart_disease_risk_factors(model, patient):



    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(patient)

    shap.initjs()

    return shap.force_plot(explainer.expected_value, shap_values, patient)



data_for_prediction = X_test.iloc[8,:].astype(float)

heart_disease_risk_factors(dtr, data_for_prediction)
shap_values = explainer.shap_values(X_train.iloc[:50])

shap.force_plot(explainer.expected_value, shap_values, X_test.iloc[:50])