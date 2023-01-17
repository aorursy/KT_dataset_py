import pandas as pd #pandas - for data manipulation

new_data = pd.read_csv('/kaggle/input/wildfire-satellite-data/fire_nrt_M6_156000.csv') #load new data (June 2020->present)

old_data = pd.read_csv('/kaggle/input/wildfire-satellite-data/fire_archive_M6_156000.csv') #load old data (Sep 2010->June 2020)

data = pd.concat([old_data.drop('type',axis=1), new_data]) #concatenate old and new data

data = data.reset_index().drop('index',axis=1)

data['satellite'] = data['satellite'].map({'Terra':0,'Aqua':1})

data['daynight'] = data['daynight'].map({'D':0,'N':1})

data.drop('instrument', axis=1, inplace=True)

data['month'] = data['acq_date'].apply(lambda x:int(x.split('-')[1]))

data = data.sample(frac=0.2)

data = data.reset_index().drop("index", axis=1)

data.head()
X = data[['latitude','longitude','month','brightness','scan','track','acq_time','bright_t31','daynight']]

y = data['frp']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# from sklearn.ensemble import RandomForestRegressor

# from sklearn.metrics import mean_absolute_error as mae

# model = RandomForestRegressor(n_estimators = 150, max_depth = 15)

# model.fit(X_train, y_train)

# mae(model.predict(X_train), y_train)

# print(f"Train MAE: {mae(model.predict(X_train), y_train)}")

# print(f"Test MAE: {mae(model.predict(X_test), y_test)}")
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error as mae

model1 = GradientBoostingRegressor(n_estimators = 100, learning_rate=0.1,

                                  max_depth = 10, random_state = 0, loss = 'ls')

model1.fit(X_train, y_train)

print(f"Train MAE: {mae(model1.predict(X_train), y_train)}")

print(f"Test MAE: {mae(model1.predict(X_test), y_test)}")
import shap

explainer = shap.TreeExplainer(model1)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)#, plot_type="bar")
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model1, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
from pdpbox import pdp, info_plots

import matplotlib.pyplot as plt

base_features = X.columns.values.tolist()

for column in X.columns:

    feat_name = column

    pdp_dist = pdp.pdp_isolate(model=model1, dataset=X_test, model_features=base_features, feature=feat_name)

    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()