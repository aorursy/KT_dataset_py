import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd #pandas - for data manipulation

import datetime as dt

from dateutil import parser

new_data = pd.read_csv('/kaggle/input/wildfire-satellite-data/fire_nrt_M6_156000.csv') #load new data (June 2020->present)

old_data = pd.read_csv('/kaggle/input/wildfire-satellite-data/fire_archive_M6_156000.csv') #load old data (Sep 2010->June 2020)

fire_data = pd.concat([old_data.drop('type',axis=1), new_data]) #concatenate old and new data

fire_data = fire_data.reset_index().drop('index',axis=1)

fire_data = fire_data[fire_data.satellite != "Aqua"]

fire_data = fire_data.sample(frac=0.1)

fire_data = fire_data.reset_index().drop("index", axis=1)
print(f"Shape of data: {fire_data.shape}")

fire_data.rename(columns={"acq_date":"Date"}, inplace=True)

fire_data["WEI Value"] = 0

fire_data['month'] = fire_data['Date'].apply(lambda x:int(x.split('-')[1]))

fire_data.head()
wei = pd.read_excel('/kaggle/input/weekly-economic-index-wei-federal-reserve-bank/Weekly Economic Index.xlsx')

wei.drop('WEI as of 7/28/2020',axis=1,inplace=True)

wei = wei.set_index("Date")

wei.head()
from tqdm import tqdm

for index in tqdm(range(len(fire_data))):

    fire_date = (fire_data["Date"][index]) 

    fire_date = parser.parse(fire_date)

    min_wei_date_value = wei.iloc[wei.index.get_loc(fire_date,method='nearest')]["WEI"]

    fire_data.loc[index, "WEI Value"] = min_wei_date_value
fire_data['daynight'] = fire_data['daynight'].map({'D':0,'N':1})

fire_data.drop('instrument', axis=1, inplace=True)
fire_data.head()
x = fire_data[['latitude','longitude','month','brightness','scan','track',

               'acq_time','bright_t31','daynight','frp', 'confidence']]

y = fire_data['WEI Value']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# from keras.models import Sequential

# from keras.layers import Dense, Dropout, BatchNormalization

# model = Sequential()

# model.add(Dense(32,input_shape=(11,),activation='relu'))

# model.add(BatchNormalization())

# model.add(Dense(64,activation='relu'))

# model.add(Dense(64,activation='relu'))

# model.add(Dense(64,activation='relu'))

# model.add(Dense(64,activation='relu'))

# model.add(Dense(32,activation='relu'))

# model.add(Dropout(0.2))

# model.add(Dense(1,activation='linear'))

# model.compile(optimizer='adam',metrics=['mae'],loss='mse')

# model.fit(X_train, y_train, epochs=20)

print("Fail")
# from sklearn.ensemble import RandomForestRegressor

# from sklearn.metrics import mean_absolute_error as mae

# model = RandomForestRegressor(n_estimators = 200, max_depth = 15)

# model.fit(X_train, y_train)

# mae(model.predict(X_train), y_train)

# print(f"Train MAE: {mae(model.predict(X_train), y_train)}")

# print(f"Test MAE: {mae(model.predict(X_test), y_test)}")
# from sklearn.ensemble import RandomForestRegressor

# parameters = {'n_estimators':[300,500,1000], 'max_depth':[5, 10, 20, 50, 100]}

# from sklearn.model_selection import GridSearchCV

# model2 = RandomForestRegressor()

# clf = GridSearchCV(model2, parameters, cv=3)

# clf.fit(X_train, y_train)
# clf.best_params_
try:

    from sklearn.ensemble import GradientBoostingRegressor

    parameters = {'n_estimators':[100,500,1000], 'max_depth':[3, 5, 15, 50],

                  "learning_rate":[0.05,0.1,0.2]}

    from sklearn.model_selection import GridSearchCV

    model2 = GradientBoostingRegressor()

    clf = GridSearchCV(model2, parameters, n_jobs=-1, cv=3)

    clf.fit(X_train, y_train)#, verbose=True)

except:

    from sklearn.ensemble import GradientBoostingRegressor

    parameters = {'n_estimators':[100,300,500,1000], 'max_depth':[3, 5, 15, 50],

                  "learning_rate":[0.05,0.1,0.2]}

    from sklearn.model_selection import GridSearchCV

    model2 = GradientBoostingRegressor()

    clf = GridSearchCV(model2, parameters, cv=3)

    clf.fit(X_train, y_train)#, verbose=True)





# from sklearn.ensemble import GradientBoostingRegressor

# model1 = GradientBoostingRegressor(n_estimators = 400, learning_rate=0.1,

#                                   max_depth = 10, random_state = 0, loss = 'ls')

# model1.fit(X_train, y_train)

# print(f"Train MAE: {mae(model1.predict(X_train), y_train)}")

# print(f"Test MAE: {mae(model1.predict(X_test), y_test)}")
clf.best_params_
from sklearn.metrics import mean_absolute_error as mae

print(f"Train MAE: {mae(clf.predict(X_train), y_train)}")

print(f"Test MAE: {mae(clf.predict(X_test), y_test)}")
# import shap

# explainer = shap.TreeExplainer(model1)

# shap_values = explainer.shap_values(X_test)

# shap.summary_plot(shap_values, X_test)#, plot_type="bar")
# import shap

# explainer = shap.TreeExplainer(model1)

# shap_values = explainer.shap_values(X_test)

# shap.summary_plot(shap_values, X_test, plot_type="bar")
# import eli5

# from eli5.sklearn import PermutationImportance

# perm = PermutationImportance(model1, random_state=1).fit(X_test, y_test)

# eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# from pdpbox import pdp, info_plots

# import matplotlib.pyplot as plt

# base_features = X.columns.values.tolist()

# for column in X.columns:

#     feat_name = column

#     pdp_dist = pdp.pdp_isolate(model=model1, dataset=X_test, model_features=base_features, feature=feat_name)

#     pdp.pdp_plot(pdp_dist, feat_name)

#     plt.show()