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

fire_data.rename(columns={"acq_date":"Date"}, inplace=True)

fire_data["WEI Value"] = 0

fire_data['month'] = fire_data['Date'].apply(lambda x:int(x.split('-')[1]))

wei = pd.read_excel('/kaggle/input/weekly-economic-index-wei-federal-reserve-bank/Weekly Economic Index.xlsx')

wei.drop('WEI as of 7/28/2020',axis=1,inplace=True)

wei = wei.set_index("Date")

from tqdm import tqdm

for index in tqdm(range(len(fire_data))):

    fire_date = (fire_data["Date"][index]) 

    fire_date = parser.parse(fire_date)

    min_wei_date_value = wei.iloc[wei.index.get_loc(fire_date,method='nearest')]["WEI"]

    fire_data.loc[index, "WEI Value"] = min_wei_date_value

fire_data['daynight'] = fire_data['daynight'].map({'D':0,'N':1})

fire_data.drop('instrument', axis=1, inplace=True)
x = fire_data[['latitude','longitude','month','brightness','scan','track',

               'acq_time','bright_t31','daynight','frp', 'confidence']]

y = fire_data['WEI Value']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
import time

start = time.time()

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error as mae

model1 = GradientBoostingRegressor(n_estimators = 400, learning_rate=0.1,

                                   max_depth = 10, random_state = 0, loss = 'ls')

model1.fit(X_train, y_train)

print(f"Train MAE: {mae(model1.predict(X_train), y_train)}")

print(f"Test MAE: {mae(model1.predict(X_test), y_test)}")

end = time.time()

print(f"Took {end-start} seconds.")
import shap

explainer = shap.TreeExplainer(model1)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)#, plot_type="bar")
import shap

explainer = shap.TreeExplainer(model1)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
import eli5

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model1, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
from pdpbox import pdp, info_plots

import matplotlib.pyplot as plt

base_features = x.columns.values.tolist()

for column in x.columns:

    feat_name = column

    pdp_dist = pdp.pdp_isolate(model=model1, dataset=X_test, model_features=base_features, feature=feat_name)

    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()
for column1 in x.columns:

    for column2 in x.columns:

        if column1 != column2:

            try:

                inter1 = pdp.pdp_interact(model=model1, 

                                          dataset=X_test, 

                                          model_features=base_features, 

                                          features=[column1, column2])



                pdp.pdp_interact_plot(pdp_interact_out=inter1, 

                                      feature_names=[column1, column2], 

                                      plot_type='contour')

                plt.show()

            except:

                pass
import folium

from folium.plugins import HeatMap



# Create basic Folium crime map

crime_map = folium.Map(location=[47.5112,-122.257], 

                       tiles = "Stamen Terrain",

                       zoom_start = 9)



# Add data for heatmp 

data_heatmap = x[['latitude','longitude','frp']]

data_heatmap = [[row['latitude'],row['longitude']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10, 

        gradient = {.35: 'blue',.55: 'purple',.68:'lime',.78:'red'}).add_to(crime_map)

crime_map
# Create basic Folium crime map

crime_map = folium.Map(location=[47.5112,-122.257], 

                       tiles = "Stamen Terrain",

                       zoom_start = 9)



# Add data for heatmp 

data_heatmap = x[['latitude','longitude','confidence']]

data_heatmap = [[row['latitude'],row['longitude']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10, 

        gradient = {.35: 'blue',.55: 'purple',.68:'lime',.78:'red'}).add_to(crime_map)

crime_map