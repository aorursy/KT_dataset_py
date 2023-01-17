# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# General tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For transformations and predictions
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

# For the tree visualization
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO

# For scoring
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_squared_error as mse


# For validation
from sklearn.model_selection import train_test_split as split

%matplotlib inline        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns


print(os.listdir("../input")) 
df=pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
import folium
from folium.plugins import HeatMap

# find the row of the house which has the highest price
maxpr=df.loc[df['price'].idxmax()]

# define a function to draw a basemap easily
def generateBaseMap(default_location=[47.5112, -122.257], default_zoom_start=9.4):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map

df_copy = df.copy()
df_copy['count'] = 1
basemap = generateBaseMap()
# add carton position map
folium.TileLayer('cartodbpositron').add_to(basemap)
s=folium.FeatureGroup(name='icon').add_to(basemap)
# add a marker for the house which has the highest price
folium.Marker([maxpr['lat'], maxpr['long']],popup='Highest Price: $'+str(format(maxpr['price'],'.0f')),
              icon=folium.Icon(color='green')).add_to(s)
# add heatmap
HeatMap(data=df_copy[['lat','long','count']].groupby(['lat','long']).sum().reset_index().values.tolist(),
        radius=8,max_zoom=13,name='Heat Map').add_to(basemap)
folium.LayerControl(collapsed=False).add_to(basemap)
basemap
print(df.shape)
print(df.info())
df.describe()
df.nunique()
ax = sns.scatterplot(x='sqft_living', y='price', hue='zipcode',data=df).sizes=(1, 1)
df_clean=df.drop(['zipcode','date','id'],axis=1)
df_clean.head()
#Target
df_clean.price.hist(bins=50);

log_price=np.log1p(df_clean.price)
log_price.hist(bins=50);
df_clean['log_price']=df_clean['price'].transform(func = lambda x : np.log1p(x))

df_clean=df_clean.drop(['price'],axis=1)
#df_clean.head()
#Condition-How good the condition is (Overall)
#Grade-overall grade given to the housing unit, based on King County grading system
features = ['condition', 'grade']
df[features].hist(figsize=(15, 5));
# Victoria- we need to change the plot i am not sure to what 
# Amit meant this (He was right, this one looks clearer):

f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=df_clean['grade'],y=df_clean['log_price'], ax=axes[0])
sns.boxplot(x=df_clean['condition'],y=df_clean['log_price'], ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='Grade', ylabel='Log_Price')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Condition', ylabel='Log_Price');


#Еffect of grade on the correlation between price and sqft_living
#ax=df_clean.groupby(['sqft_living','grade'])['log_price'].mean().unstack().plot(figsize=(14,6), marker='.',ls=' ',markersize=15)

#Еffect of condition on the correlation between price and sqft_living
#ax=df_clean.groupby(['sqft_living','condition'])['log_price'].mean().unstack().plot(figsize=(14,6), marker='.',ls=' ',markersize=10)
#View-Has been viewed
df['view'].value_counts()
plt.figure(figsize=(12,5))
ax=df_clean.sqft_living.hist(bins=100);
df_clean=df_clean.loc[df['sqft_living']<6000]
df_clean['bathrooms'].hist(figsize=(10, 5));
df_clean=df_clean.loc[df['bathrooms']<6]
#removing house with 0 bathrooms
df_clean=df_clean.loc[df['bathrooms']>0]

sns.catplot(x='bathrooms', y='log_price', data=df_clean);
df_clean['bedrooms'].value_counts()
df_clean=df_clean.loc[df['bedrooms']<10]
#Total floors (levels) in house
df['floors'].value_counts()
df_clean=df_clean.loc[df['floors']<=3]
f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=df_clean['bedrooms'],y=df_clean['log_price'], ax=axes[0])
sns.boxplot(x=df_clean['floors'],y=df_clean['log_price'], ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='Bedrooms', ylabel='Price')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='Floors', ylabel='Price');

#Аdding a new column, age of building
from datetime import date

def calculateAge(year): 
    today = date.today().year 
    age = today-year
    return age 
df_clean['building_age']=df_clean['yr_built'].apply(calculateAge)
df_clean.building_age.hist(bins=100);
#Аdding a new column, renovated indicator
def is_renovated(row):
    if row['yr_renovated']>0:
        return 1
    else:
        return 0
df_clean['is_renovated']=df_clean.apply (lambda row: is_renovated(row),axis=1)
ax=df_clean.groupby(['building_age','is_renovated'])['log_price'].mean().unstack().plot(figsize=(14,6))

#most of the houses were not renovated
#df_clean['yr_renovated'].value_counts()
df_clean['is_renovated'].value_counts(normalize=True)
#sqft_basement- We'll analize if the size of the basement has significant effect on the price or if it's a qustion of having a basement or not.
#Аdding a new column, basement indicator
def is_basement(row):
    if row['sqft_basement']>0:
        return 1
    else:
        return 0
df_clean['is_basement']=df_clean.apply (lambda row: is_basement(row),axis=1)
ax=df_clean.groupby(['sqft_living','is_basement'])['log_price'].mean().unstack().plot(figsize=(14,6))
#Histogram for log price by basement ind
ax = df_clean.hist(column='log_price', by='is_basement', bins=25, grid=False, figsize=(15,10), layout=(2,1), sharex=True, sharey=True,  zorder=2, rwidth=0.9)
df_clean=df_clean.drop(['yr_renovated', 'sqft_above', 'sqft_basement', 'yr_built'],axis=1)
df_clean.head()
#Еffect of waterfront on the correlation between price and sqft_living
#sns.lmplot('log_price', 'sqft_living', data=df_clean, 
         #  hue='waterfront');
ax = sns.scatterplot(x='log_price', y='sqft_living', hue='waterfront',data=df_clean)
df_clean['waterfront'].value_counts()
#Аdding a new column, indicator for average sqft living ratio
df_clean['sqft_liv15_ind'] =  df_clean['sqft_living'].div(df_clean['sqft_living15'])
#Аdding a new column, indicator for average sqft lot ratio
df_clean['sqft_lot15_ind'] =  df_clean['sqft_lot'].div(df_clean['sqft_lot15'])
df_clean=df_clean.drop(['sqft_lot15', 'sqft_living15'],axis=1)
fig=plt.figure(figsize=(19,5))

ax=fig.add_subplot(1,2,1)
ax.scatter(df_clean['log_price'],df_clean['sqft_liv15_ind'],edgecolors='white')
ax.set(xlabel='\nLog_Price',ylabel='\nSqft_Liv15_Ind')

ax=fig.add_subplot(1,2,2)
ax.scatter(df_clean['log_price'],df_clean['sqft_lot15_ind'], edgecolors='white')
ax.set(xlabel='\nLog_Price',ylabel='\nSqft_Lot15_Ind')
plt.figure(figsize=(15,15))
sns.heatmap(df_clean.corr(),annot=True,cmap='coolwarm')
plt.savefig('heatmap.png')

# Correlation between log_price and sqft_living is (0.69)
# Correlation between log_price and grade is (0.69)
# Correlation between log_price and bathrooms is (0.54)
# Correlation between log_price and bedrooms is (0.34)
sns.set()
cols = ['log_price', 'sqft_living', 'grade', 'bathrooms', 'view', 'bedrooms']
sns.pairplot(df_clean[cols], height = 2.5)
plt.show()
#df_clean=df_clean.drop(['long', 'lat'],axis=1)
df_clean.head()

print(df_clean.shape)

#Splitting the data
X = df_clean.drop('log_price', axis=1)
y = df_clean.log_price
X_train, X_test, y_train, y_test = split(X, y, random_state=1)
#Fitting the model

lin_model_1 = LinearRegression().fit(X_train, y_train)
list(zip(X_train.columns, lin_model_1.coef_))
#Predicting with the model
y_train_pred = lin_model_1.predict(X_train)
ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.plot(y_train, y_train, 'r')
RMSLE = msle(y_train, y_train_pred)**0.5
RMSLE
#Validating the model
y_test_pred = lin_model_1.predict(X_test)
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.plot(y_test, y_test, 'r')
RMSLE = msle(y_test, y_test_pred)**0.5
RMSLE
formula = 'log_price = ' + f'{lin_model_1.intercept_:.6f}'
for coef, feature in zip(lin_model_1.coef_, df_clean.drop('log_price', axis=1).columns[:4]):
    formula += f'{coef:+.6f}*{feature}'
print(formula)
from sklearn.tree import DecisionTreeRegressor
X = df_clean.drop('log_price',axis=1)
y =  df_clean['log_price']

model = DecisionTreeRegressor(max_leaf_nodes=50, min_samples_leaf=100)
model.fit(X, y)
!pip install pydot
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
dot_data = StringIO()  
export_graphviz(model, out_file=dot_data, feature_names=X.columns, leaves_parallel=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
Image(graph.create_png(), width=1500) 
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f'{feature:12}: {importance}')
X = df_clean.drop('log_price',axis=1).drop('lat',axis=1).drop('long',axis=1)
y =  df_clean['log_price']

model = DecisionTreeRegressor(max_leaf_nodes=50, min_samples_leaf=100)
model.fit(X, y)
dot_data = StringIO()  
export_graphviz(model, out_file=dot_data, feature_names=X.columns, leaves_parallel=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
Image(graph.create_png(), width=1500) 
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f'{feature:12}: {importance}')