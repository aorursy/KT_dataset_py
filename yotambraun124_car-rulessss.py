# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.tree import export_graphviz
import pydot
from subprocess import call
from IPython.display import Image

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv")
df.head()
df.keys()
len(df)
##this is pretty small amount of views##
df.isnull().sum()
df.info()
df_group_brand_mean_price = df.price.groupby(df.brand).mean()
df_group_brand_mean_price.plot.bar()
df_group_brand_max_price = df.price.groupby(df.brand).max()
df_group_brand_max_price.plot.bar()
df_group_brand_mileage_mean = df.mileage.groupby(df.brand).mean()
df_group_brand_mileage_mean.plot.bar()
#toyota seems to be more stronger type and peterbilt too
df_group_model_mileage_mean = df.mileage.groupby(df.model).mean()
df_group_model_mileage_mean.plot.bar()
plt.tick_params(labelsize=6)
df.model.value_counts()

df.state.value_counts().plot.bar()

change_to_dummy_brand = pd.get_dummies(df,columns=["model"])
change_to_dummy_brand
df_remove= change_to_dummy_brand.drop(columns=["Unnamed: 0","price", "brand", 'title_status', 'color', 'vin', 'lot', 'state', 'country', 'condition'],axis=1)
y=change_to_dummy_brand["price"]
x =df_remove
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
x_train.head()
regressor = LinearRegression()  
regressor.fit(x_train, y_train)
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:

y_pred = regressor.predict(x_test)

print(y_pred.shape,y_test.shape)

df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df_compare.head(50)
df1
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#try in ols method do linear regression
X = sm.add_constant(x_train)
model = sm.OLS(y_train,X)
results = model.fit()
results.params
model_summary = results.summary()
model_summary
#try anthoer method to hope to improvement - Randomforest regressor
randomforestregressormodel = RandomForestRegressor(n_estimators = 1000, random_state = 42)
randomforestregressormodel.fit(x_train,y_train)
y_predict_forest = randomforestregressormodel.predict(x_test)
y_predict_forest
mse = mean_squared_error(y_test,y_predict_forest)
rmse = np.sqrt(mse)
rmse
randomforestregressormodel.decision_path(x_test)
tree = randomforestregressormodel.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names = df_remove.keys(), rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

Image(filename = 'tree.png')





