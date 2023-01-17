# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
data.head(10)
data.info()
round(100*(data.isnull().sum()/len(data.index)),2)
data.dropna(inplace=True)
data = data.drop(columns = 'society')
data['bhk'] = data['size'].str.split().str[0]
data['bhk'].dropna(inplace = True)
data['bhk'] = data['bhk'].astype(int)
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
data.total_sqft = data.total_sqft.apply(convert_sqft_to_num)
data = data[data.total_sqft.notnull()]
data.head(10)
data = data[~(data.total_sqft/data.bhk<300)]
data.shape
cont_ = data.select_dtypes(exclude='object')
fig = plt.figure(figsize = (10,8))
for index, col in enumerate(cont_):
    plt.subplot(3,2,index+1)
    sns.boxplot(y=cont_.loc[:,col])
fig.tight_layout(pad=1.0)
data = data.drop(data[data['bath']>6].index)
data = data.drop(data[data['bhk']>7.0].index)
data['price_per_sqft'] = data['price']*100000 / data['total_sqft']
data.head(10)
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)
    return df_out   
data = remove_pps_outliers(data)
data.shape
corr = data.corr()
plt.figure(figsize = (10,8))
sns.heatmap(corr, mask=corr<0.8, annot = True, cmap='Blues')
data.drop(columns=['availability', 'size', 'area_type'], inplace=True)
data.location = data.location.str.strip()
location_stats = data['location'].value_counts(ascending = False)
location_stats
location_stats_less_than_10 = location_stats[location_stats <= 10]
location_stats_less_than_10
data.location = data.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
data = data[data['location'] != 'other']
data['location'].value_counts()
data = data[data.bath < data.bhk+2]
data.shape
num_ = data.select_dtypes(exclude = 'object')
fig = plt.figure(figsize = (10,8))
for index, col in enumerate(num_):
    plt.subplot(3,2,index+1)
    sns.distplot(num_.loc[:, col], kde=False)
fig.tight_layout(pad=1.0)
plt.hist(data.price_per_sqft, rwidth = 0.8)
plt.xlabel("Price Per Square Feet", size=13)
plt.ylabel("Count", size=13)
plt.title("Price Per Sqft Distribution", size=20)
def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk==2)]
    bhk3 = df[(df.location == location) & (df.bhk==3)]
    fig = plt.figure(figsize=(12,8))
    fig, plt.scatter(bhk2.total_sqft, bhk2.price, color='black', label = '2 BHK', s=50)
    fig, plt.scatter(bhk3.total_sqft, bhk3.price, color='red', label = '3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price(Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
plot_scatter_chart(data, 'Whitefield')
plot_scatter_chart(data, 'Electronic City')
dummies = pd.get_dummies(data.location)
dummies.head(3)
data = pd.concat([data, dummies], axis='columns')
data.head(10)
data_1 = data.drop(columns= ['location','balcony', 'price_per_sqft'])
data_1
X = data_1.drop('price', axis=1)
y = data_1.price
sc = preprocessing.StandardScaler()
X1 = sc.fit_transform(X)

X1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=10)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
score = reg.score(X_test, y_test)
print("The accuracy score for LinearRegression is ", score)
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=300)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
score_rf = rf_reg.score(X_test, y_test)
print("The accuracy score for RandomForestRegression is ", score_rf)