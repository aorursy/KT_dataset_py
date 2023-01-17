# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.io.json import json_normalize
from scipy.stats import skew,kurtosis
import squarify
from wordcloud import WordCloud
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, download_plotlyjs
from plotly import tools
from plotly.tools import FigureFactory as ff
init_notebook_mode(connected=True)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def load_data(path):
    json_columns = ['device','geoNetwork','totals','trafficSource']
    df = pd.read_csv(path,converters= {column:json.loads for column in json_columns})
    for column in json_columns:
        json_df = json_normalize(df[column])
        json_df.columns = [f"{column}.{subcolumn}" for subcolumn in json_df.columns]
        df = df.drop(column,axis=1).merge(json_df,right_index=True,left_index=True)
    return df

%%time
df = load_data("../input/train.csv")
print("shape of train data:",df.shape)

#%%time
#test_df = load_data("../input/test.csv")
#print('shape of test data:',test_df.shape)
#for i in df.columns.values:
 #  if i not in test_df.columns:
  # print(i)
df.info()
df.shape
#date format
def date_format(data):
    data['date'] = data['date'].astype("str")
    data['date'] = data['date'].apply(lambda x:x[:4] + "-" + x[4:6] + "-"+ x[6:] )
    data['date'] = pd.to_datetime(data['date'])
    data['weekday'] = data['date'].dt.weekday
    data['day'] = data['date'].dt.day
    data['year'] = data['date'].dt.year
    data['month']= data['date'].dt.month
    return data
    
df = date_format(df)
df.head()
test_df = date_format(test_df)
total = df.isnull().sum().sort_values(ascending=False)
percent = total/df.shape[0]
Null_df = pd.concat([total,percent],axis=1,keys=['total','percent'])
Null_df[:15]
def review(data):
    #data['totals.pageviews'] = data['total.pageviews'].fillna(1)
    data['totals.newVisits'] = data['totals.newVisits'].fillna(0)
    data['totals.bounces'] = data['totals.bounces'].fillna(0)
    #data['totals.pageviews'] = data['totals.pageviews'].astype(int)
    data['totals.newVisits'] = data['totals.newVisits'].astype(int)
    data['totals.bounces'] = data['totals.bounces'].astype(int)
    return data
    
df = review(df)
df['totals.pageviews'].fillna(1,inplace=True)
def normalize(df):
    df['totals.hits'] = df['totals.hits'].astype(float)
    df['totals.transactionRevenue'] = df['totals.transactionRevenue'].astype(float)
    df['totals.transactionRevenue'].fillna(0.0,inplace =True)
    df['totals.transactionRevenue_log'] = (np.log(df[df["totals.transactionRevenue"] > 0]["totals.transactionRevenue"]))
    df['totals.transactionRevenue_log'].fillna(0,inplace=True)
    return df

df = normalize(df)
col = [column for column in df.columns if df[column].nunique()==1]
df = df.drop(col,axis=1)
null_columns = [column for column in Null_df.index & df.columns if Null_df.loc[column]['percent'] > 0.5]
df = df.drop(null_columns,axis=1)
null_columns
sns.distplot(df[df['totals.transactionRevenue_log'] > 0.0]['totals.transactionRevenue_log'])
df[df['totals.transactionRevenue_log'] > 0.0]['totals.transactionRevenue_log'].describe()
print('skewness:',skew(df[df['totals.transactionRevenue_log'] > 0.0]['totals.transactionRevenue_log']))
print('kurtosis:',kurtosis(df[df['totals.transactionRevenue_log'] > 0.0]['totals.transactionRevenue_log']))
sns.countplot(df['device.deviceCategory'])
print("top device browsers:")
print(df['device.browser'].value_counts()[:7])
plt.figure(figsize=(14,6))
ax = sns.countplot(x='device.browser',data=df[df['device.browser'].isin(df['device.browser'].value_counts()[:7].index)])
ax.set_title("Brower Usage",fontsize= 20)
ax.set_xlabel("Browser Name",fontsize = 15)
ax.set_ylabel("Count",fontsize = 15)
plt.show()
top_countries = round((df['geoNetwork.country'].value_counts()[:10]/len(df['geoNetwork.country']))*100,2)
plt.figure(figsize=(10,8))
squar = squarify.plot(sizes=top_countries.values, label=top_countries.index, 
                  value=top_countries.values,
                  alpha=.4)
squar.set_axis_off()
squar.set_title("top 10 countries in %",fontsize=20)
plt.show()
plt.figure(figsize=(17,6))
ax = sns.countplot(x = 'geoNetwork.subContinent' , data = df[df['geoNetwork.subContinent'].isin(df['geoNetwork.subContinent'].value_counts()[:15].index)],palette="hls")
ax.set_title("Top 15 most frequent Sub Continents" , fontsize=20)
ax.set_xlabel("Sub Continent" , fontsize = 15)
ax.set_ylabel("Count" ,fontsize = 15)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(15,6))
ax = sns.countplot(x='device.operatingSystem',data = df[df['device.operatingSystem'].isin(df['device.operatingSystem'].value_counts()[:8].index)])
ax.set_title("Usage of Operating System",fontsize=20)
ax.set_xlabel("Operating System",fontsize = 15)
ax.set_ylabel("Count",fontsize = 15)
plt.show()
#Usage of browsers usage by most frquent OS's

crosstab_eda = pd.crosstab(index=df[df['device.operatingSystem'].isin(df['device.operatingSystem'].value_counts()[:6].index.values)]['device.operatingSystem'], 
                          columns=df[df['device.browser'].isin(df['device.browser'].value_counts()[:5].index.values)]['device.browser'])
crosstab_eda.plot(figsize=(10,10),kind='bar',stacked=True)
plt.title("Most frequent OS's by Browsers of users")
plt.xlabel("Operational System Name", fontsize=19)
plt.ylabel("Count OS", fontsize=19)
plt.xticks(rotation=0)
plt.show()
crosstab_eda 
#Usage of browsers by most frquent subcontinent  region
crosstab_continent = pd.crosstab(index= df[df['geoNetwork.subContinent'].isin(df['geoNetwork.subContinent'].value_counts()[:15].index.values)]['geoNetwork.subContinent'],
                                columns = df[df['device.browser'].isin(df['device.browser'].value_counts()[:7].index.values)]['device.browser'])
crosstab_continent.plot(kind='bar',stacked=True,figsize=(12,8))
plt.xticks(rotation=45)
plt.title("TOP 10 Most frequent Subcontinents by Browsers used", fontsize=22)
plt.xlabel("Subcontinent Name", fontsize=19)
plt.ylabel("Count Subcontinent", fontsize=19)
plt.legend(loc=1, prop={'size': 12})
plt.show()
crosstab_continent 
ax = (df[df['totals.transactionRevenue_log'] > 0].groupby(['device.browser'])[['totals.transactionRevenue_log']].sum().sort_values(by='totals.transactionRevenue_log',ascending=False)[:5]).plot.bar()
ax.set_title("Total Revenue V/s browser",fontsize=20)
ax.set_xlabel("Browser",fontsize=15)
ax.set_ylabel('Total Revenue(Natural Log)',fontsize=15)
plt.xticks(rotation=45)
plt.show()
print("Top revenue generator countries")
((df[df['totals.transactionRevenue_log'] > 0].groupby(['geoNetwork.country'])[['totals.transactionRevenue_log']].sum()).sort_values(by='totals.transactionRevenue_log',ascending=False)[:10])
ax = (df[df['totals.transactionRevenue_log']>0].groupby(['channelGrouping'])[['totals.transactionRevenue_log']].sum().sort_values(by='totals.transactionRevenue_log',ascending=False)).plot.bar()
ax.set_title("Total Revenue V/s Channels Grouping",fontsize=20)
ax.set_xlabel("Channel Groups",fontsize=15)
ax.set_ylabel('Total Revenue(Natural Log)',fontsize=15)
plt.xticks(rotation=45)
plt.show()

def plot_revenue(cols_array):
    fig = tools.make_subplots(rows = 1, cols = 2, subplot_titles = ('Total Revenue', 'Total Revenue'))
    dataT =[] 
    dataXX = []
    for col in cols_array:
          dataT.append((df[df['totals.transactionRevenue_log']>0].groupby([col])[['totals.transactionRevenue_log']].sum().sort_values(by='totals.transactionRevenue_log',ascending=False)[:7].reset_index()))
          dataXX.append(go.Bar(x = dataT[cols_array.index(col)][col],y=dataT[cols_array.index(col)]['totals.transactionRevenue_log'],name=col ))
          fig.append_trace(dataXX[cols_array.index(col)],1,cols_array.index(col)+1)
    py.iplot(fig)
cols_t = ['device.browser','device.operatingSystem']
plot_revenue(cols_t)
cols_2 = [ 'geoNetwork.continent','geoNetwork.country']
plot_revenue(cols_2)
plot_revenue(['geoNetwork.region',
       'geoNetwork.subContinent'])
py.init_notebook_mode(connected=True)
dataL =  (df[df['totals.transactionRevenue_log']>0].groupby(['date'])[['totals.transactionRevenue_log']].sum().sort_values(by='totals.transactionRevenue_log',ascending=False)).reset_index()
dataX = go.Bar(x=dataL['date'],y=dataL['totals.transactionRevenue_log'],marker=dict(color = '#F57F17'),name='Total revenue')
py.iplot([dataX])
plt.figure(figsize=(10,7))
wordcloud = WordCloud(
                          max_words=30,
                          max_font_size=45
                         ).generate(' '.join(df['trafficSource.source']))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Most frequent used Traffic Source",fontsize = 20)
plt.show()
def clearRare(columnname, limit = 1000):
    vc = df[columnname].value_counts()
    common = vc > limit
    common = set(common.index[common].values)
    print("Set", sum(vc <= limit), columnname, "categories to 'other';", end=" ")
    df.loc[df[columnname].map(lambda x: x not in common), columnname] = 'other'
    print("now there are", df[columnname].nunique(), "categories in df")
clearRare('geoNetwork.networkDomain')
clearRare("device.browser")
clearRare("device.operatingSystem")
clearRare("geoNetwork.country")
clearRare("geoNetwork.city")
clearRare("geoNetwork.metro")
clearRare("trafficSource.campaign")
clearRare('geoNetwork.metro')
clearRare('geoNetwork.region')
clearRare('geoNetwork.subContinent')
df['trafficSource.campaign'].describe()
df.drop('trafficSource.campaign',axis=1,inplace=True)
df.drop('visitStartTime',axis=1,inplace=True)
train_x = df.drop(['date','fullVisitorId','sessionId','visitId','totals.transactionRevenue_log'],axis=1)
train_y = df['totals.transactionRevenue_log']
#train_x = train_x.drop('trafficSource.campaign',axis=1)
train_x.head()
train_x['device.isMobile'] = train_x['device.isMobile'].astype(str)
train_x['totals.pageviews'] = train_x['totals.pageviews'].astype(float)
categorical_col = train_x.select_dtypes(include=[np.object]).columns
numerical_col = train_x.select_dtypes(include=[np.number]).columns
categorical_col,numerical_col
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_col:
    train_col = list(train_x[col].values.astype(str))
    le.fit(train_col)
    train_x[col] = le.transform(train_col)
    
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for col in numerical_col :
    train_x[col] = (train_x[col] - np.mean(train_x[col]))/(np.max(train_x[col]) - np.min(train_x[col]))
from sklearn.model_selection import train_test_split
trainX,crossX,trainY,crossY = train_test_split(train_x.values,train_y,test_size=0.25,random_state = 20)
import lightgbm as lgb 

lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 500, "learning_rate" : 0.02,'max_bin':500, 
              "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9,'num_iteration':1200}
lgb_train = lgb.Dataset(trainX, label=trainY)
lgb_val = lgb.Dataset(crossX, label=crossY)
model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_val], early_stopping_rounds=150, verbose_eval=20)
