import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import seaborn as sns

from datetime import datetime

import plotly.express as px

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns; sns.set(style='white')

#%config InlineBackend.figure_format = 'retina'

from mpl_toolkits.mplot3d import Axes3D



from scipy.cluster import hierarchy

from scipy.spatial.distance import pdist

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans



from sklearn import ensemble, decomposition

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split 

from sklearn.pipeline import Pipeline
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")

train.head()
test.head()
train.shape
test.shape
# Thanks for this data transformation to https://www.kaggle.com/nischaydnk/covid19-week5-visuals-randomforestregressor

last_date = train.Date.max()

df_countries = train[train['Date']==last_date]

df_countries = df_countries.groupby('Country_Region', as_index=False)['TargetValue'].sum()

df_trend = train.groupby(['Date','Country_Region'], as_index=False)['TargetValue'].sum()

df_trend = df_trend.merge(df_countries, on='Country_Region')

df_trend.rename(columns={'Country_Region':'Country', 'TargetValue_x':'Cases'}, inplace=True)

df_trend
# Find date start COVID19 growth

country_list = df_trend['Country'].unique()

country_stage = pd.DataFrame(columns = ['Country', 'COVID_start', 'COVID_max', 'COVID_now'])

for i in range(len(country_list)):

    country_i = df_trend[df_trend['Country'] == country_list[i]].reset_index(drop=True)

    country_stage.loc[i,'Country'] = country_list[i]                                                    # country name

    country_stage.loc[i,'COVID_start'] = country_i[country_i['Cases']!=0]['Cases'].cumsum().idxmin()    # date of the first cases

    country_stage.loc[i,'COVID_max'] = np.argmax(country_i['Cases'])                                    # date of the maximum

    country_stage.loc[i,'COVID_now'] = country_i.Cases[len(country_i)-1]/country_i.Cases.max()          # % from maximum at the end date
country_stage.sort_values(by='COVID_max')
country_stage_now = country_stage[['Country','COVID_now', 'COVID_max']].sort_values(by='COVID_now', ascending=False)
print("Cases now as % from maximum in each country")

plt.figure(figsize=(20,10))

plt.plot(range(len(country_stage_now.Country)), country_stage_now.COVID_now, marker='p');
data = country_stage[['COVID_start', 'COVID_max', 'COVID_now']]
# Thanks to https://www.kaggle.com/kashnitsky/topic-7-unsupervised-learning-pca-and-clustering

inertia = []

pca = PCA(n_components=2)

# fit X and apply the reduction to X 

x_3d = pca.fit_transform(data)

#x_3d=data

for k in range(1, 8):

    kmeans = KMeans(n_clusters=k, random_state=1).fit(x_3d)

    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, 8), inertia, marker='s');

plt.xlabel('$k$')

plt.ylabel('$J(C_k)$');
# Thanks to https://www.kaggle.com/arthurtok/a-cluster-of-colors-principal-component-analysis

# Set a 3 KMeans clustering

kmeans = KMeans(n_clusters=3, random_state=0)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_3d)

LABEL_COLOR_MAP = {0 : 'r',

                   1 : 'g',

                   2 : 'b'}



label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

plt.figure(figsize = (7,7))

plt.scatter(x_3d[:,0],x_3d[:,1], c= label_color, alpha=0.9)

plt.show()
# Thanks to https://www.kaggle.com/nischaydnk/covid19-week5-visuals-randomforestregressor

last_date = train.Date.max()

df_countries = train[train['Date']==last_date]

df_countries = df_countries.groupby('Country_Region', as_index=False)['TargetValue'].sum()

df_countries = df_countries.nlargest(20,'TargetValue')

df_trend = train.groupby(['Date','Country_Region'], as_index=False)['TargetValue'].sum()

df_trend = df_trend.merge(df_countries, on='Country_Region')

df_trend.rename(columns={'Country_Region':'Country', 'TargetValue_x':'Cases'}, inplace=True)

df_trend
df_trend_without_US = df_trend[df_trend['Country'] != 'US']

px.line(df_trend_without_US, x='Date', y='Cases', color='Country', title='COVID19 Total Cases growth for top 20 (without USA) worst affected countries')
# https://www.kaggle.com/kashnitsky/topic-7-unsupervised-learning-pca-and-clustering

tsne = TSNE(random_state=172)



X_tsne = tsne.fit_transform(data)



plt.figure(figsize=(12,10))

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=range(len(country_list)), 

            edgecolor='none', alpha=0.9, s=40,

            cmap=plt.cm.get_cmap('nipy_spectral', 3))

plt.colorbar()

plt.title('MNIST. t-SNE projection');
# Thanks to https://www.kaggle.com/kashnitsky/topic-7-unsupervised-learning-pca-and-clustering

distance_mat = pdist(data) 

# linkage — is an implementation if agglomerative algorithm

Z = hierarchy.linkage(distance_mat, 'single')

plt.figure(figsize=(20, 10))

dn = hierarchy.dendrogram(Z, color_threshold=7.4)
ID=train['Id']

FID=test['ForecastId']
train=train.drop(columns=['County','Province_State','Id'])

test=test.drop(columns=['County','Province_State','ForecastId'])
# Dates covert

da= pd.to_datetime(train['Date'], errors='coerce')

train['Date']= da.dt.strftime("%Y%m%d").astype(int)

da= pd.to_datetime(test['Date'], errors='coerce')

test['Date']= da.dt.strftime("%Y%m%d").astype(int)
# Encoding

l = LabelEncoder()

X = train.iloc[:,0].values

train.iloc[:,0] = l.fit_transform(X.astype(str))

X = train.iloc[:,4].values

train.iloc[:,4] = l.fit_transform(X)

l = LabelEncoder()

X = test.iloc[:,0].values

test.iloc[:,0] = l.fit_transform(X.astype(str))

X = test.iloc[:,4].values

test.iloc[:,4] = l.fit_transform(X)
# Split datasets

y_train=train['TargetValue']

x_train=train.drop(['TargetValue'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
# Training ExtraTreesRegressor model and data prediction

pip = Pipeline([('scaler2' , StandardScaler()),

                        ('ExtraTreesRegressor: ', ExtraTreesRegressor(n_estimators=5))]) # On competition I set n_estimators=500

pip.fit(x_train , y_train)

prediction = pip.predict(x_test)

acc=pip.score(x_test,y_test)

acc
# Forming output

output=pd.DataFrame({'id':FID,'TargetValue':pip.predict(test)})

a=output.groupby(['id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['id'])['TargetValue'].quantile(q=0.95).reset_index() 
# Quantiles 0.05%, 0.5%, 0.95%

a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']

a
# Submit

sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()