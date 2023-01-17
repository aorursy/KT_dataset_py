# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))#



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#eksik kütüphaneler eklendi



import matplotlib.pyplot as plt 



import plotly.express as px

import datetime

import seaborn as sns

import plotly.graph_objects as go

import warnings

warnings.filterwarnings('ignore')

import folium 

from folium import plugins

%matplotlib inline





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        



covid19verileri = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#dataset okutuldu.



covid19verileri.head()

covid19verileri.info()

#Veriler hakkında bilgiler gösterilmektedir
covid19verileri= covid19verileri.drop(['SNo'],axis=1)
covid_confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

covid_recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

covid_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')



covid19verileri['ObservationDate']=pd.to_datetime(covid19verileri['ObservationDate'])

covid19verileri['Last Update']=pd.to_datetime(covid19verileri['Last Update'])





grouping = covid19verileri.groupby('ObservationDate')['Last Update', 'Confirmed', 'Deaths'].sum().reset_index()







grouping.head()



fig = px.line(grouping, x="ObservationDate", y="Confirmed", 

              title="Süreç Boyunca Dünya Genelinde Onaylanmış Vakaların Analizi")

fig.show()

#Pandemi süreci boyunca dünyadaki vakaların onaylanma istatistikleri verilmiştir.


covid19_new = covid19verileri

covid19_new['Active'] = covid19_new['Confirmed'] - (covid19_new['Deaths'] + covid19_new['Recovered'])



line_data = covid19_new.groupby(['ObservationDate'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

line_data = line_data.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Active', 'Recovered', 'Deaths'], var_name='Case', value_name='Count')



fig = px.line(line_data, x='ObservationDate', y='Count', color='Case', title='Süreç Boyunca Dünyadaki Vakalar')

fig.show()



#Pandemi süreci boyunca dünyadaki vakaların, aktif taşıyıcı istatistikleri,ölüm istatistikleri,iyileşme istatistikleri ve onaylanma istatistikleri verilmiştir.
fig = px.scatter(covid19verileri, y="Deaths",x = "Recovered", color="Country/Region",

                 size='Confirmed')

fig.show()

#Grafite Kurtarılan hastaların ve buna oranla ölümlerin istatistigi ülkelere göre belirlenerek getirilmiştir.
covid19verileri.sort_values('Confirmed') #Tablomuzdaki veriler onaylanma 'confirmed' durumuna göre sıralanarak getirilmiştir.
import seaborn as sns
sns.pairplot(covid19verileri)
sns.countplot(covid19verileri['ObservationDate'])
sns.scatterplot(x= 'Confirmed',y ='Deaths', data = covid19verileri)
from sklearn.linear_model import LinearRegression



sns.set()



cols = covid19verileri.columns

covid19verileri.columns = [col.lower() for col in cols]
covid19verileri.rename(columns={

    'observationdate' : 'observation_date',

    'country/region' : 'country',

    'province/state' : 'province_state', 

    'last update' : 'last_update',

}, inplace=True)
covid19verileri['observation_date'] = pd.to_datetime(covid19verileri['observation_date'])



covid19verileri.sort_values('observation_date', inplace=True)
covid19verileri['diseased'] = covid19verileri['confirmed'] - covid19verileri['recovered'] - covid19verileri['deaths']



df_series = covid19verileri.groupby('observation_date').agg({

    'country' : 'nunique',

    'confirmed' : 'sum',

    'deaths' : 'sum',

    'recovered' : 'sum',

    'diseased' : 'sum',

})
covid19verileri.drop(['last_update', 'last_update'], axis=1, inplace=True)
for i in range(7, 15):

    df_series[f'confirmed_lag_{i}'] = df_series['confirmed'].shift(i)

    df_series[f'deaths_lag_{i}'] = df_series['deaths'].shift(i)

    df_series[f'recovered_lag_{i}'] = df_series['recovered'].shift(i)

    df_series[f'diseased_lag_{i}'] = df_series['diseased'].shift(i)
sns.set(style="white")



fig, ax = plt.subplots(figsize=(11, 9))



# Korelasyon matrisi oluşturulmuştur

corr = df_series.corr()



# Üst üçgen için bir maske oluşturuldu

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Özel bir farklı renk eşlemesi oluşturun

cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)



# Diyagonal ve üst kısım olmadan çizim korelasyon matrisi

sns.heatmap(corr, mask=mask, cmap=cmap, linewidths=.5)
df_series.dropna().shape




train_cols = [col for col in df_series.columns if 'deaths_lag_' in col] 



', '.join(train_cols) 



num_split = 7



X = np.log10(df_series.dropna()[train_cols])

y = np.log10(df_series.dropna()['deaths'])



X_train = X[:-num_split]

y_train = y[:-num_split]

X_test = X[-num_split:]

y_test = y[-num_split:]
model = LinearRegression()

model.fit(X_train, y_train)
#Cizim tahminleri ile gerçek değerlerin karşılaştırılması verilmiştir

#Bunun için üzerinde işlemlerin yapılacak değişkenler tanımlanmıştır



predictions = model.predict(X_test)



df_predictions = pd.DataFrame()

df_predictions['y_pred_log'] = predictions

df_predictions['y_true_log'] = y_test.values

df_predictions['y_pred'] = 10 ** predictions

df_predictions['y_true'] = 10 ** y_test.values



df_predictions['absolute_pct_error'] = abs((df_predictions['y_pred'] - df_predictions['y_true']) / df_predictions['y_true']) * 100
#Tanımlanan değişkenler ile günlük değerlere göre karşılaştırma yapılmış ve grafik ile görüntülenmiştir

fig, ax = plt.subplots(figsize=(14, 5))



ax.plot(y_train, 'bo--')

ax.plot(y_test, 'go--')

ax.plot(pd.Series(predictions, index = y_test.index), 'ro--')



plt.title('Günlük Değerler')

plt.show()
#Tanımlanan değişkenler ile mutlak değerlere göre karşılaştırma yapılmış ve grafik ile görüntülenmiştir



fig, ax = plt.subplots(figsize=(14, 5))



ax.plot(10 ** y_train, 'bo--')

ax.plot(10 ** y_test, 'go--')

ax.plot(10 ** pd.Series(predictions, index = y_test.index), 'ro--')



plt.title('Mutlak Değerler')

plt.show()
#Sonuç olarak tahminler ve hataların tablosu



df_predictions['y_pred'] = round(df_predictions['y_pred'])

df_predictions