
import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import datetime as datetime
from sklearn import preprocessing
import plotly.express as px
import plotly.offline as py
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
from random import random
from datetime import date
from datetime import datetime
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
np.set_printoptions(precision=3)
import os
os.listdir("../input/")
path = "../input/covid19-in-italy/"
df_regions = pd.read_csv(path+'covid19_italy_region.csv',decimal=",")
df_regions.head()
df_regions.describe()
df_regions.tail(10)
df_regions.head(10)
cols = ['HospitalizedPatients','IntensiveCarePatients','TotalHospitalizedPatients','HomeConfinement',
'CurrentPositiveCases','NewPositiveCases','Recovered','Deaths','TotalPositiveCases','TestsPerformed']
df_italy=df_regions.groupby(by=['Date'], as_index=False)[cols].sum()
df_italy.head()
bolge_isim = 'RegionName'
kod_isim = 'RegionCode'
alan_veri = 'Date'
variables = 'TotalPositiveCases'
df_regions[bolge_isim].unique()
df_trentino = df_regions[df_regions[kod_isim]==4].copy()
df_trentino.columns
df_trentino = df_trentino.groupby(by=[kod_isim,alan_veri],as_index=False).sum()
df_trentino[kod_isim]='Trentino Alto-Adige'
df_trentino = df_trentino.groupby(by=[kod_isim,alan_veri],as_index=False).sum()
df_trentino[bolge_isim]='Trentino Alto-Adige'
df_regions = df_regions[df_regions[kod_isim]!=4]
df_regions = pd.concat([df_regions,df_trentino],axis=0)
df_regions.reset_index()
def stringToDatetime(col, fmt):
    return pd.to_datetime(col, format=fmt)
def dfNorm(df):
    df['timestp']=stringToDatetime(df[alan_veri ],'%Y-%m-%d %H:%M:%S')
    df['timestp']=df['timestp'].dt.normalize()
    print('min date = ',min(df['timestp']),'max date = ',max(df['timestp']))
    return df
df_italy = dfNorm(df_italy)
df_regions = dfNorm(df_regions)
y_true  = df_italy[variables]
#y_true = df_regions[df_regions[kod_isim]=='Lombardia'][variables]
y_true = y_true.values
y_true
N_train = len(y_true)
N_train
N_test = 15
X_train = np.arange(0,N_train).reshape(-1, 1)
X_train
X_test = np.arange(N_train,N_train+N_test+1).reshape(-1, 1)
X_test
lrm = LinearRegression()
lrm.fit(X_train,y_true)
y_lrm_val = lrm.predict(X_train)
y_lrm_pre = lrm.predict(X_test)
def plotGraph(X_train, y_true, y_valid, X_test, y_pred):
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(X_train, y_true, color='blue')
    plt.plot(X_train, y_valid, color='red',linestyle='dashed')
    plt.plot(X_test, y_pred, color='red')
    x_conn = [X_train[len(X_train)-1],X_test[0]]
    y_conn = [y_true[len(X_train)-1],y_pred[0]]
    plt.plot(x_conn, y_conn, color='gray')
    y_conv = [y_valid[len(X_train)-1],y_pred[0]]
    plt.plot(x_conn, y_conv, color='gray',linestyle='dashed')
plotGraph(X_train, y_true, y_lrm_val, X_test, y_lrm_pre)
regs = df_regions[kod_isim].unique()
nc = 3
nr = int(len(regs)/nc)+1
fig, ax = plt.subplots(nrows=nr,ncols=nc,figsize=(18,30))
i = 0
j = 0
while i<nr:
    for j in range(nc):
        k = nc*i+j
        if (k<len(regs)):
            reg = regs[k]
            y_true_reg = df_regions[df_regions[kod_isim]==reg][variables].values
            lrr = LinearRegression()
            lrr.fit(X_train,y_true_reg)
            y_lrr_val = lrr.predict(X_train)
            y_lrr_pre = lrr.predict(X_test)
            ax[i,j].plot(X_train, y_true_reg, color='blue')
            ax[i,j].plot(X_train, y_lrr_val, color='red', linestyle='dashed')
            ax[i,j].plot(X_test, y_lrr_pre, color='red')
            ax[i,j].set_title(reg)            
    i = i + 1
fig = px.sunburst(df_regions.sort_values(by='NewPositiveCases', ascending=False).reset_index(drop=True), path=["Country", "RegionName"], values="NewPositiveCases", title='Confirmed Cases', color_discrete_sequence = px.colors.qualitative.Prism)
fig.data[0].textinfo = 'label+text+value'
fig.show()