# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/earthquake-indian-subcontinent/Earthquake.csv',engine='python')

df.head()
len(df.index)
df.nunique()
df.drop_duplicates(subset ="time",keep = False, inplace = True) 
len(df.index)
df.fillna('')

#df.isnull().sum().sum()
df.columns
df['time'] = pd.to_datetime(df['time'])

df.head()
df = df.drop(columns = ['magType', 'nst','gap', 'dmin', 'rms', 'net', 'id', 'updated', 'type',

       'horizontalError', 'depthError', 'magError', 'magNst', 'status',

       'locationSource', 'magSource'])
#sns.pairplot(df,vars=df.columns[1:],)

sns.pairplot(df)
sns.violinplot(df['mag'],orient='v')
df[df['mag']==df['mag'].max()]
(df['mag']>=3.5).value_counts()
new_df=df.loc[df['mag'] >= 3.5].reset_index().drop(columns=['index'])

len(new_df.index)
new_df.head()
diction={}
new_df.drop(['place'], axis =1 , inplace = True)
sns.pairplot(new_df,vars=new_df.columns[1:])
sns.violinplot(new_df['mag'],orient='v')
new_df.plot(x='time',y='mag',figsize=(20,10))
for i in range(2000,2020,5):

    mask = (new_df['time'] > str(i+1)+'-1-1') & (new_df['time'] <= str(i+5)+'-12-31')

    diction.update({(str(i+1)+'-'+str(i+5)):new_df.loc[mask].reset_index().drop(columns=['index'])})
diction['2001-2005'].head()
mean_mag=[]

time=[]

interval=[]

sqrt_dE=[] 

b=[]

a=[]

niu=[]

delta_M=[]

max_mag=[]

for key in diction:

    interval+=[key]

    time+=[diction[key].time.max()-diction[key].time.min()]

    mean_mag+=[diction[key].mag.mean()]

    sqrt_dE+=[sum((10**(11.8+1.5*diction[key].mag))**0.5)]

    n=len(diction[key])

    Ni=[]

    for i in range(0,len(diction[key]),1):

        Ni+=[(diction[key].mag[i]<=diction[key].mag).sum()]

    sum_1=0

    for i in range(0,len(diction[key]),1):

        sum_1+=diction[key].mag[i]*np.log10(Ni[i])

    sum_mi=sum(diction[key].mag)

    sum_ni=0

    for i in range(0,len(diction[key]),1):

        sum_ni+=np.log10(Ni[i])

    sum_mi2=sum(diction[key].mag**2)

    b_temp=0

    b_temp=(n*sum_1-sum_mi*sum_ni)/(sum_mi**2-n*sum_mi2)

    b+=[b_temp]

    a_temp=0

    for i in range(0,len(diction[key]),1):

        a_temp+=(np.log(Ni[i])+b_temp*diction[key].mag[i])/n

    a+=[a_temp]

    niu_temp=0

    for i in range(0,len(diction[key]),1):

        niu_temp+=((np.log(Ni[i])-(a_temp-b_temp*diction[key].mag[i]))**2/(n-1))

    niu+=[niu_temp]

    delta_M+=[abs(diction[key].mag.max()-a_temp/b_temp)/10]

    max_mag+=[diction[key].mag.max()]

time=pd.to_timedelta(time, errors='coerce').days

sqrt_dE = [i / j for i, j in zip(sqrt_dE , time)]

df=pd.DataFrame({'period':interval,

                'T':time,

                'mean_mag':mean_mag,

                'Speed':sqrt_dE,

                'b':b,

                'niu':niu,

                'delta_M':delta_M,

                'max_mag': max_mag,

                #'Ptest':[0]*24,

                #'Ytest':[0]*24,

                })

df.head()
df.plot(x='period',y='mean_mag',figsize=(20,10))
df.plot(x='period',y='max_mag',figsize=(20,10))
df.to_csv(r'quakes_toTrain.csv', index = None, header=True)
import numpy as np

import pandas as pd

from pandas import read_csv

from matplotlib import pyplot

import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statsmodels.tsa.vector_ar.var_model import VAR

from sklearn.metrics import mean_squared_error

get_ipython().run_line_magic('matplotlib', 'inline')
df1 = pd.read_csv('/kaggle/working/quakes_toTrain.csv')

df1.dtypes

temp=[]

for i in range(len(df1.period)):

    temp += [pd.to_datetime(df1.period[i][:4] , format = '%Y')]

df1['period'] = temp;

data = df1.drop(['period'], axis=1)

data.index = df1.period

data.head()
plt.rcParams["figure.figsize"] = (20,3)

data.mean_mag.plot()

data.max_mag.plot()

pyplot.show()
df1.hist(figsize=(20,10))
autocorrelation_plot(data.mean_mag)

autocorrelation_plot(data.max_mag)

pyplot.show()
data.corr()


johan_test_temp = data



coint_johansen(johan_test_temp,-1,1)
train = data[:int(0.8*(len(data)))]

valid = data[int(0.8*(len(data))):]

model = VAR(endog=train)

model_fit = model.fit()

prediction = model_fit.forecast(model_fit.y, steps=len(valid))
model = VAR(endog=data)

model_fit = model.fit()

yhat = model_fit.forecast(model_fit.y, steps=3)

print(yhat)

temp_list=[]

for i in range(len(yhat)):

    yhat[i][2] = abs(yhat[i][2])

    yhat[i][0] = yhat[i][0].astype(int)

    df_temp=pd.DataFrame(yhat, columns =['T','mean_mag','Speed','b','niu','delta_M','max_mag'])

    years=pd.to_datetime('2016-1-1')

    temp_list += [years.replace(year = years.year+(i+1)*5)]

df_temp['period'] = temp_list

df_temp.index = df_temp.period

df_temp = df_temp.drop(['period'], axis=1)

data=data.append(df_temp)

data.head()
data.mean_mag.plot()

data.max_mag.plot()

pyplot.show()
data