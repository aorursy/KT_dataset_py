import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler



from xgboost import XGBRegressor



from sklearn.model_selection import cross_val_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
training_data=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test_data=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")



training_data.head()
training_data.info()
training_data.isnull().sum()
training_data["Country_Region"].unique()
test_data.info()
test_data.isnull().sum()
training_data.drop(["Province_State","County","Id"],axis=1,inplace=True)

training_data.head()
test_data.drop(["Province_State","County","ForecastId"],axis=1,inplace=True)

test_data.head()
training_data.isnull().sum()
training_data.info()
training_data.describe()
training_data["Target"].value_counts()
training_data.hist(figsize=(20,15))
sns.set(style="darkgrid")

sns.countplot(x="Target",data=training_data)
plt.pie(x=training_data.groupby(by=["Target"])["TargetValue"].sum(),labels=training_data["Target"].unique(),autopct='%1.1f%%')
plt.pie(x=training_data.groupby(by=["Country_Region"])["TargetValue"].sum(),labels=training_data["Country_Region"].unique(),autopct='%.0f%%',

           radius=3.0,wedgeprops = {'linewidth': 0.0,"edgecolor":"k"},pctdistance=0.8,labeldistance=1.5,textprops={"fontsize":20},shadow=True,

           startangle=-90,rotatelabels=True)

plt.show()
last_date=training_data.Date.max()

df=training_data[training_data["Date"]==last_date]

df
df=df.groupby(by=["Country_Region"],as_index=False)["TargetValue"].sum()

df
countries=df.nlargest(5,"TargetValue")

countries
cases=training_data.groupby(by=["Date","Country_Region"],as_index=False)["TargetValue"].sum()

cases
cases=cases.merge(countries,on="Country_Region")

cases
plt.figure(figsize=(15,10))

sns.set(style="darkgrid")

sns.lineplot(x="Date",y="TargetValue_x",hue="Country_Region",data=cases)
training_data.corr()
training_data.drop(["Target"],inplace=True,axis=1)

test_data.drop(["Target"],inplace=True,axis=1)

training_data
le=LabelEncoder()
training_data["Country_Region"]=le.fit_transform(training_data["Country_Region"])
training_data.head()
test_data["Country_Region"]=le.fit_transform(test_data["Country_Region"])
test_data.head()
training_data.Date=training_data.Date.apply(lambda x:x.split("-"))
test_data.Date=test_data.Date.apply(lambda x:x.split("-"))
def month_day(dataset):

    month=[]

    day=[]

    for i in dataset.Date:

        month.append(int(i[1]))

        day.append(int(i[2]))

    dataset["month"]=month

    dataset["day"]=day

    dataset=dataset.drop(["Date"],axis=1)

    return dataset
training_data=month_day(training_data)

test_data=month_day(test_data)

training_data.head()
test_data.head()
scaler=MinMaxScaler()
y=training_data["TargetValue"].values
training_data.drop(["TargetValue"],axis=1,inplace=True)

training_data.head()
x=scaler.fit_transform(training_data)

x
xgb=XGBRegressor()
performance=cross_val_score(xgb,x,y,cv=10,scoring="neg_mean_absolute_error")

mae=-performance
mae
mae.mean()
test_data=scaler.transform(test_data)

test_data
xgb.fit(x,y)

prediction_xgb=xgb.predict(test_data)

prediction_xgb=np.around(prediction_xgb)

prediction_xgb
xgb_1500=XGBRegressor(n_estimators=1500,learning_rate=0.05,max_depth=15)
xgb_1500.fit(x,y)
prediction=xgb_1500.predict(test_data)
submission=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")

submission.head()
test_copy=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')
output = pd.DataFrame({'Id': test_copy.ForecastId  , 'TargetValue': prediction})

output.head()
a=output.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()
sub.info()