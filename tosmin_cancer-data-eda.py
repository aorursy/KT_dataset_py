import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.preprocessing import scale
from scipy import stats
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/cancer-data-2017/cancer2017.csv', engine='python')
df.shape
#viewing columns...
df.columns
#redefine the column name for smooth analysis
df.columns = [c.strip() for c in df.columns.values.tolist()]
df.columns = [c.replace(' ','') for c in df.columns.values.tolist()] 
df.columns

df.rename(columns = {'Brain/nervoussystem':'Brain', 'Lung&bronchus':'Lung','Non-HodgkinLymphoma':'Lymphoma','Colon&rectum':'Colon'}, inplace = True)
df.head()
df.tail()
df.replace({r'[^\x00-\x7F]+':np.nan}, regex=True, inplace=True)
df.head()
df.info()
for i in range(0,df.shape[0]): 
    for j in range(1,df.shape[1]): 
        if ',' in str(df.iloc[i][j]): 
            df.iloc[i][j]=df.iloc[i][j].replace(',','') 
df.head()
df=df.apply(pd.to_numeric, errors='ignore')
df.info()
df1=df.ffill(axis=0)
df1.head(10).style.background_gradient(cmap='Reds')
stats=df1.describe().style.background_gradient(cmap='icefire')
stats
df1.corr().style.background_gradient(cmap='bone')
df1.skew().sort_values()
df1.kurt().sort_values()
var=['State', 'Brain', 'Femalebreast', 'Colon', 'Leukemia', 'Liver', 'Lung','Lymphoma', 'Ovary', 'Pancreas', 'Prostate']
plt.figure(figsize=(30,10))
corr = df1[var].corr()
sns.heatmap(data=corr, annot=True);
print('correlation in betwn Liver and Pancares:',pearsonr(df1.Liver, df1.Pancreas))
print(sm.OLS(df1.Liver, df1.Pancreas).fit().summary())
chart =sns.lmplot(y= 'Liver', x='Pancreas', data=df1)
print('correlation in betwn Lung and Prostate:',pearsonr(df1.Lung, df1.Prostate))
print(sm.OLS(df1.Lung, df1.Prostate).fit().summary())
chart =sns.lmplot(y= 'Lung', x='Prostate', data=df1)
print('correlation in betwn barin and Ovary:',pearsonr(df1.Brain, df1.Ovary))
print(sm.OLS(df1.Brain, df1.Ovary).fit().summary())
chart =sns.lmplot(y= 'Brain', x='Ovary', data=df1)
print('correlation in betwn Liver and Lung & bronchus:',pearsonr(df1.Liver, df1.Lung))
print(sm.OLS(df1.Liver, df1.Lung).fit().summary())
chart =sns.lmplot(y= 'Liver', x='Lung', data=df1)
var=['Brain', 'Femalebreast', 'Colon', 'Leukemia', 'Liver','Lung','Lymphoma', 'Ovary', 'Pancreas', 'Prostate']
sns.pairplot(df1,palette='coolwarm',hue= 'Lung')
var1=df1.loc[:, df.columns != 'State']
type(var1)
var1=list(var1)
type(var1)
z = df1[var1].groupby(df1['State']).sum() #plotting the state w.r.t cancer
z.T.plot(kind='barh', figsize=(20,10));
y=list(df1.columns)
x='State'
i=1
fig,ax=plt.subplots(nrows=5,ncols=2,figsize=(20,20))
for row in ax:
    for col in row:
        col.bar(df1[x],df1[y[i]])
        i=i+1
i=0
for ax in fig.axes:
    plt.title(var1[i])
    i=i+1
    plt.sca(ax)
    plt.xticks(rotation=90)
fig.tight_layout()
plt.show()
s=df1.Brain+df1.Femalebreast+df1.Colon+df1.Leukemia+df1.Liver+df1.Lung+df1.Lymphoma+df1.Ovary+df1.Pancreas+df1.Prostate

df2=df1.assign(s=s)
df3=df2[['State','s']]
df3.head(10).style.background_gradient(cmap='Oranges')
fig = go.Figure(go.Funnelarea(text =df3.State,values = df3.s))
fig.show()
fig,ax=plt.subplots(1,1,figsize=(10,5))
sns.distplot(df1[var1].mean(axis=1),bins=30,color='red');
fig,ax=plt.subplots(1,1,figsize=(10,5))
sns.distplot(df1[var1].std(axis=1),bins=30,color='green');
plt.rcParams['figure.figsize'] = (20, 8)
df1.plot(kind='bar', stacked=True);
labels = []
for r in df1.iloc[:,0]:
    labels.append(r)
plt.xticks(np.arange(50), labels, rotation=90);
plt.xlabel('Cancer');
plt.ylabel('count');
var=['Brain', 'Femalebreast', 'Colon', 'Leukemia', 'Liver','Lung','Lymphoma', 'Ovary', 'Pancreas', 'Prostate']
plt.figure(figsize=(20,8))
df1[var].boxplot()
plt.title("Numerical variables cancer", fontsize=20)
plt.show()
#lung vs liver
plt.figure(figsize=(20,8))
plt.xlabel("lung")
plt.ylabel("liver")
plt.suptitle("Joint distribution of lung vs liver", fontsize= 15)
plt.plot(df1['Lung'], df1['Liver'], 'bo', alpha=0.2)
plt.show()
#lung vs brest
plt.figure(figsize=(20,8))
plt.xlabel("Femalebreast")
plt.ylabel("liver")
plt.suptitle("Joint distribution of lung vs Femalebreast", fontsize= 15)
plt.plot(df1['Lung'], df1['Femalebreast'], 'bo', alpha=0.2)
plt.show()
#lung vs leukemia
plt.figure(figsize=(20,8))
plt.xlabel("lung")
plt.ylabel("Leukemia")
plt.suptitle("Joint distribution of lung vs Leukemia", fontsize= 15)
plt.plot(df1['Lung'], df1['Leukemia'], 'bo', alpha=0.2)
plt.show()

#fig.set_size_inches(20,10)
#sns.scatterplot(x='Lung',y='Leukemia',data=df1, size = 20);
