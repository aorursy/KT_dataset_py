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

from statsmodels.sandbox.regression.predstd import wls_prediction_std

from sklearn.preprocessing import scale

from scipy import stats

sns.set_style("darkgrid")

plt.style.use("fivethirtyeight")

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/heart-disease-uci/heart.csv')
df.shape
df.columns
df.head()
df.tail()
df.isna().sum()
df.info()
df.describe().style.background_gradient(cmap='Reds')
#correlation in between the columns

df.corr().style.background_gradient(cmap="Greens")
df.skew().sort_values()
df.kurt().sort_values()
y = df['target']

x =  df[['chol','cp','thalach']]

model = sm.OLS(y, x)

results = model.fit()

print(results.summary())
print('Parameters: ', results.params)

print('Standard errors: ', results.bse)
sns.scatterplot(x = 'chol', y = 'trestbps', color = 'green', data = df)

plt.xlabel('cholestrol')            

plt.ylabel('trestbps') 

plt.title('Cholestrol Vs Trestbps');
fig = px.histogram(df, y="cp", x="age", color="sex",marginal="rug",hover_data=df.columns)

fig.show()
fig = px.scatter(df, x="trestbps", y="chol", color="age",size='cp', hover_data=['sex'])

fig.show()
fig = px.box(df, x="age", y="chol", points="all")

fig.show()
#thal 3 = normal, 6 = fixed defect, 7 = reversable defect (category feature)

fig, ax=plt.subplots(1,2,figsize=(20,8))

sns.countplot(x='thal',data=df,hue='target',palette='Set2',ax=ax[0])

ax[0].set_xlabel("number of major vessels colored by flourosopy")

df.thal.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Reds');

sns.jointplot(x = 'cp', y = 'age', kind = 'kde', color ='Green', data = df);
df1=df[['age','restecg']]
fig= go.Figure(go.Funnelarea(text=df1['age'],values=df1['restecg']))

fig.show()
df['heartratemax']=240-df['age']

df['heartrateratio']=df['thalach']/df['heartratemax']
df2=df[['age','trestbps','chol','thalach','oldpeak','ca','heartratemax','heartrateratio','target']]
sns.pairplot(df2)
fig,ax=plt.subplots(2,2,figsize=(20,8))

sns.boxenplot(y='chol',data=df,x='sex',hue='target',palette='twilight',ax=ax[0,0])

ax[0,0].set_title("Cholestrol V/S Sex");

sns.boxenplot(y='chol',data=df,x='cp',hue='target',ax=ax[0,1],palette='Spectral')

ax[0,1].set_title("Cholestrol V/S Chest Pain");

sns.swarmplot(y='chol',data=df,x='thal',hue='target',ax=ax[1,0],palette='copper')

ax[1,0].set_title("Cholestrol V/S Thalium stress test result")

sns.swarmplot(y='chol',data=df,x='oldpeak',hue='target',ax=ax[1,1],palette='Set2')

ax[1,1].set_title("Cholestrol V/S ST depression induced by exercise relative to rest");

plt.xticks(rotation=90)

plt.grid()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))

sns.violinplot(x="target", y="age", data=df,color = 'pink',ax=axes[0][0]).set_title('Age')

sns.swarmplot(x="target", y="age", data=df,ax = axes[0][0])



sns.violinplot(x="target", y="trestbps", data=df,color = 'pink',ax = axes[0][1]).set_title('Resting Blood Pressure')

sns.swarmplot(x="target", y="trestbps", data=df,ax = axes[0][1])



sns.violinplot(x="target", y="chol", data=df,color = 'pink',ax = axes[1][0]).set_title('Cholesterol')

sns.swarmplot(x="target", y="chol", data=df,ax = axes[1][0])



sns.violinplot(x="target", y="thalach", data=df,color = 'pink',ax = axes[1][1]).set_title('Max Heart Rate Achieved')

sns.swarmplot(x="target", y="thalach", data=df,ax = axes[1][1])



sns.violinplot(x="target", y="oldpeak", data=df,color = 'pink',ax = axes[2][0]).set_title('ST Depression Peak')

sns.swarmplot(x="target", y="oldpeak", data=df,ax = axes[2][0])



sns.violinplot(x="target", y="heartrateratio", data=df2,color = 'pink',ax = axes[2][1]).set_title('Peak Heart Rate to Max Heart Rate Ratio')

sns.swarmplot(x="target", y="heartrateratio", data=df2,ax = axes[2][1]);



plt.grid()

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
plt.figure(figsize=(30,10))

sns.heatmap(df.corr(), annot=True);
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(20, 8))

women = df[df['sex'] == 0]

men = df[df['sex'] == 1]



ax = sns.distplot(women[women['target'] == 1].age, bins=18, label = 'sick', ax = axes[0], kde =False, color="green")

ax = sns.distplot(women[women['target'] == 0].age, bins=40, label = 'not_sick', ax = axes[0], kde =False, color="red")

ax.legend()

ax.set_title('Female')



ax = sns.distplot(men[men['target']==1].age, bins=18, label = 'sick', ax = axes[1], kde = False, color="green")

ax = sns.distplot(men[men['target']==0].age, bins=40, label = 'not_sick', ax = axes[1], kde = False, color="red")

ax.legend()

ax.set_title('Male');
df["sex"].value_counts()
fig,ax=plt.subplots(1, 3, figsize=(20, 8))

sns.countplot(x = "sex", hue = "target", data = df, ax = ax[0])

sns.swarmplot(x = "sex", y = "age", hue = "target", data = df, ax = ax[1])

sns.violinplot(x = "sex", y = "age", hue= "target", split = True, data = df, ax=ax[2])

sns.despine(left=True)

plt.legend(loc="upper right");
df.thal.unique()

sns.boxenplot(x = "thal", y = "trestbps", data = df, hue="target")
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(17,10))

var3 = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']



for idx, feature in enumerate(var3):

    ax = axes[int(idx/4), idx%4]

    if feature != 'target':

        sns.countplot(x=feature, hue='target', data=df, ax=ax)