# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

%matplotlib inline
df = pd.read_csv('../input/world-happiness/2015.csv',encoding='utf8', engine='python')
df.head()
df.info ()
df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.describe()
df[df['Country'].str.count('^[pP].*')>0]
plt.figure(figsize=(15,10))

sns.barplot(x=df ['Happiness Rank'],y = df['Region'], palette = sns.cubehelix_palette(70))

plt.xticks(rotation = 90) # slope of the words in the x axis 

plt.xlabel('Happiness Rank')

plt.ylabel('Region')

plt.title('Happiness Rank vs Region')

plt.show()
f,ax=plt.subplots(1,1,figsize=(30,6))

sns.countplot(x="Region",data=df, palette="muted")
sns.pairplot(data=df,kind='reg',size=5,x_vars=['Happiness Score'],y_vars=['Health (Life Expectancy)'])
plt.figure(figsize=(20,8))

corr = df.corr()

ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),

                 annot=True,square=True)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

)
sns.jointplot(df['Happiness Score'],df['Health (Life Expectancy)'],kind='kde',color='y',data=df)
sns.pairplot(df,hue = 'Region', vars = ['Standard Error', 'Economy (GDP per Capita)','Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual'] )
f,ax=plt.subplots(1,1,figsize=(25,6))

sns.kdeplot(df.loc[(df['Region']=='Western Europe'), 'Economy (GDP per Capita)'], color='r', shade=True, Label='Western Europe')

sns.kdeplot(df.loc[(df['Region']=='North America'), 'Economy (GDP per Capita)'], color='g', shade=True, Label='North America')

sns.kdeplot(df.loc[(df['Region']=='Australia and New Zealand'), 'Economy (GDP per Capita)'], color='b', shade=True, Label='Australia and New Zealand')

sns.kdeplot(df.loc[(df['Region']=='Southern Asia'), 'Economy (GDP per Capita)'], color='c', shade=True, Label='Southern Asia')

plt.xlabel('GDP per Capita') 
f,ax=plt.subplots(1,3,figsize=(25,5))

box1=sns.boxplot(data=df['Standard Error'],ax=ax[0],color='m')

ax[0].set_xlabel('Standard Error')

box1=sns.boxplot(data=df['Family'],ax=ax[1],color='m')

ax[1].set_xlabel('Family')

box1=sns.boxplot(data=df['Health (Life Expectancy)'],ax=ax[2],color='m')

ax[2].set_xlabel('Health (Life Expectancy)')
f,ax=plt.subplots(1,1,figsize=(25,6))

sns.violinplot(x="Region", y="Trust (Government Corruption)",data=df, palette="muted")
sns.pairplot(df)
Happiness_Score= pd.DataFrame(df["Happiness Score"])

Happiness_Score.describe()
happiness=[]

for i in Happiness_Score["Happiness Score"]:

    if i<5.5:

        happiness.append("UNHAPPY")

    else:

        happiness.append("HAPPY")



# Join our Hapiness_Score dataframe into the main dataframe

predicted_happiness = pd.DataFrame(happiness,columns=["PREDICTED_HAPPINESS"])

predicted_happiness = pd.DataFrame(predicted_happiness["PREDICTED_HAPPINESS"].astype('category'))



data = pd.concat([df,predicted_happiness],axis=1)
predicted_happiness["PREDICTED_HAPPINESS"].value_counts()
f, axes = plt.subplots(1, 1, figsize=(5, 4))

sns.countplot(predicted_happiness["PREDICTED_HAPPINESS"])

plt.xlabel("Predicted Happiness")

plt.ylabel("Number of Countries")
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz
# Remove any duplicate columns (if any)

data = data.loc[:,~data.columns.duplicated()]

data.shape
data.isnull().values.any()
data.isnull().sum()
#y=pd.DataFrame(data["predicted_happiness"])

y=predicted_happiness

y.shape
x=df.drop(['Country','Region','Happiness Score'],axis = 1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import classification_report

ran_class=RandomForestClassifier()

ran_class.fit(x_train,y_train)

ran_predict=ran_class.predict(x_test)

print(classification_report(y_test,ran_predict))

accuracy3=ran_class.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, ran_predict)

sns.heatmap(cm, annot= True)
from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import classification_report, confusion_matrix

logistic = LogisticRegression()

logistic.fit(x_train,y_train)

y_pred=logistic.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy=logistic.score(x_test,y_test)

print(accuracy*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
from sklearn.naive_bayes  import GaussianNB 

from sklearn.metrics import classification_report, confusion_matrix

nvclassifier = GaussianNB ()

nvclassifier.fit(x_train,y_train)

y_pred=nvclassifier .predict(x_test)

print(classification_report(y_test,y_pred))

acc=nvclassifier.score(x_test,y_test)

print(acc*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)