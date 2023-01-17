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
data17 = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

data19 = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

data15 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")

data16 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

data18 = pd.read_csv("/kaggle/input/world-happiness/2018.csv")
data16.info()
data16.describe()
data16.corr()
import matplotlib.pyplot as plt

import seaborn as sns



# plt.figure(figsize=(15,13))

sns.heatmap(data16.corr(),annot=True,)
print(data15.columns)



temp = data15



d15 = temp[['Country',"Happiness Rank",

            'Happiness Score',

            'Economy (GDP per Capita)',

            'Family',

            'Health (Life Expectancy)',

            'Freedom',

            'Trust (Government Corruption)',

            'Generosity']].copy()



d15["Year"] = 2015

d15 = d15.rename(columns={'Country':'Country',

                          "Happiness Rank": "Rank",

                          'Happiness Score': "Score",

                          'Economy (GDP per Capita)':"GDP",

                          'Family':"Social_sup",

                          'Health (Life Expectancy)':"Life_exp",

                          'Freedom':"Freedom",

                          'Trust (Government Corruption)':"Corruption",

                          'Generosity':"Generosity"})

d15
d15[d15["Country"]=="Turkey"]
print(data16.columns)



temp = data16



d16 = temp[['Country',

            "Happiness Rank",

            'Happiness Score',

            'Economy (GDP per Capita)',

            'Family', 'Health (Life Expectancy)',

            'Freedom',

            'Trust (Government Corruption)',

            'Generosity']].copy()



d16["Year"] = 2016

d16 = d16.rename(columns={'Country':'Country',

                          "Happiness Rank": "Rank",

                          'Happiness Score': "Score",

                          'Economy (GDP per Capita)':"GDP",

                          'Family':"Social_sup",

                          'Health (Life Expectancy)':"Life_exp",

                          'Freedom':"Freedom",

                          'Trust (Government Corruption)':"Corruption",

                          'Generosity':"Generosity"})

d16
print(data17.columns)



temp = data17

d17 = temp[['Country',

            "Happiness.Rank",

            'Happiness.Score',

            'Economy..GDP.per.Capita.',

            'Family', 'Health..Life.Expectancy.',

            'Freedom',

            'Trust..Government.Corruption.',

            'Generosity']].copy()

d17["Year"] = 2017

d17 = d17.rename(columns={'Country':'Country',

                          "Happiness.Rank": "Rank",

                          'Happiness.Score': "Score",

                          'Economy..GDP.per.Capita.':"GDP",

                          'Family':"Social_sup",

                          'Health..Life.Expectancy.':"Life_exp",

                          'Freedom':"Freedom",

                          'Trust..Government.Corruption.':"Corruption",

                          'Generosity':"Generosity"})

d17
print(data18.columns)



temp = data18

d18 = temp[['Country or region',

            "Overall rank",

            'Score',

            'GDP per capita',

            'Social support',

            'Healthy life expectancy',

            'Freedom to make life choices',

            'Perceptions of corruption',

            'Generosity']].copy()

d18["Year"] = 2018

d18 = d18.rename(columns={'Country or region':'Country',

                          "Overall rank": "Rank",

                          'Score': "Score",

                          'GDP per capita':"GDP",

                          'Social support':"Social_sup",

                          'Healthy life expectancy':"Life_exp",

                          'Freedom to make life choices':"Freedom",

                          'Perceptions of corruption':"Corruption",

                          'Generosity':"Generosity"})

d18
print(data19.columns)



temp = data19

d19 = temp[['Country or region',

            "Overall rank",

            'Score',

            'GDP per capita',

            'Social support',

            'Healthy life expectancy',

            'Freedom to make life choices',

            'Perceptions of corruption',

            'Generosity']].copy()

d19["Year"] = 2019

d19 = d19.rename(columns={'Country or region':'Country',

                          "Overall rank": "Rank",

                          'Score': "Score",

                          'GDP per capita':"GDP",

                          'Social support':"Social_sup",

                          'Healthy life expectancy':"Life_exp",

                          'Freedom to make life choices':"Freedom",

                          'Perceptions of corruption':"Corruption",

                          'Generosity':"Generosity"})

d19
df = d15.append([d16,d17,d18,d19]).reset_index(drop=True)



# df = d15.append([d16,d17,d18,d19]).set_index(["Country","Year"])



df
df.isnull().sum()
df[df["Corruption"].isna()]
df.loc[489,"Corruption"] = float(df[df['Country']=='United Arab Emirates'].groupby('Country').mean()["Corruption"])
df.loc[489]
data_tr = df[df["Country"]=="Turkey"].reset_index(drop=True).copy()

data_tr
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt



x_tra = data_tr.drop(["Country","Rank","Score","Year"], axis=1)

y_tra = data_tr["Score"]



model_etr = ExtraTreesRegressor()

model_etr.fit(x_tra,y_tra)

print(model_etr.feature_importances_) 

feat_imp = pd.Series(model_etr.feature_importances_, index=x_tra.columns)

feat_imp.nlargest().plot(kind='barh')

plt.show()
print(sns.heatmap(data_tr.drop(["Year","Rank"],axis=1).corr(),annot=True))



cor_target = abs(data_tr.corr()["Score"])

relevant_features = cor_target[cor_target>0.5]

print(relevant_features)



print("\n\n",df[["GDP","Social_sup","Life_exp","Freedom","Corruption"]].corr())
g = sns.PairGrid(data_tr, x_vars=["Year", 'Score'], y_vars=['Freedom','Corruption','Social_sup', 'GDP',  'Life_exp'])

g.map(sns.regplot)
dat19 = d19.copy().drop(["Year"],axis=1)

dat19
from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt



x_d19 = dat19.drop(["Country","Rank","Score"], axis=1)

y_d19 = dat19["Score"]



model_etr = ExtraTreesRegressor()

model_etr.fit(x_d19,y_d19)

feat_imp = pd.Series(model_etr.feature_importances_, index=x_tra.columns)

feat_imp.nlargest().plot(kind='barh')

plt.show()
cor_target = abs(dat19.corr()["Score"])

relevant_features = cor_target[cor_target>0.5]

print(relevant_features)



print("\n\n",dat19[["GDP","Social_sup","Life_exp","Freedom","Corruption"]].corr())



# print("\n\n",dat19.corr())

sns.heatmap(dat19.corr(), annot=True)
dat19_hi = dat19[["Score","Country","GDP","Social_sup","Life_exp"]]

# dat19_hi = dat19_hi.nlargest(5,"Score")



print("Her sütun için en yüksek puan alan 5 ülke: \n\n")



for col in dat19_hi.drop(["Country"], axis=1).columns:

    print(f"{col}:\n",dat19_hi.nlargest(5,col),"\n\n")

    

import plotly.express as px





fig = px.scatter(dat19_hi, y="Score", x="Social_sup",

                 size="GDP", color="Life_exp",

                 hover_name="Country",

                 size_max=20,

                 labels= {"Score":"Happiness Score",

                         "Life_exp":"Life Expectancy",

                         "Social_sup":"Social Support",

                         "GDP":"GDP per Capita"},

                 title = "2019 Selected Happiness Index Chart")





fig.show()
# Example Machine Learning Procedure





data = dat19.copy().set_index("Country")

features = ['GDP', 'Social_sup', 'Life_exp', 'Freedom','Corruption', 'Generosity']



from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.metrics import mean_absolute_error



y = data.Score

x = data[features]

x_tr, x_val, y_tr, y_val = train_test_split(x,y,test_size=0.20)



print(x.describe())

print("\n\n",x.head(),"\n\n",y.head())





from sklearn.tree import DecisionTreeRegressor

model_tree = DecisionTreeRegressor()

model_tree.fit(x_tr,y_tr)

preds_tree = model_tree.predict(x_val)

print("\n\nDecision Tree: ",mean_absolute_error(y_val, preds_tree))



from sklearn.ensemble import RandomForestRegressor

model_forest = RandomForestRegressor()

model_forest.fit(x_tr,y_tr)

preds_forest = model_forest.predict(x_val)

print("\n\nRandom Forest: ",mean_absolute_error(y_val, preds_forest))



from xgboost import XGBRegressor

model_xgb = XGBRegressor()

model_xgb.fit(x_tr,y_tr)

preds_xgb = model_xgb.predict(x_val)

print("\n\nXGBoost: ", mean_absolute_error(preds_xgb, y_val))