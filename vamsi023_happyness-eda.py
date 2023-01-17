# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression 

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')
_2015_report = pd.read_csv("../input/world-happiness/2015.csv")

_2016_report = pd.read_csv("../input/world-happiness/2016.csv")

_2017_report = pd.read_csv("../input/world-happiness/2017.csv")
_2015_report.describe()

_2016_report.describe()

_2017_report.describe()

_2015_report.head(2)

_2016_report.head(2)

_2017_report.head(2)
_2015_report["Year"] = 2015

_2016_report["Year"] = 2016

_2017_report["Year"] = 2017

clmns = ['Country', 'Happiness Rank', 'Happiness Score',

       'Whisker High','Whisker Low', 'Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom', 'Generosity', 'Trust (Government Corruption)',

       'Dystopia Residual', 'Year']

_2017_report.set_axis(clmns,axis=1)

_2017_report = _2017_report[['Country', 'Happiness Rank', 'Happiness Score',

       'Whisker High','Whisker Low', 'Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom',  'Trust (Government Corruption)','Generosity',

       'Dystopia Residual', 'Year']]

total_report = pd.concat([_2015_report,_2016_report,_2017_report],axis=0,sort=False).reset_index().drop('index',axis=1)

total_report.head(5)
pd.concat([total_report.describe(),

total_report.isnull().sum().to_frame().transpose().set_index(pd.Index(['missing']))],axis=0,sort=False)

total_report.isnull().sum().sort_values(ascending=False).apply(lambda x:x/len(total_report))
total_report_ = total_report[['Country', 'Region', 'Happiness Rank', 'Happiness Score', 'Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',

       'Generosity', 'Dystopia Residual', 'Year', ]]

regions = total_report_.Region.value_counts().index

region_countries={}

for reg in regions:

    region_countries[reg] = total_report_.query('Region == "'+reg+'"')['Country'].drop_duplicates().values

def impute_regions(countries):

    regions = []

    for country in countries:

        imputed = False

        for reg,con in region_countries.items():

            if country in con:

                regions.append(reg)

                imputed = True

        if imputed == False:

            regions.append("Asia")

    return regions

total_report_.loc[total_report_.Region.isnull(),'Region'] = impute_regions(total_report_.loc[total_report_.Region.isnull(),'Country'])

total_report_["Country"] = total_report_.Country.astype("category")

total_report_["Region"] = total_report_.Region.astype("category")
total_report_.columns

sns.pairplot(total_report_[['Happiness Score',

       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',

       'Freedom', 'Trust (Government Corruption)', 'Generosity',

       'Dystopia Residual']])
plt.figure(figsize=(12,8))

sns.heatmap(total_report_[['Happiness Score',

       'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',

       'Freedom', 'Trust (Government Corruption)', 'Generosity',

       'Dystopia Residual']].corr(),vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True,annot=True)
plt.figure(figsize=(8,5))

for c in total_report_.Region.cat.categories:

    sns.distplot(total_report_.loc[total_report_.Region==str(c),"Freedom"],hist=False,label=str(c),kde=True);

plt.figure(figsize=(8,5))

for c in total_report_.Region.cat.categories:

    sns.distplot(total_report_.loc[total_report_.Region==str(c),"Economy (GDP per Capita)"],hist=False,label=str(c),kde=True);
trust = np.sqrt(total_report['Trust (Government Corruption)'])



sns.jointplot(trust,total_report_["Happiness Score"])

sns.jointplot(total_report['Trust (Government Corruption)'],total_report_["Happiness Score"])



df = pd.concat([pd.DataFrame(trust),pd.DataFrame(total_report_["Happiness Score"])],axis=1)

df.corr()
encoder = LabelEncoder()

total_report_["Country"] = encoder.fit_transform(total_report_["Country"])

total_report_["Region"] = encoder.fit_transform(total_report_["Region"])

total_report_["Trust New"] = trust

X = total_report_[[

       'Economy (GDP per Capita)', 'Family', 

       'Freedom', 'Trust (Government Corruption)', 'Generosity',

       'Dystopia Residual']]

Y = total_report_["Happiness Score"]

X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size=0.2)

lmr = LinearRegression();

lmr.fit(X_train,Y_train)

y_pred = lmr.predict(X_test)

df = pd.DataFrame({'Actual': Y_test.as_matrix().flatten(), 'Predicted': y_pred.flatten()})

"R-2 value:"+str(np.sqrt(metrics.mean_squared_error(y_pred,Y_test)))

residuals = df["Actual"]-df["Predicted"]

coef = pd.Series(lmr.coef_).apply(lambda x: round(x,5))

pd.DataFrame(coef.as_matrix(),X.columns,columns=['Coeff'])

sns.jointplot(y_pred,Y_test)

sns.jointplot(y_pred,residuals)
_2018_report = pd.read_csv("../input/2018-report/WorldHappiness2018_Data.csv")

_2018_report.columns

y_new = lmr.predict(_2018_report[['GDP_Per_Capita',

       'Healthy_Life_Expectancy', 'Freedom_To_Make_Life_Choices', 'Generosity',

       'Perceptions_Of_Corruption', 'Residual']].dropna())

df_2 = pd.DataFrame({'Actual':_2018_report.Score[:-1].dropna().as_matrix().flatten(),'Predicted':y_new.flatten()})

residuals = df_2['Actual']-df_2['Predicted']

"R-2 score:"+str(np.sqrt(metrics.mean_squared_error(df["Actual"],df["Predicted"])))

sns.jointplot(df_2['Actual'],df_2['Predicted'])

sns.jointplot(df_2['Predicted'],residuals)