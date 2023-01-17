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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.info()
df.describe()
df.isnull().sum()
sns.countplot(df['quality']);
df.columns
g = sns.FacetGrid(df, hue="quality",height=5)

g = g.map(sns.distplot, "alcohol")

plt.legend();
g = sns.FacetGrid(df, hue="quality",height=5)

g = g.map(sns.distplot, "pH")

plt.legend();
from statsmodels.formula.api import ols      # For calculation of Ordinary least squares for ANOVA

from statsmodels.stats.anova import _get_covariance,anova_lm # For n-way ANOVA

from statsmodels.stats.multicomp import pairwise_tukeyhsd # For performing the Tukey-HSD test

from statsmodels.stats.multicomp import MultiComparison # To compare the levels  independent variables with the 

import scipy.stats as stats 
df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol','quality'])
df_melt.head()
df_melt['variable'].unique()
df_melt.columns = ['index', 'treatments', 'value']
model = ols('value ~ C(treatments)', data=df_melt).fit()

anova_table = anova_lm(model, typ=2)

anova_table

#Type 1 2 and 3 yield same result if the data is balanced
### lets check for quality and alcohol
formula = 'alcohol ~ C(quality)'

model = ols(formula, df).fit()

aov_table = anova_lm(model)

aov_table
formula = 'pH ~ C(quality)'

model = ols(formula, df).fit()

aov_table = anova_lm(model)

aov_table
sns.pointplot(x='quality', y='alcohol', data=df,ci=0.95,color='g');

sns.pointplot(x='quality', y='pH', data=df,ci=0.95,color='r');
#Causal relation bwetween pH and quality

mc = MultiComparison(df['pH'], df['quality'])

mc_results = mc.tukeyhsd(alpha=0.05)

print(mc_results)
#causal relation bwetween alcohol and quality

mc = MultiComparison(df['alcohol'], df['quality'])

mc_results = mc.tukeyhsd(alpha=0.05)

print(mc_results)
import statsmodels.api as sm
X = df.drop('quality',axis=1)

y = df['quality']
model = sm.OLS(y, X).fit()

predictions = model.predict(X)
print(model.summary())
from scipy.stats import pearsonr

from statsmodels.compat import lzip

import statsmodels.stats.api as sms

from statsmodels.stats.outliers_influence import variance_inflation_factor
df[df.columns].corr(method='pearson')
plt.figure(figsize=(15,6))

sns.heatmap(df.corr(method='pearson'),annot=True);
#Formulae = (1/1-R^2)

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns
vif

# All VIF values are different , if two or more than two values have same VIF or variability the 
sns.residplot(predictions,y-predictions);
name = ['GQ', 'p-value']

test = sms.het_goldfeldquandt(y-predictions,X)

lzip(name, test)

#failed to reject null hypothesis so data is homoscedastic
from scipy.stats import shapiro

shapiro(np.abs(y-predictions))

# Error term is normally distributed as it rejects the null hypothesis
res = model.resid

fig = sm.qqplot(res,fit=True,line='45')

plt.show()

##Red line denotes normal line

##blue dots are the error terms
from sklearn.metrics import mean_squared_error

import math

print('MSE',mean_squared_error(y,predictions))

print('RMSE',math.sqrt(mean_squared_error(y,predictions)))
from sklearn.metrics import mean_absolute_error

print('MAE',mean_absolute_error(y,predictions))