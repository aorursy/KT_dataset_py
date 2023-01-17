# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt           # for the following plots I import the matplotlib library
pred_15 = pd.read_csv("../input/2015.csv");

pred_16 = pd.read_csv("../input/2016.csv");    # I store them separately to work easily w/o confusions

pred_17 = pd.read_csv("../input/2017.csv");
pred_16.info()  #checking values and the types
pred_16.head()      # what I see first is our data is sorted by Happiness Rank, 
pred_16.tail()  # most important value seems to be "Happiness Score"

pred_16.describe()  # getting more involved with the statistical side of data

corrmat = pred_16.corr()

sns.heatmap(corrmat, vmax=.8, square=True)
plt.clf()  # to clear our plots before re-creating them.



plt.figure(figsize=(16,15));  # to make it easy to divide and see



plt.subplot(3,1,1);      # 1st row 

plt.scatter(pred_16['Happiness Score'], pred_16['Freedom'], color='g');

plt.xlabel("Happiness Score");

plt.ylabel("Freedom");



plt.subplot(3,1,2);     # 2nd row

plt.scatter(pred_16['Happiness Score'], pred_16['Trust (Government Corruption)'], color='b');

plt.xlabel("Happiness Score");

plt.ylabel("Trust (Government Corruption)");



plt.subplot(3,1,3);     # 3rd row

plt.scatter(pred_16['Happiness Score'], pred_16['Generosity'], color='r');

plt.xlabel("Happiness Score");

plt.ylabel("Generosity");





plt.suptitle("FREEDOM / TRUST / GENEROSITY",fontsize=18)

plt.tight_layout()

plt.show()
import plotly.plotly as py 

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
pred_16[['Happiness Score','Region']].groupby('Region')['Happiness Score'].median().reset_index().sort_values('Happiness Score')
sns.set(rc={'figure.figsize':(10,10)})

#g = sns.FacetGrid(happiness, col="Region",  margin_titles=True, col_wrap=3)



#(g.map(plt.scatter, "Generosity","Happiness Score", edgecolor="w")).add_legend()

sns.scatterplot("Economy (GDP per Capita)","Happiness Score",hue='Region', data=pred_16)



from keras.layers import Dense

from keras.models import Sequential

from keras.regularizers import l1

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
pred_16.head()
pred_16.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score',

       'LCI', 'UCI', 'Economy', 'Family',

       'Health', 'Freedom', 'Trust',

       'Generosity', 'Dystopia_Residual']
y = pred_16.Happiness_Score

X = pred_16.drop('Happiness_Score', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
all_data = pd.concat((X_train.loc[:,'Country':'Region'],

                      X_test.loc[:,'Country':'Region']))
#log transform the target:

pred_16["Happiness_Score"] = np.log1p(pred_16["Happiness_Score"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = X_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
#creating matrices for sklearn:

X_train = all_data[:pred_16.shape[0]]

X_test = all_data[pred_16.shape[0]:]

y = pred_16.Happiness_Score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(10),

                     coef.sort_values().tail(10)])
import matplotlib



import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")