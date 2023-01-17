# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import statsmodels.api as sm

import scipy.stats as stats

import seaborn as sns

sns.set("talk","whitegrid",font_scale=1,font="sans-serif",color_codes=True)

from pylab import rcParams

plt.rcParams["figure.figsize"] = [10,10]

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv(r"../input/salary/Salary.csv")

df.head()
sns.heatmap(df.isnull())

plt.title("Detect Missing Values")

plt.show()
dfcorr = df.corr()

dfcorr
dfcov = df.cov()

dfcov
df.describe().transpose()
df.plot(kind="area",alpha=0.8)

plt.title("YearsExperience vs Salary Area Plot")

plt.show()
sns.boxplot(df["YearsExperience"])

plt.title("YearsExperience Box-and-Whisker Plot")

plt.show()
sns.boxplot(df["Salary"])

plt.title("Salary Box-and-Whisker Plot")

plt.show()
sns.distplot(df["YearsExperience"])

plt.title("YearsExperience Histogram")

plt.ylabel("Related YearsExperience Frequency")

plt.show()
sns.distplot(df["Salary"])

plt.title("Salary Histogram")

plt.ylabel("Related Salary Frequency")

plt.show()
sns.regplot(x="YearsExperience",y="Salary",data=df)

plt.title("YearsExperience vs Salary - Test data")

plt.show()
x = np.array(df["YearsExperience"])

y = np.array(df["Salary"])
x = x.reshape(-1,1)

y = y.reshape(-1,1)
x_constant = sm.add_constant(x)
model = sm.OLS(y,x_constant).fit()

model.predict(x_constant)

model.summary()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
lm = LinearRegression(fit_intercept=True)

lm.fit(x_train,y_train)
y_pred = lm.predict(x_test)

pd.DataFrame(y_pred, columns = ["Predicted_Salary"])
pd.DataFrame(y_test, columns = ["Actual_Salary"])
lm.intercept_
lm.coef_
MAE = metrics.mean_absolute_error(y_test,y_pred)

MSE = metrics.mean_squared_error(y_test,y_pred)

RMSE = np.sqrt(MSE)

R2 = metrics.r2_score(y_test,y_pred)

lmModelEvaluation = [[MAE,MSE,RMSE,R2]]

lmModelEvaluationdata = pd.DataFrame(lmModelEvaluation, columns = ("MAE","MSE","RMSE","R2"))

lmModelEvaluationdata
plt.scatter(x_test,y_pred,color="navy",alpha=0.8, s=200)

plt.plot(x_test,y_pred,color="red",lw=4)

plt.title("YearsExperience vs Salary - Test data")

plt.xlabel("YearsExperience")

plt.ylabel("Salary")

plt.show()
plt.scatter(x_train,y_train,color="navy",alpha=0.8, s=200)

plt.plot(x_test,y_pred,color="red",lw=2)

plt.title("YearsExperience vs Salary - Training data")

plt.xlabel("YearsExperience")

plt.ylabel("Salary")

plt.show()
plt.scatter(y_test,y_pred,color="navy",alpha=0.8, s=200)

plt.axhline(color="red",lw=4)

plt.title("Actual Salary vs Predicted Salary")

plt.xlabel("Actual Salary")

plt.ylabel("Predicted Salary")

plt.show()
residual = y_test - y_pred

model_residual = model.resid

model_fitted = model.fittedvalues

model_leverage = model.get_influence().hat_matrix_diag

model_norm_residual = model.get_influence().resid_studentized_internal

model_norm_residual_sqrt = np.array(np.abs(model_norm_residual))
np.mean(model_residual)


plt.scatter(y_pred,residual,color="navy",alpha=0.8, s=200)

plt.axhline(color="red",lw=4)

plt.title("Predicted Salary vs Residual Salary")

plt.xlabel("Predicted Salary")

plt.ylabel("Residual Salary")

plt.show()
fig = sm.graphics.qqplot(model_residual, dist=stats.norm, fit=True, line="45")

plt.title("Normal Probability Plot")

plt.show()
fig = sm.graphics.influence_plot(model, criterion="Cook's D")
sns.boxplot(model_residual)

plt.title("Residual Box-and-Whisker Plot")

plt.show()
sns.distplot(model_residual,bins=50)

plt.ylabel("Related Residual Frequency")

plt.title("Residual Histogram")

plt.show()
sns.residplot(model_fitted,model_residual)

plt.title("Fitted Salary vs Residuals Salary")

plt.xlabel("Fitted Salary")

plt.ylabel("Residual Values")

plt.show()
sns.residplot(model_fitted,model_norm_residual_sqrt)

plt.title("Fitted Salary vs Studentized Residuals")

plt.xlabel("Fitted Salary")

plt.ylabel("Studentized Residual Salary")

plt.show()
sns.residplot(model_leverage,model_residual)

plt.title("Leverage Salary vs Residual Salary")

plt.xlabel("Leverage Salary")

plt.ylabel("Residuals Salary")

plt.show()
sns.residplot(model_fitted,model_norm_residual_sqrt)

plt.title("Fitted Salary vs Studentized Residual Salary")

plt.xlabel("Fitted")

plt.ylabel("Standardized Residuals Salary")

plt.axhline(y=0.5, color="red")

plt.axhline(y=-0.5, color="red")

plt.show()
sns.residplot(model_leverage,model_norm_residual_sqrt)

plt.title("Leverage Salary vs Studentized Residual Salary")

plt.xlabel("Leverage")

plt.ylabel("Standardized Residuals Salary")

plt.axhline(y=0.5, color="red")

plt.axhline(y=-0.5, color="red")

plt.show()
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(model_residual)

plt.title("Residual Lag vs Autocorrelation Plot")

plt.show()
plot_acf(model_residual)

plt.show()
plot_pacf(model_residual)

plt.show()