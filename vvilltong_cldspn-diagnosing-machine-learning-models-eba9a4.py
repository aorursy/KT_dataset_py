import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv, os
from datetime import datetime
from IPython.core.display import Image

# We can display plots in the notebook using this line of code
%matplotlib inline
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
data_diabetes = load_diabetes()
ddcol = ['Age','Sex','Body Mass Index','Avg Blood Pressure', 'Blood Serum 1',
         'Blood Serum 2','Blood Serum 3','Blood Serum 4','Blood Serum 5','Blood Serum 6']
dd = pd.DataFrame(data_diabetes['data'],
                  columns= ddcol)
dd['target'] = data_diabetes['target']
dd.head()
#Features have been mean centered and scaled by the standard deviation times n_samples (i.e. the sum of squares of each column totals 1).
dd.describe()
corr = dd.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot=True, fmt=".2f",
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(dd.loc[:, dd.columns != 'target'], dd['target'], test_size=0.33, random_state=420)
from sklearn.linear_model import LinearRegression
### code here
linreg = LinearRegression()
linreg.fit(X_train_reg , y_train_reg)
y_pred_reg = linreg.predict(X_test_reg)
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_pred_reg, y_test_reg)
print('Model Mean Absolute Error- {}'.format(MAE))
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_pred_reg, y_test_reg)
rmse = np.sqrt(mse)
print('Model Root Mean Square Error- {}'.format(rmse))
fig3 = plt.figure(2)
fig3.set_size_inches(10.5, 10.5)
frame1= fig3.add_axes((.1,.3,.8,.6))
plt.title('Predictions and Expected Values')
plt.plot(outcome['Age'], outcome['pred'],'-r')
plt.plot(outcome['Age'], outcome['actual'],'.b') 

plt.grid()

#Calculate difference

frame2= fig3.add_axes((.1,.1,.8,.2))
plt.title('Residual Error')
plt.plot(outcome['Age'], outcome['diff'], 'ob')
plt.plot(outcome['Age'], [0]*len(X_test_reg), 'r')
plt.ylim(-400, 300)
plt.grid()

sns.residplot(y_pred_reg , y_test_reg)
outcome = X_test_reg.copy()
outcome['actual'] = y_test_reg
outcome['pred'] = y_pred_reg
#Outliers: 
difference_clean = outcome['pred'] - outcome['actual']
difference_clean.plot.density()
difference_clean.plot.hist()
outcome['diff'] = outcome['pred'] - outcome['actual']
print('\n-----------------------------\n',outcome.sample(5),'\n-----------------------------\n')
outcome.describe()
outcome.columns
sns.pairplot(outcome , diag_kind="kde" , kind = 'reg', x_vars = ['Age', 'Sex', 'Body Mass Index', 'Avg Blood Pressure', 'Blood Serum 1',
       'Blood Serum 2', 'Blood Serum 3', 'Blood Serum 4', 'Blood Serum 5',
       'Blood Serum 6'] , y_vars = ['actual', 'pred', 'diff'])
X_train_reg.columns
# Rerun in statsmode: 
import statsmodels.api as sm
X = X_train_reg[[
    'Age', 
    'Sex', 'Body Mass Index', 'Avg Blood Pressure', 'Blood Serum 1',
       'Blood Serum 2'
                 , 'Blood Serum 3'
                 , 'Blood Serum 4'
    , 'Blood Serum 5'
    ,       'Blood Serum 6'
]]
y = y_train_reg
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()


#Just one variable

fig = plt.figure(figsize=(15,8))

# pass in the model as the first parameter, then specify the 
# predictor variable we want to analyze
fig = sm.graphics.plot_regress_exog(model, "Age", fig=fig)
fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(model, fig=fig)

from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd, iv_l, iv_u = sm.sandbox.regression.predstd .wls_prediction_std(model)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(x, y, 'o', label="data")
ax.plot(x, y_true, 'b-', label="True")
ax.plot(x, model.fittedvalues, 'r--.', label="OLS")
ax.plot(x, iv_u, 'r--')
ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');







