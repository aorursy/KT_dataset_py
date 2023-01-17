import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot

import cufflinks as cf

%matplotlib inline

pd.get_option('display.width')

pd.set_option('display.width', 120)

sns.set(rc={'figure.figsize':(12.7,8.27)})

pd.options.mode.chained_assignment = None
data = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
data.drop('Serial No.', inplace=True, axis=1)

data.rename({'Chance of Admit ': 'Chance of Admit', 'LOR ':'LOR'}, axis=1, inplace=True)
## Head of the data

data.head()
## General statistics of the data

data.describe()
## Correlation coeffecients 

data.corr()
## Correlation coeffecients heatmap

sns.heatmap(data.corr(), annot=True).set_title('Correlation Factors Heat Map', color='black', size='30')

# Isolating GRE Score data

GRE = pd.DataFrame(data['GRE Score'])

GRE.describe()
# # Probability Distribution

sns.distplot(GRE).set_title('Probability Distribution for GRE Test Scores', size='30')

plt.show()
# Correlation Coeffecients for GRE Score Test

GRE_CORR = pd.DataFrame(data.corr()['GRE Score'])

GRE_CORR.drop('GRE Score', axis=0, inplace=True)

GRE_CORR.rename({'GRE Score': 'GRE Correlation Coeff'}, axis=1, inplace=True)

GRE_CORR
# Isolating and describing TOEFL Score

TOEFL = pd.DataFrame(data['TOEFL Score'], columns=['TOEFL Score'])

TOEFL.describe()
# Probability distribution for TOEFL Scores

sns.distplot(TOEFL).set_title('Probability Distribution for TOEFL Scores', size='30')

plt.show()
# Isolating and describing the CGPA

CGPA = pd.DataFrame(data['CGPA'], columns=['CGPA'])

CGPA.describe()
sns.distplot(CGPA).set_title('Probability Distribution Plot for CGPA', size='30')

plt.show()
RES_Count = data.groupby(['Research']).count()

RES_Count = RES_Count['GRE Score']

RES_Count = pd.DataFrame(RES_Count)

RES_Count.rename({'GRE Score': 'Count'}, axis=1, inplace=True)

RES_Count.rename({0: 'No Research', 1:'Research'}, axis=0, inplace=True)

plt.pie(x=RES_Count['Count'], labels=RES_Count.index, autopct='%1.1f%%')

plt.title('Research', pad=5, size=30)

plt.show()
# Isolating and describing 

University_Rating = data.groupby(['University Rating']).count()

University_Rating = University_Rating['GRE Score']

University_Rating = pd.DataFrame(University_Rating)

University_Rating.rename({'GRE Score': 'Count'}, inplace=True, axis=1)

University_Rating
# Barplot for the distribution of the University Rating

sns.barplot(University_Rating.index, University_Rating['Count']).set_title('University Rating', size='30')

plt.show()
#Isolating and describing

SOP = pd.DataFrame(data.groupby(['SOP']).count()['GRE Score'])

SOP.rename({'GRE Score':'Count'}, axis=1, inplace=True)

SOP
# Barplot for SOP 

sns.barplot(SOP.index, SOP['Count']).set_title('Statement of Purpose', size='30')

plt.show()
LOR = pd.DataFrame(data.groupby(['LOR']).count()['GRE Score'])

LOR.rename({'GRE Score':'Count'}, axis=1, inplace=True)

LOR
# Distribution of the LOR

sns.barplot(LOR.index, LOR['Count']).set_title('Letter of Recommendation', size='30')

plt.show()
data['Chance of Admit']

sns.distplot(data['Chance of Admit']).set_title('Probability Distribution of Chance of Admit', size='30')

plt.show()
data.describe()['Chance of Admit']
COA_corr = pd.DataFrame(data.corr()['Chance of Admit'])

COA_corr.rename({'Chance of Admit': 'Correlation Coeffecient'}, axis=1, inplace=True)

COA_corr.drop('Chance of Admit', inplace=True)

COA_corr.sort_values(['Correlation Coeffecient'], ascending=False, inplace=True)

COA_corr_x = COA_corr.index

COA_corr_y = COA_corr['Correlation Coeffecient']

sns.barplot(y=COA_corr_x,x=COA_corr_y).set_title('Chance of Admit Correlation Coeffecients', size='30')

plt.show()
COA_corr
X = data.drop(['Chance of Admit'], axis=1)

y = data['Chance of Admit']
#Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X[['CGPA','GRE Score', 'TOEFL Score']] = scaler.fit_transform(X[['CGPA','GRE Score', 'TOEFL Score']])
#Splitting

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_predictions = lr.predict(X_test)
from sklearn.metrics import r2_score, mean_squared_error

lr_r2 = r2_score(y_test, lr_predictions)

lr_mse = mean_squared_error(y_test, lr_predictions)

lr_rmse = np.sqrt(lr_mse)

print('Linear Regression R2 Score: {0} \nLinear Regression MSE: {1}, \nLinear Regression RMSE:{2}'.format(lr_r2, lr_mse, lr_rmse))
sns.set(rc={'figure.figsize':(12.7,8.27)})

sns.distplot((y_test - lr_predictions))

plt.title('Linear Regression (All Features) Residuals', fontdict={'fontsize':20}, pad=20)

plt.show()
sns.set(rc={'figure.figsize':(12.7,8.27)})

# sns.(y_test, lr_predictions)

sns.scatterplot(y_test, lr_predictions)

plt.show()
X_selected = X[['CGPA', 'GRE Score', 'TOEFL Score']]

X_sel_train, X_sel_test, y_train, y_test = train_test_split(X_selected, y, random_state=101)
lr_sel = LinearRegression()

lr_sel.fit(X_sel_train, y_train)

lr_sel_predictions = lr_sel.predict(X_sel_test)
lr_sel_r2 = r2_score(y_test, lr_sel_predictions)

lr_sel_mse = mean_squared_error(y_test, lr_sel_predictions)

lr_sel_rmse = np.sqrt(lr_sel_mse)

print('Linear Regression R2 Score: {0} \nLinear Regression MSE: {1}, \nLinear Regression RMSE:{2}'.format(lr_sel_r2, lr_sel_mse, lr_sel_rmse))
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import GridSearchCV
paramGrid = {'eta0':[0.01, 0.005, 0.1, 0.0005], 'n_iter':[500, 1000, 5000]}

sgd = SGDRegressor()

search = GridSearchCV(estimator=sgd, param_grid=paramGrid, n_jobs=-1)

search.fit(X_train, y_train)
search.best_estimator_
search.best_score_
sgd_predictions = search.predict(X_test)

np.sqrt(mean_squared_error(y_test, sgd_predictions))
def ensemble(x):

    pred_lr = lr.predict(x)

#     x_sel = x[['CGPA', 'GRE Score','TOEFL Score']]

#     pred_lr_sel = lr_sel.predict(x_sel)

    pred_sgd = search.predict(x)

    return (pred_lr + pred_sgd) / 2    
ensemble_predictions = ensemble(X_test)

r2_score(y_test, ensemble_predictions)

np.sqrt(mean_squared_error(y_test, ensemble_predictions))