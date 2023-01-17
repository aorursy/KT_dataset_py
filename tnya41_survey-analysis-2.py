import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statistics
df= pd.read_csv("../input/survey_13.csv")
df
df.tail(3)
df.info()
df.shape
df.columns
df.describe()
df.drop(['Timestamp','Institute name'], inplace= True ,axis= 1)
df.head()
df.dtypes
df.nunique()
df['CTC offered by Company in LPA during first job (as a Fresher)'].plot( kind= 'bar', title='CTC offered by Company in LPA during first job (as a Fresher)')
df['Valid Experience in months( Excluding Internship Experience)'].plot( kind= 'bar',title='Valid Experience in months( Excluding Internship Experience)')
plt.figure(figsize=(15,8))

sns.countplot(x='CTC offered by Company in LPA during first job (as a Fresher)',data=df)
### Plot for "year"
plt.figure(figsize=(15,8))
sns.countplot(x='Year of graduation',data=df)
# counting categories of "Experience in month"
plt.figure(figsize=(15,8))

sns.countplot(x='Valid Experience in months( Excluding Internship Experience)',data=df)
statistics.median(df['CTC offered by Company in LPA during first job (as a Fresher)'])
statistics.mean(df['CTC offered by Company in LPA during first job (as a Fresher)'])
labels=df['CTC offered by Company in LPA during first job (as a Fresher)'].value_counts().index
colors=['cyan','pink','orange','lightgreen','yellow','red','blue','green','teal','maroon','navy','gray','olive','purple','salmon','fuchsia']
explode=[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
values=df['CTC offered by Company in LPA during first job (as a Fresher)'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Analysis of CTC as Freshers',color='black',fontsize=10)
plt.show()
df.dtypes
# Important categorical variable to convert into dummy variables.
categorical_columns=['Highest Qualification','Year of graduation','Job Profile', 'Job Location-City']
df[categorical_columns].nunique()
# Get dummy variables for categorical variables
df = pd.get_dummies(data = df, columns = categorical_columns )

# Copying dataframe
df = df.copy()
df.head()
df.shape
X = df.drop(['CTC offered by Company in LPA during first job (as a Fresher)','Name of Company you worked for, as a Fresher'], axis=1)
y = df['CTC offered by Company in LPA during first job (as a Fresher)']

# ## 2.1) Multiple Linear Regression

# Importing Library for Linear Regression
from sklearn.linear_model import LinearRegression

# Fitting simple linear regression to the training data
regressor=LinearRegression()
LR_model=regressor.fit(X,y)
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LR_model, X = X, y = y, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy of Multiple Linear Regression ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))

# Importing Library for Linear regression
import statsmodels.api as sm

# train the model using training set
LR_model = sm.OLS(y, X).fit()
LR_model.summary()
# predicting the test set results
y_pred= LR_model.predict(X)
y_pred
# Calculate MAPE
def mape(y,y_pred):
    mape = np.mean(np.abs((y - y_pred)/y))*100
    return mape

mape(y,y_pred)
# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = LR_model.predict(X)
rmse_for_train = np.sqrt(mean_squared_error(y,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))
# Calculating RMSE for test data to check accuracy
y_pred= LR_model.predict(X)
rmse_for_test =np.sqrt(mean_squared_error(y,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))
# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y,y_pred)))