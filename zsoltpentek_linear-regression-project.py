import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
ecom_customers = pd.read_csv('../input/ecommerce-customers/Ecommerce Customers')
ecom_customers.head()
ecom_customers.info()
ecom_customers.describe()
sns.set_style(style='darkgrid')
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=ecom_customers, color='black')
sns.set_style(style='darkgrid')
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=ecom_customers, color='black')
sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=ecom_customers)
sns.pairplot(data=ecom_customers)
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=ecom_customers)
ecom_customers.columns
X = ecom_customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = ecom_customers['Yearly Amount Spent']
X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
from sklearn.linear_model import LinearRegression
lm.fit(X=X_train, y=y_train)
lm.coef_
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
predictions = lm.predict(X_test)
predictions
sns.scatterplot(x=y_test, y=predictions)
from sklearn import metrics
metrics.mean_absolute_error(y_true=y_test, y_pred=predictions)
metrics.mean_squared_error(y_true=y_test, y_pred=predictions)
np.sqrt(metrics.mean_squared_error(y_true=y_test, y_pred=predictions))
sns.distplot((y_test-predictions),bins=10)