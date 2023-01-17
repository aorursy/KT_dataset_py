#Import all required libraries


from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

%matplotlib inline

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.io import output_notebook
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

import warnings
warnings.filterwarnings('ignore')
output_notebook()
%matplotlib inline
#Loading the dataset
url = "../input/winequality-red.csv"
wine = pd.read_csv(url, sep=';')
#Check Dataset:
wine.head()
wine.info()
wine.describe()
"""From above we can see that minimun value of quality is 3 while maximum value is 8.
So now we will check the number of observations per quality value """

wine['quality'].value_counts().sort_index()

print(wine.shape)
print(wine.columns)
wine.rename(columns={'fixed acidity': 'fixedacidity','citric acid':'citricacid','volatile acidity':'volatileacidity','residual sugar':'residualsugar','free sulfur dioxide':'freesulfurdioxide','total sulfur dioxide':'totalsulfurdioxide'}, inplace=True)
print(wine.head())
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixedacidity', data = wine)
quality_values = ['Good','Average','Better']
q_range = (2,5,7,8)

wine['rating'] = pd.cut(wine['quality'], bins = q_range, labels = quality_values)
print(wine.head())
wine.rating.value_counts().sort_index()
wine.groupby('rating').mean()
# Checking the corelation between the independant variable and dependant variables:
correlation = wine.corr()
plt.figure(figsize=(12, 5))

sns.heatmap(correlation,annot = True, linewidth = 1, vmin = -1, cmap ="RdBu_r")
correlation['quality'].sort_values(ascending=False)
for column in wine.columns:
    if wine[column].dtype == type(object):
        le = LabelEncoder()
        wine[column] = le.fit_transform(wine[column])
#One hot Encoding
label_encoder = LabelEncoder()
wine['rating'] = label_encoder.fit_transform(wine.rating)
print(wine['rating'].head())
# Creating the X and y Values:
X = wine.drop('rating', axis =1)
y=wine['rating']
# Splitting train and test data sets:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3 , random_state = None)
#Standardizing the training data:
standardizer = StandardScaler()
standardizer.fit(X_train)
X_train = standardizer.transform(X_train)
X_test = standardizer.transform(X_test)
Log_reg = LogisticRegression()
#Fit the model and predict the X_test
Log_reg.fit(X_train, y_train)
pred_LR = Log_reg.predict(X_test)
#Print Classifiction Report:
print(classification_report(y_test, pred_LR))
RFClassifier = RandomForestClassifier()
#Fit and predict the model
RFClassifier.fit(X_train, y_train)
pred_RFC = RFClassifier.predict(X_test)
#Print Classification Report
print(classification_report(y_test, pred_RFC))
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
# Fit and predict our model
bnb.fit(X_train,y_train)
pred_bnb = bnb.predict(X_test)
# Print Classification Report
print(classification_report(y_test, pred_bnb))
svc = SVC()
# Fit and Predict the model
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
# Print Classification Report
print(classification_report(y_test, pred_svc))
sgd = SGDClassifier()
# Fit and Predict the model
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
# Print Classification Report
print(classification_report(y_test , pred_sgd))