#importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
#Reading data
data = pd.read_csv("../input/heart-disease-prediction-using-logistic-regression/framingham.csv")
data.head()
#Pie chart of data
data['TenYearCHD'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',shadow=True)
#Plotting histogram of age with respect to TenYearCHD
plt.figure(figsize=(12, 6))
sns.countplot('age',hue='TenYearCHD',data=data)
plt.figure(figsize=(15, 12))

plt.subplot(3,3,1)
sns.countplot('male',hue='TenYearCHD',data=data)
plt.subplot(3,3,2)
sns.countplot('education',hue='TenYearCHD',data=data)
plt.subplot(3,3,3)
sns.countplot('currentSmoker',hue='TenYearCHD',data=data)
plt.subplot(3,3,4)
sns.countplot('BPMeds',hue='TenYearCHD',data=data)
plt.subplot(3,3,5)
sns.countplot('prevalentStroke',hue='TenYearCHD',data=data)
plt.subplot(3,3,6)
sns.countplot('prevalentHyp',hue='TenYearCHD',data=data)
plt.subplot(3,3,7)
sns.countplot('diabetes',hue='TenYearCHD',data=data)

plt.show()
plt.figure(figsize=(15, 12))

plt.subplot(3,3,1)
sns.boxplot(data['TenYearCHD'], data['totChol'], palette = 'viridis')
plt.subplot(3,3,2)
sns.boxplot(data['TenYearCHD'], data['sysBP'], palette = 'viridis')
plt.subplot(3,3,3)
sns.boxplot(data['TenYearCHD'], data['diaBP'], palette = 'viridis')
plt.subplot(3,3,4)
sns.boxplot(data['TenYearCHD'], data['BMI'], palette = 'viridis')
plt.subplot(3,3,5)
sns.boxplot(data['TenYearCHD'], data['heartRate'], palette = 'viridis')
plt.subplot(3,3,6)
sns.boxplot(data['TenYearCHD'], data['glucose'], palette = 'viridis')

plt.show()
#Checking if data is missing
data.info()
print("------------------------")
data.isnull().sum()
#Calculating the Missing Values % contribution in data
data_null = data.isna().mean().round(4) * 100
data_null.sort_values(ascending=False).head()
#A dendrogram is a diagram that shows the hierarchical relationship between features
#Here we are looking at dendogram of missing data
import missingno as msno
msno.dendrogram(data)
#Imputing the missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

data_new = pd.DataFrame(imputer.fit_transform(data))
data_new.columns = data.columns
data_new.index = data.index
data_new.isnull().sum()
#Plotting boxplot of features to find outliers
plt.figure(figsize=(15, 8))

plt.subplot(3,3,1)
sns.boxplot(data_new['age'],color='yellow')
plt.subplot(3,3,2)
sns.boxplot(data_new['cigsPerDay'],color='yellow')
plt.subplot(3,3,3)
sns.boxplot(data_new['totChol'],color='yellow')
plt.subplot(3,3,4)
sns.boxplot(data_new['sysBP'],color='yellow')
plt.subplot(3,3,5)
sns.boxplot(data_new['diaBP'],color='yellow')
plt.subplot(3,3,6)
sns.boxplot(data_new['BMI'],color='yellow')
plt.subplot(3,3,7)
sns.boxplot(data_new['heartRate'],color='yellow')
plt.subplot(3,3,8)
sns.boxplot(data_new['glucose'],color='yellow')

plt.show()
"""
Z-score:
This score helps to understand if a data value is greater or smaller than mean and how far away it is from the mean.
If the z score of a data point is more than 3/-3, it indicates that the data point is quite different from the other data points. Such a data point can be an outlier.
"""

from scipy import stats
z = np.abs(stats.zscore(data_new))
threshold = 3
print(np.where(z > 3)) # The first array contains the list of row numbers and second array respective column numbers

#Removing outliers
data_new = data_new[(z < 3).all(axis=1)]
#Plotting boxplot of features to find outliers
plt.figure(figsize=(15, 8))

plt.subplot(3,3,1)
sns.boxplot(data_new['age'],color='yellow')
plt.subplot(3,3,2)
sns.boxplot(data_new['cigsPerDay'],color='yellow')
plt.subplot(3,3,3)
sns.boxplot(data_new['totChol'],color='yellow')
plt.subplot(3,3,4)
sns.boxplot(data_new['sysBP'],color='yellow')
plt.subplot(3,3,5)
sns.boxplot(data_new['diaBP'],color='yellow')
plt.subplot(3,3,6)
sns.boxplot(data_new['BMI'],color='yellow')
plt.subplot(3,3,7)
sns.boxplot(data_new['heartRate'],color='yellow')
plt.subplot(3,3,8)
sns.boxplot(data_new['glucose'],color='yellow')

plt.show()
#Looking at some of the properties such as mean and std of each feature
data_new.describe()
#Viewing class distribution
plt.figure(figsize=(6, 4))
sns.countplot('TenYearCHD', data=data_new)
plt.title('Class Distributions')
#We will use Robust scaler for scaling as it is less prone to outliers
from sklearn.preprocessing import RobustScaler
sc = RobustScaler()

data_new['age'] = sc.fit_transform(data_new['age'].values.reshape(-1,1))
data_new['cigsPerDay'] = sc.fit_transform(data_new['cigsPerDay'].values.reshape(-1,1))
data_new['totChol'] = sc.fit_transform(data_new['totChol'].values.reshape(-1,1))
data_new['sysBP'] = sc.fit_transform(data_new['sysBP'].values.reshape(-1,1))
data_new['diaBP'] = sc.fit_transform(data_new['diaBP'].values.reshape(-1,1))
data_new['BMI'] = sc.fit_transform(data_new['BMI'].values.reshape(-1,1))
data_new['heartRate'] = sc.fit_transform(data_new['heartRate'].values.reshape(-1,1))
data_new['glucose'] = sc.fit_transform(data_new['glucose'].values.reshape(-1,1))

data_new.head()
X = data_new.iloc[:,:-1].values
y = data_new.iloc[:, -1].values
#SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

over = SMOTE()
under = RandomUnderSampler()
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

X, y = pipeline.fit_resample(X, y)
print(X.shape, y.shape)

#reshaping
y = y.reshape(len(y), 1)
print(X.shape, y.shape)
#Viewing class distribution after resampling
df_temp = {'TenYearCHD' : y[:,0]}
df = pd.DataFrame(df_temp)

plt.figure(figsize=(6, 4))
sns.countplot('TenYearCHD', data = df)
plt.title('Class Distributions after resampling')
#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
y_train = y_train.reshape(len(y_train))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))