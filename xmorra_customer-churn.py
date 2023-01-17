#data wrangling & assesing 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#loading data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Models        
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#warnings
import warnings 
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
# Infos (value count and type for columns) summary of the dataframe
df.info()
n = []
l = df['TotalCharges'].tolist()
for i in l:
    try:
        n.append(float(i))
    except:
        n.append(0)
df['TotalCharges'] = n        
df['TotalCharges'].astype('float')
# Check for Null values
df.isnull().sum()
df.columns
catcolumns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod',]
for cat in catcolumns:
    print('{} unique values are [{}]\n'.format(cat,df[cat].unique()))
catToChange = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for cat in catToChange:
    df[cat] = df[cat].apply(lambda x: 'No' if x == 'No internet service' else x)
#let's make sure
for cat in catcolumns:
    print('{} unique values are [{}]\n'.format(cat,df[cat].unique()))
fig, axes =plt.subplots(4,4, figsize=(60,20), sharex=True,facecolor='white')
axes = axes.flatten()
for ax, catplot in zip(axes, catcolumns):
    sns.countplot(y=catplot, data=df, ax=ax, hue='Churn')
# Open the figure picture in a new tab to be able to zoom    
df.groupby('Churn')['tenure'].mean().plot(kind='barh',color=['lightblue','lightgreen']);
df.groupby('Churn')['MonthlyCharges'].mean().plot(kind='barh',color=['lightblue','lightgreen']);
df.groupby('Churn')['TotalCharges'].mean().plot(kind='barh',color=['lightblue','lightgreen']);
features = ['Contract','SeniorCitizen','Partner','InternetService','TechSupport','PaperlessBilling','PaymentMethod','OnlineSecurity','Dependents','tenure','MonthlyCharges','TotalCharges']
#Label Encoding for object to numeric conversion
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in features[:9]:
    df[feat] = le.fit_transform(df[feat].astype(str))

print (df.info())
#Target
Y = df['Churn'].values
#Inputs
X = df[features].values
#Split to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#Define the model
model = XGBClassifier(learning_rate = 0.1,n_estimators=200, max_depth=6)
#train the model
model.fit(X_train, y_train)
#Check training accuracy
trainingAccuracy =  metrics.accuracy_score(y_train,model.predict(X_train))
print("Training Accuracy: %.2f%%" % (trainingAccuracy * 100.0))
#Check testing accuracy
testingAccuracy =  metrics.accuracy_score(y_test, model.predict(X_test))
print("Testing Accuracy: %.2f%%" % (testingAccuracy * 100.0))
#Add the name of the features to the model
model.get_booster().feature_names = features
#Get the importance of each feature
importance = model.get_booster().get_score(importance_type="gain")
#Visualize the resutlt
importance
clf = MLPClassifier(max_iter=300).fit(X_train, y_train)
#Check training accuracy
trainingAccuracy =  metrics.accuracy_score(y_train,clf.predict(X_train))
print("Training Accuracy: %.2f%%" % (trainingAccuracy * 100.0))
#Check testing accuracy
testingAccuracy =  metrics.accuracy_score(y_test, clf.predict(X_test))
print("Testing Accuracy: %.2f%%" % (testingAccuracy * 100.0))