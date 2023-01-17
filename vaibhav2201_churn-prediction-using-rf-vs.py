import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import seaborn as sns
from sklearn.linear_model import LogisticRegression

rawdata=pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data=rawdata.copy()
data.shape
data.head(5)
data.info()

data['TotalCharges']=data['TotalCharges'].convert_objects(convert_numeric=True)
def uni(columnname):
    print(columnname,"--" ,data[columnname].unique())
dataobject=data.select_dtypes(['object'])
len(dataobject.columns)
for i in range(1,len(dataobject.columns)):
    uni(dataobject.columns[i])
    
def labelencode(columnname):
    data[columnname] = LabelEncoder().fit_transform(data[columnname])
for i in range(1,len(dataobject.columns)):
    labelencode(dataobject.columns[i])
data.info()
for i in range(1,len(dataobject.columns)):
     uni(dataobject.columns[i])

data.info()
df=data.copy()
dfl=data.copy()
unwantedcolumnlist=["customerID","gender","MultipleLines","PaymentMethod","tenure"]
df = df.drop(unwantedcolumnlist, axis=1)
features = df.drop(["Churn"], axis=1).columns
df_train, df_val = train_test_split(df, test_size=0.30)

print(df_train.shape)
print(df_val.shape)
df_train.isnull().sum()
df_val.isnull().sum()
df_train['TotalCharges'].fillna(df_train['TotalCharges'].mean(), inplace=True)
df_val['TotalCharges'].fillna(df_val['TotalCharges'].mean(), inplace=True)

clf = RandomForestClassifier(n_estimators=30 , oob_score = True, n_jobs = -1,random_state =50, max_features = "auto", min_samples_leaf = 50)
clf.fit(df_train[features], df_train["Churn"])

# Make predictions
predictions = clf.predict(df_val[features])
probs = clf.predict_proba(df_val[features])
display(predictions)
score = clf.score(df_val[features], df_val["Churn"])
print("Accuracy: ", score)
data['Churn'].value_counts()
get_ipython().magic('matplotlib inline')
confusion_matrix = pd.DataFrame(
    confusion_matrix(df_val["Churn"], predictions), 
    columns=["Predicted False", "Predicted True"], 
    index=["Actual False", "Actual True"]
)
display(confusion_matrix)

# Calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = roc_curve(df_val["Churn"], probs[:,1])
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
