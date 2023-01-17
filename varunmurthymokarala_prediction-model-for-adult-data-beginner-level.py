import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
train_data = pd.read_csv('../input/adult-incomes-in-the-united-states/adult.data', names=columns, sep=' *, *', na_values='?')
test_data  = pd.read_csv('../input/adult-incomes-in-the-united-states/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')
train_data.info()
test_data.info()
num_attributes = train_data.select_dtypes(include=['integer'])
print(num_attributes.columns)
num_attributes.hist(figsize=(10,10))
cat_attributes = train_data.select_dtypes(include=['object'])
print(cat_attributes.columns)
sns.countplot(y='workClass', hue='income', data = cat_attributes)
sns.countplot(y='occupation', hue='income', data = cat_attributes)
X=train_data.select_dtypes(include=['integer'])
X.head()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
y=train_data.iloc[:,-1]
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='linear',gamma='auto', C=10)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
train_data = pd.read_csv('../input/adult-incomes-in-the-united-states/adult.data', names=columns, sep=' *, *', na_values='?')
test_data  = pd.read_csv('../input/adult-incomes-in-the-united-states/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')
train_data.info()
test_data.info()
num_attributes = train_data.select_dtypes(include=['integer'])
num_attributes.hist(figsize=(10,10))
cat_attributes = train_data.select_dtypes(include=['object'])
sns.countplot(y='workClass', hue='income', data = cat_attributes)
sns.countplot(y='occupation', hue='income', data = cat_attributes)
X=train_data.select_dtypes(include=['integer'])
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
y=train_data.iloc[:,-1]
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model=GaussianNB().fit(X_train,y_train)
y_pred=model.predict(X_test)
print('The Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(' The Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
