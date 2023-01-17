# Pandas for reading the data:
import pandas as pd

# Libraries for plotting of the data:
import seaborn as sns
import matplotlib.pyplot as plt

# Libraries for Preprocessing of the data:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
import category_encoders as ce

# Libraries for Predicting on the data:
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Metrics to measure how good we do with predicting on the data:
from sklearn.metrics import *
data = pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")
data.head(7)
survived = data['Survived']

# Dropping the labels.
# Also dropping the PassengerId, Firstname and Lastname columns because they seem useless:
data.drop(['PassengerId', 'Firstname', 'Lastname', 'Survived'], axis=1, inplace=True)
data.head()
plt.figure(figsize=(7,7))
sns.set_context("poster", font_scale=0.7)
sns.set_palette("Reds")
sns.countplot(survived)
sns.set_palette(['skyblue', 'pink'])
plt.figure(figsize=(7,7))
sns.countplot(data['Sex'])
plt.figure(figsize=(20,7))
sns.set_context("poster", font_scale=0.6)
sns.countplot(data['Country'])
plt.figure(figsize=(7,7))
sns.violinplot(data=data, x='Sex',y='Age')
data.isnull().sum()
c = (data.dtypes == 'object')

categorical = list(c[c].index)
cat = ce.CatBoostEncoder()

# Fitting the data to the labels:
cat.fit(data[categorical], survived)

# Transforming the columns:
data[categorical] = cat.transform(data[categorical])
scale = StandardScaler()
scaleddata = pd.DataFrame(scale.fit_transform(data), columns=data.columns)
train, test, ytrain, ytest = train_test_split(scaleddata, y, train_size=0.7, test_size=0.3)
ran = RandomForestClassifier(n_estimators=500)

ran.fit(train, ytrain)

ranpred = ran.predict(test)

print("The Accuracy of this model is :", accuracy_score(ranpred, ytest)*100)
xgb = XGBClassifier(n_estimators=300)

xgb.fit(train, ytrain)

xpred = xgb.predict(test)
print("The Accuracy of this model is :", accuracy_score(xpred, ytest)*100)
tree = DecisionTreeClassifier()

tree.fit(train, ytrain)

treepred = tree.predict(test)
print("The Accuracy of this model is :", accuracy_score(treepred, ytest)*100)