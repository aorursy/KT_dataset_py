# Basics
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from collections import OrderedDict
import missingno as msno
from sklearn.preprocessing import StandardScaler

# Sampling
from sklearn.model_selection import train_test_split

# Classifiier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import accuracy_score

%matplotlib inline
sns.set(color_codes=True)
pal = sns.color_palette("Set2", 10)
sns.set_palette(pal)

TitanicTrain = pd.read_csv("../input/train.csv")
TitanicTrain.columns, TitanicTrain.shape 
TitanicTrain.head(3)
TitanicTrain.info()
msno.matrix(TitanicTrain)
categ =  [ 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
conti = ['Fare', 'Age']

#Distribution
fig = plt.figure(figsize=(30, 10))
for i in range (0,len(categ)):
    fig.add_subplot(3,3,i+1)
    sns.countplot(x=categ[i], data=TitanicTrain);  

for col in conti:
    fig.add_subplot(3,3,i + 2)
    sns.distplot(TitanicTrain[col].dropna());
    i += 1
    
plt.show()
fig.clear()
fig = plt.figure(figsize=(30, 10))
i = 1
for col in categ:
    if col != 'Survived':
        fig.add_subplot(3,3,i)
        sns.countplot(x=col, data=TitanicTrain,hue='Survived');
        i += 1

# Box plot survived x age
fig.add_subplot(3,3,6)
sns.swarmplot(x="Survived", y="Age", hue="Sex", data=TitanicTrain);
fig.add_subplot(3,3,7)
sns.boxplot(x="Survived", y="Age", data=TitanicTrain)

# fare and Survived
fig.add_subplot(3,3,8)
sns.violinplot(x="Survived", y="Fare", data=TitanicTrain)

# correlations with the new features
corr = TitanicTrain.drop(['PassengerId'], axis=1).corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig.add_subplot(3,3,9)
sns.heatmap(corr, mask=mask, cmap=cmap, cbar_kws={"shrink": .5})
plt.show()
fig.clear()
title = ['Mlle','Mrs', 'Mr', 'Miss','Master','Don','Rev','Dr','Mme','Ms','Major','Col','Capt','Countess']

def ExtractTitle(name):
    tit = 'missing'
    for item in title :
        if item in name:
            tit = item
    if tit == 'missing':
        tit = 'Mr'
    return tit

TitanicTrain["Title"] = TitanicTrain.apply(lambda row: ExtractTitle(row["Name"]),axis=1)
plt.figure(figsize=(13, 5))
fig.add_subplot(2,1,1)
sns.countplot(x='Title', data=TitanicTrain,hue='Survived');
# Age
MedianAge = TitanicTrain.Age.median()
TitanicTrain.Age = TitanicTrain.Age.fillna(value=MedianAge)
# Embarked replace NaN with the mode value
ModeEmbarked = TitanicTrain.Embarked.mode()[0]
TitanicTrain.Embarked = TitanicTrain.Embarked.fillna(value=ModeEmbarked)
# Fare have 1 NaN missing value on the Submission dataset
MedianFare = TitanicTrain.Fare.median()
TitanicTrain.Fare = TitanicTrain.Fare.fillna(value=MedianFare)
# Cabin
TitanicTrain["Cabin"] = TitanicTrain.apply(lambda obs: "No" if pd.isnull(obs['Cabin']) else "Yes", axis=1)
TitanicTrain = pd.get_dummies(TitanicTrain,drop_first=True,columns=['Sex','Title','Cabin','Embarked'])
scale = StandardScaler().fit(TitanicTrain[['Age', 'Fare']])
TitanicTrain[['Age', 'Fare']] = scale.transform(TitanicTrain[['Age', 'Fare']])
Target = TitanicTrain.Survived
Features = TitanicTrain.drop(['Survived','Name','Ticket','PassengerId'],axis=1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size = 0.3, random_state=42)
Target = TitanicTrain.Survived

MlRes= {}
def MlResult(model,score):
    MlRes[model] = score
    print(MlRes)
    
roc_curve_data = {}
def ConcatRocData(algoname, fpr, tpr, auc):
    data = [fpr, tpr, auc]
    roc_curve_data[algoname] = data    
# Logistic Regression : 
logi_reg = LogisticRegression()
# Fit the regressor to the training data
logi_reg.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = logi_reg.predict(X_test)

# Score / Metrics
accuracy = logi_reg.score(X_test, y_test)
MlResult('Logistic Regression',accuracy)
# Create a random Forest Classifier instance
rfc = RandomForestClassifier(n_estimators = 100)
# Fit to the training data
rfc.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = rfc.predict(X_test)

# Score / Metrics
accuracy = rfc.score(X_test, y_test) # = accuracy
MlResult('Random Forest',accuracy)
#Features importance
def FeaturesImportance(data,model):
    features = data.columns.tolist()
    fi = model.feature_importances_
    sorted_features = {}
    for feature, imp in zip(features, fi):
        sorted_features[feature] = round(imp,3)

    # sort the dictionnary by value
    sorted_features = OrderedDict(sorted(sorted_features.items(),reverse=True, key=lambda t: t[1]))

    #for feature, imp in sorted_features.items():
        #print(feature+" : ",imp)

    dfvi = pd.DataFrame(list(sorted_features.items()), columns=['Features', 'Importance'])
    #dfvi.head()
    plt.figure(figsize=(15, 5))
    sns.barplot(x='Features', y='Importance', data=dfvi);
    plt.xticks(rotation=90) 
    plt.show()
    
#Features importance
FeaturesImportance(TitanicTrain,rfc)
svm = SVC(probability=True)
# Fit to the training data
svm.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = svm.predict(X_test)

# Score / Metrics
accuracy = svm.score(X_test, y_test) # = accuracy
MlResult('SVC',accuracy)
# print score for each model
#for algo,score in MlRes.items():
#    print(algo+" : {}".format(round(score,3)))
res = pd.DataFrame(list(MlRes.items()), columns=['Model', 'Score']).sort_values("Score", ascending=False)
print(res)
plt.figure(figsize=(8, 5))
sns.barplot(x='Score', y='Model', data=res);
plt.xticks(rotation=90) 
plt.xlim([0.6, 0.9])
plt.show()