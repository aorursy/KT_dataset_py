import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/winequality-red.csv")
df.head(10)
quality = df["quality"].values
category = []
for num in quality:
    if num<5:
        category.append("Bad")
    elif num>6:
        category.append("Good")
    else:
        category.append("Medium")
    
alcohol = df["alcohol"].values
bracket = []
for i in alcohol:
    if i < 10.5:
        bracket.append("Considerable")
    else:
        bracket.append("Excessive")
#Create new data
bracket = pd.DataFrame(data=bracket, columns=["bracket"])
category = pd.DataFrame(data=category, columns=["category"])
data = pd.concat([df,bracket,category],axis=1)
data.drop(["alcohol","quality"],axis=1,inplace=True)          
data.head(10)
plt.figure(figsize=(10,6))
sns.countplot(data["category"],palette="pastel")
data["category"].value_counts()
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True, cmap = "YlOrRd")
plt.figure(figsize=(12,6))
sns.barplot(x=df["quality"],y=df["alcohol"],palette="colorblind")
plt.figure(figsize=(12,6))
sns.jointplot(y=df["density"],x=df["alcohol"],kind="scatter")
facet = sns.FacetGrid(data, col = "bracket", hue = "category", height = 4, palette = "deep")
facet.map(plt.scatter, 'fixed acidity', 'volatile acidity')
plt.legend()
data.drop(["bracket"],axis=1,inplace=True)    
data.head(10)
#Make bins
X= data.iloc[:,:-1].values
y=data.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
labelencoder_y =LabelEncoder()
y= labelencoder_y.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
pred_svc =svc.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(y_test,pred_svc))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
pred_knn=knn.predict(X_test)
print(classification_report(y_test, pred_knn))
from sklearn.model_selection import train_test_split
X_train30, X_test30, y_train30, y_test30 = train_test_split(X, y, test_size = 0.3,random_state=0)
sc_X = StandardScaler()
X_train30 = sc_X.fit_transform(X_train30)
X_test30 = sc_X.transform(X_test30)
#Applying SVC

svc = SVC()
svc.fit(X_train30,y_train30)
pred_svc30 =svc.predict(X_test30)

print(classification_report(y_test30,pred_svc30))
#Applying Random Forest

rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train30, y_train30)
pred_rfc30 = rfc.predict(X_test30)
print(classification_report(y_test30, pred_rfc30))
#Applying K-Nearest Neighbor

knn = KNeighborsClassifier()
knn.fit(X_train30,y_train30)
pred_knn30=knn.predict(X_test30)
print(classification_report(y_test30, pred_knn30))
results = pd.DataFrame({'models': ["SVC","Random Forest","KNN"],
                           'accuracies20': [accuracy_score(y_test,pred_svc),accuracy_score(y_test,pred_rfc),accuracy_score(y_test,pred_knn)],
                          'accuracies30': [accuracy_score(y_test30,pred_svc30),accuracy_score(y_test30,pred_rfc30),accuracy_score(y_test30,pred_knn30)]})
results