import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style("darkgrid")
dataRed = pd.read_csv('../input/winequality-red.csv')
dataRed.head()
dataRed.describe()
wineDict = {3:"bad",4:"bad",5:"passable",6:"passable",7:"good",8:"good"}
dataRed["wineQuality"] = dataRed.quality.map(wineDict)
dataRed.head()
sns.pairplot(dataRed,hue="wineQuality")
plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
sns.countplot(data=dataRed,x="quality")

plt.subplot(1,2,2)
sns.countplot(data=dataRed,x="wineQuality")
plt.tight_layout()
plt.figure(figsize=(12,12))
sns.heatmap(data=dataRed.corr(),annot=True,cmap="Blues",linewidths=1)
plt.figure(figsize = (16,20))
nrows = 4
ncols = 2

plt.subplot(nrows,ncols,1)
sns.violinplot(data=dataRed,x="quality",y="alcohol")
plt.subplot(nrows,ncols,2)
sns.violinplot(data=dataRed,x="wineQuality",y="alcohol")

plt.subplot(nrows,ncols,3)
sns.violinplot(data=dataRed,x="quality",y="volatile acidity")
plt.subplot(nrows,ncols,4)
sns.violinplot(data=dataRed,x="wineQuality",y="volatile acidity")

plt.subplot(nrows,ncols,5)
sns.violinplot(data=dataRed,x="quality",y="sulphates")
plt.subplot(nrows,ncols,6)
sns.violinplot(data=dataRed,x="wineQuality",y="sulphates")

plt.subplot(nrows,ncols,7)
sns.violinplot(data=dataRed,x="quality",y="citric acid",split=True)
plt.subplot(nrows,ncols,8)
sns.violinplot(data=dataRed,x="wineQuality",y="citric acid",split=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataRed.drop(["quality","wineQuality"],axis=1))
scaled_features = scaler.transform(dataRed.drop(["quality","wineQuality"],axis=1))
df_feat = pd.DataFrame(scaled_features,columns=dataRed.columns[:-2])
df_feat.head()
X = scaled_features
y = dataRed["wineQuality"]
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print(classification_report(y_test,knn_pred))
print(confusion_matrix(y_test,knn_pred))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree_pred = dtree.predict(X_test)
print(classification_report(y_test,dtree_pred))
print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
svc_pred = svc_model.predict(X_test)
print(classification_report(y_test,svc_pred))
print(confusion_matrix(y_test,svc_pred))
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))
features = list(dataRed.columns)
features
importance = RandomForestClassifier(random_state=0,n_jobs=-1)
imp_model = importance.fit(X,y)
model_importances = model.feature_importances_
indices = np.argsort(model_importances)[::-1]
names = [features[i] for i in indices]
plt.figure(figsize=(10,6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]),model_importances[indices])
plt.xticks(range(X.shape[1]), names,rotation=60)
plt.show()
