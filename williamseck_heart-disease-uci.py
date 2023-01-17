import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')

from sklearn.model_selection import cross_val_score

from statistics import mean

from lightgbm import LGBMClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('../input/heart.csv')
df.head()
df.info()
df.describe()
df.shape
df.columns
pd.isnull(df).sum()
df.hist(bins=10,figsize=(15,15),color= "green", grid=False)

plt.show()
# Matrice couleur des données

def plot_correlation_map( df ):

    corr = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', #ou tout simplement: df.corr()

       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']].corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )

plot_correlation_map(df)
corr = df.corr()['target'].abs().sort_values()

corr
def plot_cat(data, x_axis, y_axis, hue):

    plt.figure()    

    sns.barplot(x=x_axis, y=y_axis, hue=hue, data=data)

    sns.set_context("notebook", font_scale=1.6)

    plt.legend(loc="upper right", fontsize="medium")

plot_cat(df,"sex", "target", None) 
plot_cat(df,"fbs", "target", "sex") 
X= df.drop('target',axis=1)  # on peut aussi donner les valeurs de x et de y

y=df['target']
# MinMaxScaler

X['trestbps'] = MinMaxScaler().fit_transform(X['trestbps'].values.reshape(-1, 1))

X['chol'] = MinMaxScaler().fit_transform(X['chol'].values.reshape(-1, 1))

X['thalach'] = MinMaxScaler().fit_transform(X['thalach'].values.reshape(-1, 1))

X['age'] = MinMaxScaler().fit_transform(X['age'].values.reshape(-1, 1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LGBMClassifier(n_estimators=200, max_depth=5)
clf.fit(X_train,y_train)
scores = cross_val_score(clf,X,y,scoring='roc_auc', cv=5)

print('AUC')

print(np.mean(scores))

print(np.std(scores))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

params = {'n_neighbors':[i for i in range(1,33,2)]}
model = GridSearchCV(knn,params,cv=10)
model.fit(X_train,y_train)

model.best_params_           #print's parameters best values
predict = model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

print('Accuracy Score: ',accuracy_score(y_test,predict))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,predict),5)*100,'%')
cnf_matrix = confusion_matrix(y_test,predict)

cnf_matrix
class_names = [0,1]

fig,ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',

           fmt = 'g')

ax.xaxis.set_label_position('top')

plt.tight_layout()

plt.title('Confusion matrix', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
coef = pd.Series(clf.feature_importances_, index = X.columns)

imp_coef = coef.sort_values(ascending=False)

print(imp_coef)
plt.title("Feature Importance in Ensemble model")

imp_coef.plot(kind="barh")

plt.show()