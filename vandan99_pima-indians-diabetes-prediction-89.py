import numpy as np

import pandas as pd

from timeit import default_timer as timer

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve,auc

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

data.head()
print(data.info())
# Check if balanced or imbalanced class

print(data['Outcome'].value_counts())
# visualising

sns.countplot(x='Outcome',data=data)

ax=plt.gca()

for p in ax.patches:

    ax.annotate(p.get_height(),(p.get_x()+0.35,p.get_height()+5))

plt.tight_layout()
for i in data.columns[1:8]:

    print('Missing values in',i,':',len(data[data[i]==0]))
cols=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

for i in cols:

    data[i]=data[i].replace(to_replace=0,value=np.NaN)
data.head()
# Visualize glucose data distribution 

plt.figure()

sns.set(style='white')



sns.distplot(data[data['Outcome']==1]['Glucose'].dropna(),label='Diabetic',kde_kws={'linewidth': 2});

b = sns.distplot(data[data['Outcome']==0]['Glucose'].dropna(),label='Non-Diabetic',kde_kws={'linewidth': 2});

plt.legend();

b.set_xlabel('Glucose Levels');
# Filling Glucose values with median acc. to outcome

data.loc[(data['Outcome'] == 0 ) & (data['Glucose'].isnull()), 'Glucose'] = data[data['Outcome'] == 0 ]['Glucose'].mean()

data.loc[(data['Outcome'] == 1 ) & (data['Glucose'].isnull()), 'Glucose'] = data[data['Outcome'] == 1 ]['Glucose'].mean()
print(data['Glucose'].isnull().sum())
# Visualize Insulin data distribution 

plt.figure()



sns.distplot(data[data['Outcome']==1]['Insulin'].dropna(),label='Diabetic',kde_kws={'linewidth': 2});

b = sns.distplot(data[data['Outcome']==0]['Insulin'].dropna(),label='Non-Diabetic',kde_kws={'linewidth': 2});

plt.legend();

b.set_xlabel('Insulin Levels');
# Filling Insulin values with median acc. to outcome

data.loc[(data['Outcome'] == 0 ) & (data['Insulin'].isnull()), 'Insulin'] = data[data['Outcome'] == 0 ]['Insulin'].median()

data.loc[(data['Outcome'] == 1 ) & (data['Insulin'].isnull()), 'Insulin'] = data[data['Outcome'] == 1 ]['Insulin'].median()
print(data['Insulin'].isnull().sum())
# Visualize SkinThickness data distribution 

plt.figure()



sns.distplot(data[data['Outcome']==1]['SkinThickness'].dropna(),label='Diabetic',kde_kws={'linewidth': 2});

b = sns.distplot(data[data['Outcome']==0]['SkinThickness'].dropna(),label='Non-Diabetic',kde_kws={'linewidth': 2});

plt.legend();

b.set_xlabel('Skin Thickness');
# Filling SkinThickness values with median acc. to outcome

data.loc[(data['Outcome'] == 0 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = data[data['Outcome'] == 0 ]['SkinThickness'].median()

data.loc[(data['Outcome'] == 1 ) & (data['SkinThickness'].isnull()), 'SkinThickness'] = data[data['Outcome'] == 1 ]['SkinThickness'].median()
print(data['SkinThickness'].isnull().sum())
# Visualize Blood Pressure data distribution 

plt.figure()



sns.distplot(data[data['Outcome']==1]['BloodPressure'].dropna(),label='Diabetic',kde_kws={'linewidth': 2});

b = sns.distplot(data[data['Outcome']==0]['BloodPressure'].dropna(),label='Non-Diabetic',kde_kws={'linewidth': 2});

plt.legend();

b.set_xlabel('Blood Pressure levels');
# Filling BloodPressure values with median acc. to outcome

data.loc[(data['Outcome'] == 0 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = data[data['Outcome'] == 0 ]['BloodPressure'].mean()

data.loc[(data['Outcome'] == 1 ) & (data['BloodPressure'].isnull()), 'BloodPressure'] = data[data['Outcome'] == 1 ]['BloodPressure'].mean()
print(data['BloodPressure'].isnull().sum())
# Visualize SkinThickness data distribution 

plt.figure()



sns.distplot(data[data['Outcome']==1]['BMI'].dropna(),label='Diabetic',kde_kws={'linewidth': 2});

b = sns.distplot(data[data['Outcome']==0]['BMI'].dropna(),label='Non-Diabetic',kde_kws={'linewidth': 2});

plt.legend();

b.set_xlabel('BMI');
# Filling BloodPressure values with median acc. to outcome

data.loc[(data['Outcome'] == 0 ) & (data['BMI'].isnull()), 'BMI'] = data[data['Outcome'] == 0 ]['BMI'].median()

data.loc[(data['Outcome'] == 1 ) & (data['BMI'].isnull()), 'BMI'] = data[data['Outcome'] == 1 ]['BMI'].median()
print(data['BMI'].isnull().sum())
data.head()
plt.figure(figsize=(12,10)) 

p=sns.heatmap(data.corr(), annot=True,cmap ='YlGnBu')  
sns.pairplot(data, hue="Outcome", vars=['Pregnancies','Glucose','BloodPressure','SkinThickness','BMI']);
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

scaled = std.fit_transform(data.iloc[:,:8])

X = pd.DataFrame(scaled,columns=data.columns[:8])

y = data['Outcome'].astype(int)
display(X.head(), y.head())
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)



clf=GradientBoostingClassifier(max_depth=4,random_state=0);



clf.fit(X_train, y_train);
print('Training set Score :',clf.score(X_train,y_train))

print('Test set Score :',round(clf.score(X_test, y_test)*100,2))
def scores_table(model):

    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    res = []

    for sc in scores:

        scores = cross_val_score(model, X, y, cv = 5, scoring = sc)

        res.append(scores)

    df = pd.DataFrame(res).T

    df.loc['mean'] = df.mean()

    df.loc['std'] = df.std()

    df= df.rename(columns={0: 'accuracy', 1:'precision', 2:'recall',3:'f1',4:'roc_auc'})

    return df
scores_table(clf)
y_pred = clf.predict(X_test)

confusion_matrix(y_test, y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
def plot_ruc(model, X_train, y_train, X_test, y_test):

    y_pred = model.fit(X_train, y_train).decision_function(X_test)



    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    roc_auc_lr = auc(fpr, tpr)



    plt.figure()

    plt.plot(fpr,tpr, lw=3, label='DTC ROC curve (area = {:0.2f})'.format(roc_auc_lr));



    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.title('ROC Curve', fontsize=16)

    plt.legend(loc='lower right', fontsize=13)

    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--');



def plot_prc(model, X_train, y_train, X_test, y_test):

    y_pred = model.fit(X_train, y_train).decision_function(X_test)

    

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)



    plt.figure()

    plt.xlim([0.0, 1.01])

    plt.ylim([0.0, 1.01])

    plt.plot(precision, recall, label='Precision-Recall Curve',lw=3)

    plt.xlabel('Precision', fontsize=16)

    plt.ylabel('Recall', fontsize=16)
clf=GradientBoostingClassifier(max_depth=4,random_state=0)

plot_ruc(clf,X_train, y_train, X_test, y_test)

plot_prc(clf,X_train, y_train, X_test, y_test)
sns.barplot(y=X.columns,x=clf.feature_importances_,orient="h");
start = timer()

grid_values = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1],

               'max_depth': [2,3,4,5,6,7]}

clf=GradientBoostingClassifier(n_estimators=250,random_state=0)

grid_clf_auc = GridSearchCV(clf, param_grid = grid_values, scoring = 'recall',cv=5,n_jobs=-1)

grid_clf_auc.fit(X_train, y_train)

y_decision_fn_scores_auc = grid_clf_auc.decision_function(X_test) 



end = timer()



print('Test set AUC: ', roc_auc_score(y_test, y_decision_fn_scores_auc))

print('Grid best parameter: ', grid_clf_auc.best_params_)

print('Grid best score (recall): ', grid_clf_auc.best_score_)

print('Time taken: {0:.2f}',end-start,'seconds')

clf2=GradientBoostingClassifier(n_estimators=250,max_depth=2,learning_rate=0.01,random_state=0)

clf2.fit(X_train, y_train)

sc = clf2.score(X_test, y_test)

print('Test set score: {0:.2f} %'.format(sc*100))
# New confusion matrix

y_pred = clf2.predict(X_test)

confusion_matrix(y_test, y_pred)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
scores_table(clf2)
plot_ruc(clf2,X_train, y_train, X_test, y_test)

plot_prc(clf2,X_train, y_train, X_test, y_test)