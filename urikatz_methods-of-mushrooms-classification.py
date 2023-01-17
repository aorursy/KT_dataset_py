import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/mushrooms.csv')
df.head()
df.info()
df.shape
print('Class in %')
df['class'].value_counts()/df['class'].count()*100
df_onehot = pd.get_dummies(df);
df_onehot = df_onehot.drop(['class_e'],axis=1) # Now , class_p is an indicator of 1=poisonous , 0 = edible
df_onehot.head()
corr = df_onehot.corr().loc[:,'class_p']
top_10_corr =corr.abs().sort_values(ascending=False).head(n=11).iloc[1:]
top_10_corr
highcorr = pd.DataFrame()
for var in top_10_corr.index:
    highcorr[var] = 100*df_onehot[['class_p',var]].groupby([var]).sum()/3916

highcorr
from sklearn.model_selection import train_test_split
X = df_onehot.drop(['class_p'],axis=1)
y = df_onehot.class_p
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.85,random_state = 42)

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier , export_graphviz
import graphviz
treeclf = DecisionTreeClassifier(random_state=42)
treeclf.fit(X_train,y_train)
mean_cv_score =cross_val_score(treeclf, X_train, y_train, cv=10,scoring='accuracy').mean()
print('Decision Tree Classifier mean 10 cv Accuracy score:{0:.3}'.format(mean_cv_score))
from sklearn.metrics import confusion_matrix
y_pred_tree = treeclf.predict(X_test)
conf = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(5,5))
sns.heatmap(conf, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Greens')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix', size = 15)
plt.show()
dot_data = export_graphviz(treeclf, out_file=None,feature_names=X.columns,filled=True, rounded=True,special_characters=True)  
graphviz.Source(dot_data)  

tree_features = pd.Series(treeclf.feature_importances_).sort_values(ascending=False).where(lambda x:x>0).dropna()
plt.figure(figsize=(8,5))
tree_features.plot.bar(color='g',align='center')
plt.xticks(range(len(tree_features.index)),X.columns.values[tree_features.index])
plt.title('Feature importances')
plt.ylabel('Importance')
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
logreg = LogisticRegression()
logreg.fit(X_train,y_train);
c_space = np.logspace(-5, 8, 50) #log space for the C parameter
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}
logreg_cv = GridSearchCV(logreg,param_grid,cv=10)
logreg_cv.fit(X_train,y_train);
print("Tuned Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Accuracy: {}".format(logreg_cv.best_score_))
y_pred_logreg = logreg_cv.predict(X_test)
conf = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(5,5))
sns.heatmap(conf, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Greens')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix for Log regression', size = 15)
plt.show()
from sklearn.naive_bayes import BernoulliNB
bnb_clf = BernoulliNB()
bnb_clf.fit(X_train, y_train)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
y_pred_bnb = bnb_clf.predict(X_test)
conf = confusion_matrix(y_test, y_pred_bnb)
plt.figure(figsize=(5,5))
sns.heatmap(conf, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Greens')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix for Naive Bays', size = 15)
plt.show()