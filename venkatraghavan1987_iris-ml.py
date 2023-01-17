import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc
iris = pd.read_csv('../input/iris/Iris.csv')
df = iris.copy()
df.head()
df.drop(columns = 'Id', inplace = True)
df.shape
df['Species'].value_counts()
df.isnull().any()
df.dtypes
df.describe()
df[df['Species'] == 'Iris-setosa'].describe()
df[df['Species'] == 'Iris-versicolor'].describe()
df[df['Species'] == 'Iris-virginica'].describe()
df[(df['PetalWidthCm'] >= 0) & (df['PetalWidthCm'] < 1)]
sns.pairplot(df, hue = 'Species')

plt.show()
col_list = df.columns.tolist()

col_list
X = df[col_list[:-1]].values

Y = df[col_list[-1]].values
X_train_dt, X_test_dt, Y_train_dt, Y_test_dt = train_test_split(X, Y, train_size = 0.70, random_state = 1, stratify = Y)



print('X_train dimension:', X_train_dt.shape)

print('X_test dimension:', X_test_dt.shape)

print('Y_train dimension:', Y_train_dt.shape)

print('Y_test dimension:', Y_test_dt.shape)
clf_decision_tree = tree.DecisionTreeClassifier(random_state = 1)
cross_validation_decision_tree_split_strategy = StratifiedKFold(n_splits = 3)
cv_scores_decision_tree = cross_val_score(clf_decision_tree, X_train_dt, Y_train_dt, 

                                          cv = cross_validation_decision_tree_split_strategy)



print('Average CV score:', round(np.mean(cv_scores_decision_tree),4))



print('Standard deviation of the CV scores:', round(np.std(cv_scores_decision_tree),4))
parameter_grid_decision_tree = {'criterion' : ['gini', 'entropy'],

                 'splitter' : ['best', 'random'],

                 'max_depth' : [1,2,3,4,5],

                 'max_features' : [1,2,3,4],

                 'min_samples_split' : [2,3,4,5],

                 'min_samples_leaf' : [1,2,3,4,5],

                 }



grid_search_decision_tree = GridSearchCV(clf_decision_tree, param_grid = parameter_grid_decision_tree, 

                                         cv = cross_validation_decision_tree_split_strategy)



grid_search_decision_tree.fit(X_train_dt, Y_train_dt)



print('Best parameters:', grid_search_decision_tree.best_params_)
best_params_clf_decision_tree = tree.DecisionTreeClassifier(**grid_search_decision_tree.best_params_, random_state = 1)



cv_scores_best_params_decision_tree = cross_val_score(best_params_clf_decision_tree, X_train_dt, Y_train_dt, 

                                                       cv = cross_validation_decision_tree_split_strategy)



print("Best parameters Cross Validation score:", round(np.mean(cv_scores_best_params_decision_tree),4))



print('Standard deviation for Best param used CV:', round(np.std(cv_scores_best_params_decision_tree),4))
final_clf_decision_tree = tree.DecisionTreeClassifier(**grid_search_decision_tree.best_params_, random_state = 1)



final_training_decision_tree = final_clf_decision_tree.fit(X_train_dt, Y_train_dt)
Y_pred_decision_tree = final_training_decision_tree.predict(X_test_dt)



Y_probs_decision_tree = final_training_decision_tree.predict_proba(X_test_dt)
print('Test score using Decision Tree:', round(final_training_decision_tree.score(X_test_dt, Y_test_dt),4))
confusion_matrix(Y_test_dt, Y_pred_decision_tree)
print(classification_report(Y_test_dt, Y_pred_decision_tree))
y_test_dummies = pd.get_dummies(Y_test_dt, drop_first=False).values



fpr = dict()



tpr = dict()



roc_auc = dict()



n_classes = y_test_dummies.shape[1]



for i in range(n_classes):

    

    fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], Y_probs_decision_tree[:, i])

    

    roc_auc[i] = auc(fpr[i], tpr[i])

    

colors = ['blue', 'red', 'green']



for i, color in zip(range(n_classes), colors):

    

    plt.plot(fpr[i], tpr[i], color=color,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))

    

    

plt.plot([0, 1], [0, 1], 'k--')



plt.xlim([-0.05, 1.0])



plt.ylim([0.0, 1.05])



plt.xlabel('False Positive Rate')



plt.ylabel('True Positive Rate')



plt.title('ROC curve for multi-class data using Decision Tree')



plt.legend(loc="lower right")



plt.show()  
X_train_log, X_test_log, Y_train_log, Y_test_log = train_test_split(X, Y, train_size = 0.70, random_state = 1, stratify = Y)



print('X_train dimension:', X_train_log.shape)

print('X_test dimension:', X_test_log.shape)

print('Y_train dimension:', Y_train_log.shape)

print('Y_test dimension:', Y_test_log.shape)
clf_logit = LogisticRegression(multi_class = 'multinomial', max_iter = 1000, random_state = 1)
cross_validation_logit_split_strategy = StratifiedKFold(n_splits = 3)
cv_scores_logit = cross_val_score(clf_logit, X_train_log, Y_train_log, cv = cross_validation_logit_split_strategy)



print('Average CV score:', round(np.mean(cv_scores_logit),4))



print('Standard deviation of the CV scores:', round(np.std(cv_scores_logit),4))
param_grid_logit = {

    'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_search_logit = GridSearchCV(clf_logit, param_grid = param_grid_logit, cv = cross_validation_logit_split_strategy)



grid_search_logit.fit(X_train_log, Y_train_log)



print('Best parameters:', grid_search_logit.best_params_)



print('Best score:', round(grid_search_logit.best_score_,4))
best_params_clf_logit = LogisticRegression(**grid_search_logit.best_params_, random_state = 1, max_iter = 500)



cv_scores_best_params_logit = cross_val_score(best_params_clf_logit, X_train_log, Y_train_log, 

                                                       cv = cross_validation_logit_split_strategy)



print("Best parameters Cross Validation score:", round(np.mean(cv_scores_best_params_logit),4))



print('Standard deviation for Best param used CV:', round(np.std(cv_scores_best_params_logit),4))
final_clf_logit = LogisticRegression(multi_class = 'multinomial', 

                                     **grid_search_logit.best_params_, max_iter = 500, random_state = 1)



final_training_logit = final_clf_logit.fit(X_train_log, Y_train_log)
Y_pred_logit = final_training_logit.predict(X_test_log)



Y_probs_logit = final_training_logit.predict_proba(X_test_log)
print('Test score using Logistic regression:', round(final_training_logit.score(X_test_log, Y_test_log),4))
confusion_matrix(Y_test_log, Y_pred_logit)
print(classification_report(Y_test_log, Y_pred_logit))
y_test_dummies_logit = pd.get_dummies(Y_test_log, drop_first=False).values



fpr_logit = dict()



tpr_logit = dict()



roc_auc_logit = dict()



n_classes_logit = y_test_dummies_logit.shape[1]



for i in range(n_classes_logit):

    

    fpr_logit[i], tpr_logit[i], _ = roc_curve(y_test_dummies_logit[:, i], Y_probs_logit[:, i])

    

    roc_auc_logit[i] = auc(fpr_logit[i], tpr_logit[i])

    

colors_logit = ['blue', 'red', 'green']



for i, color_logit in zip(range(n_classes_logit), colors_logit):

    

    plt.plot(fpr_logit[i], tpr_logit[i], color=color_logit,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc_logit[i]))

    

    

plt.plot([0, 1], [0, 1], 'k--')



plt.xlim([-0.05, 1.0])



plt.ylim([0.0, 1.05])



plt.xlabel('False Positive Rate')



plt.ylabel('True Positive Rate')



plt.title('ROC curve for multi-class data using Logistic Regression')



plt.legend(loc="lower right")



plt.show()  
X_train_knn, X_test_knn, Y_train_knn, Y_test_knn = train_test_split(X, Y, train_size = 0.70, random_state = 1, stratify = Y)



print('X_train dimension:', X_train_knn.shape)

print('X_test dimension:', X_test_knn.shape)

print('Y_train dimension:', Y_train_knn.shape)

print('Y_test dimension:', Y_test_knn.shape)
cv_scores = []



neighbors = list(np.arange(3,30,2))



for n in neighbors:

    

    knn = KNeighborsClassifier(n_neighbors = n)

    

    cross_val = cross_val_score(knn, X_train_knn, Y_train_knn, cv = 3, scoring = 'accuracy')

    

    cv_scores.append(np.mean(cross_val))

    

print('Average Cross Validation scores:', cv_scores)
error = [1-x for x in cv_scores]



optimal_n = neighbors[error.index(min(error))]



print('Best K is:', optimal_n)



knn_optimal = KNeighborsClassifier(n_neighbors=optimal_n)



cross_val_best_K = cross_val_score(knn_optimal, X_train_knn, Y_train_knn, cv = 3, scoring = 'accuracy')



cv_score_best_k = np.mean(cross_val_best_K)



print('Cross Validation score using best K: {0:.4f}'.format(cv_score_best_k))



print('Standard deviation of CV score using best K: {0:.4f}'.format(np.std(cross_val_best_K)))
knn_final_model = KNeighborsClassifier(n_neighbors = optimal_n)



knn_final_model.fit(X_train_knn, Y_train_knn)



Y_pred_KNN = knn_final_model.predict(X_test_knn)



Y_probs_KNN = knn_final_model.predict_proba(X_test_knn)



acc_KNN = accuracy_score(Y_test_knn, Y_pred_KNN)



print("The optimal k = {0}, in which accuracy is {1:.4f}".format(optimal_n,acc_KNN))
confusion_matrix(Y_test_knn, Y_pred_KNN)
print(classification_report(Y_test_knn, Y_pred_KNN))
y_test_dummies_KNN = pd.get_dummies(Y_test_knn, drop_first=False).values



fpr_KNN = dict()



tpr_KNN = dict()



roc_auc_KNN = dict()



n_classes_KNN = y_test_dummies_KNN.shape[1]



for i in range(n_classes_KNN):

    

    fpr_KNN[i], tpr_KNN[i], _ = roc_curve(y_test_dummies_KNN[:, i], Y_probs_KNN[:, i])

    

    roc_auc_KNN[i] = auc(fpr_KNN[i], tpr_KNN[i])

    

colors_KNN = ['blue', 'red', 'green']



for i, color_KNN in zip(range(n_classes_KNN), colors_KNN):

    

    plt.plot(fpr_KNN[i], tpr_KNN[i], color=color_KNN,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc_KNN[i]))

    

    

plt.plot([0, 1], [0, 1], 'k--')



plt.xlim([-0.05, 1.0])



plt.ylim([0.0, 1.05])



plt.xlabel('False Positive Rate')



plt.ylabel('True Positive Rate')



plt.title('ROC curve for multi-class data using KNN')



plt.legend(loc="lower right")



plt.show()  