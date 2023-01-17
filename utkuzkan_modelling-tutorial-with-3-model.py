# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Load libraries
import numpy as np
import pandas as pd # pandas is so commonly used, it's shortened to pd
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns # seaborn gets shortened to sns
import matplotlib.pyplot as plt
# We want our plots to appear in the notebook
%matplotlib inline 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv") # 'DataFrame' shortened to 'df'
df.shape # (rows, columns)
# Everything except target variable
X = df.drop("target", axis=1)

# Target variable
y = df.target.values
# Independent variables (no target column)
X.head()
seed = 12345
test_size = 0.20

# Random seed for reproducibility
np.random.seed(seed)

# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X, # independent variables 
                                                    y, # dependent variable
                                                    test_size =test_size) # percentage of data to use for test set
# Test options and evaluation metric
num_folds = 10 # permanently 
scoring = 'accuracy'
# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier()}
# Create function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(seed)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores
model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test);


model_scores
model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();
# Tune scaled KNN
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler', StandardScaler()), ('knn',KNeighborsClassifier())])
 
param_grid = {
    'knn__n_neighbors': np.arange(1,30)
    }
 
gs_knn = GridSearchCV(pipe,return_train_score=True, param_grid=param_grid, cv=num_folds,scoring=scoring, n_jobs=-1, verbose=2)
gs_knn.fit(X_train, y_train)
 
print('Best Parameter:', gs_knn.best_params_)
print('Best Score:', gs_knn.best_score_)

score_cols = ['mean_train_score', 'std_train_score',
              'mean_test_score', 'std_test_score']

knn_result = pd.DataFrame(gs_knn.cv_results_).head()


means = gs_knn.cv_results_['mean_test_score']
stds = gs_knn.cv_results_['std_test_score']
params = gs_knn.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

grid_df = pd.DataFrame(gs_knn.cv_results_,
                       columns=['param_knn__n_neighbors',
                                'mean_train_score',
                                'mean_test_score',
                                'std_test_score'])

grid_df.set_index('param_knn__n_neighbors', inplace=True)
%matplotlib inline
plt.rcParams["figure.figsize"] = [20, 5]

ax = grid_df.plot.line(marker='.')
ax.set_xticks(grid_df.index);
print("Best: %f using %s" % (gs_knn.best_score_, gs_knn.best_params_))
# Evaluate the model
gs_knn.score(X_test, y_test)
# Make preidctions on test data with GridSearchCV
y_gs_preds = gs_knn.predict(X_test)
knn = KNeighborsClassifier(n_neighbors= 9)
knn.fit(X_train,y_train)
y_preds = knn.predict(X_test)
sns.set(font_scale=1.5) # Increase font size

def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    
def plot_roc_curve_custom(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_metrics(accuracy,precise,recall,f1,title):
    # Visualizing cross-validated metrics
    metrics = pd.DataFrame({"Accuracy": accuracy,
                            "Precision": precise,
                            "Recall": recall,
                            "F1": f1},
                          index=[0])
    metrics.T.plot.bar(title=title, legend=False);
conf_matrix=confusion_matrix(y_gs_preds,y_test)

df_cm = pd.DataFrame(conf_matrix, range(2),range(2))
sns.set(font_scale=1.4)
pltt = sns.heatmap(df_cm, annot=True,annot_kws={"size": 12}, cmap="YlGnBu",  fmt='g')
#without gridSearch result
conf_matrix=confusion_matrix(y_preds,y_test)

df_cm = pd.DataFrame(conf_matrix, range(2),range(2))
sns.set(font_scale=1.4)
pltt = sns.heatmap(df_cm, annot=True,annot_kws={"size": 12}, cmap="YlGnBu",  fmt='g')
from sklearn.metrics import plot_roc_curve
# Plot ROC curve and calculate AUC metric
plt.rcParams["figure.figsize"] = [5, 5]
plot_roc_curve(gs_knn, X_test, y_test);
from sklearn.metrics import plot_roc_curve
# Plot ROC curve and calculate AUC metric
plot_roc_curve(knn, X_test, y_test);
# Show classification report with GridSearchCV
print(classification_report(y_test, y_gs_preds))
# Show classification report
print(classification_report(y_test, y_preds))
# Instantiate best model with best hyperparameters (found with GridSearchCV)
cv_knn = KNeighborsClassifier(n_neighbors= 9)


pipe = Pipeline([
    ('sc', StandardScaler()),
    ('lr', KNeighborsClassifier(n_neighbors= 9))
])


# Cross-validated accuracy score
cv_acc = cross_val_score(pipe,
                         X,
                         y,
                         cv=num_folds, # 5-fold cross-validation
                         scoring="accuracy") # accuracy as scoring
np.mean(cv_acc)

pltt = sns.distplot(pd.Series(cv_acc,name='CV scores distribution'), color='r')
# Cross-validated precision score
cv_precision = cross_val_score(pipe,
                                       X,
                                       y,
                                       cv=num_folds, # 5-fold cross-validation
                                       scoring="precision") # precision as scoring
np.mean(cv_precision)
pltt = sns.distplot(pd.Series(cv_precision,name='CV Precision distribution'), color='r')

# Cross-validated recall score
cv_recall = cross_val_score(pipe,
                                    X,
                                    y,
                                    cv=num_folds, # 10-fold cross-validation
                                    scoring="recall") # recall as scoring
np.mean(cv_recall)
pltt = sns.distplot(pd.Series(cv_recall,name='CV Recall distribution'), color='r')
# Cross-validated F1 score
cv_f1 = cross_val_score(pipe,
                                X,
                                y,
                                cv=num_folds, # 5-fold cross-validation
                                scoring="f1") # f1 as scoring
np.mean(cv_f1)
pltt = sns.distplot(pd.Series(cv_f1,name='CV F1 distribution'), color='r')
# Cross-validated F1 score
roc_auc = cross_val_score(pipe,
                                X,
                                y,
                                cv=num_folds, # 5-fold cross-validation
                                scoring="roc_auc") # f1 as scoring


print("Classification Error=%0.5f" % np.mean(roc_auc))
pltt = sns.distplot(pd.Series(roc_auc,name='CV ROC_AUC distribution'), color='r')
cv_metrics
# Visualizing cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                            "Precision": cv_precision,
                            "Recall": cv_recall,
                            "F1": cv_f1},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);
cv_y_scores_knn = cross_val_predict(pipe, X, y, cv=num_folds,method="predict_proba")#[:,1]
cv_y_pred = cross_val_predict(pipe, X, y, cv=num_folds)
# Generate fpr and tpr for KNN classifers using roc_curve function
fpr_knn, tpr_knn, threshold_knn = roc_curve(y, cv_y_pred)
# Plot each ROC Curves
plot_roc_curve_custom(fpr_knn, tpr_knn,"CV_KNN=%0.5f" % roc_auc_score(y,cv_y_scores_knn[:,1]))
conf_mat = confusion_matrix(y, cv_y_pred)

df_cm = pd.DataFrame(conf_mat, range(2),range(2))
sns.set(font_scale=1.4)
pltt = sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, cmap="YlGnBu",  fmt='g')
# Show classification report
print(classification_report(y, cv_y_pred))
conf_mat
TP = conf_mat[1,1]
TN = conf_mat[0,0]
FP = conf_mat[0,1]
FN = conf_mat[1,0]
cross_val_Accuracy =(TP+TN) / float(TP+TN+FP+FN)
print("Accuracy=%0.2f" % cross_val_Accuracy)
cle = 1- cross_val_Accuracy
print("Classification Error=%0.2f" % cle)
cross_val_recall = TP / float(TP+FN)
print("Recall=%0.2f" % cross_val_recall)
TN / float(TN+FP)
cross_val_FPR = FP / float(TN+FP)
cross_val_FPR
cross_val_precise = TP / float(TP + FP)
cross_val_precise
cross_val_F1 = 2* ((cross_val_precise*cross_val_recall) / (cross_val_precise+cross_val_recall))
cross_val_F1
plot_metrics(cross_val_Accuracy,cross_val_precise,cross_val_recall,cross_val_F1,"Cross-Validated Metrics")