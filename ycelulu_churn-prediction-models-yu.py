# Importing libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import seaborn as sns #visualization
import matplotlib.pyplot as plt #visualization
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import (plot_confusion_matrix, confusion_matrix, 
                             accuracy_score, mean_squared_error, r2_score, 
                             roc_auc_score, roc_curve, classification_report, 
                             precision_recall_curve, auc, f1_score, 
                             average_precision_score, precision_score, recall_score)
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import scale, StandardScaler, RobustScaler, MinMaxScaler


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

%config InlineBackend.figure_format = 'retina'

# to display all columns and rows:
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);  # to display all columns and rows
pd.set_option('display.float_format', lambda x: '%.2f' % x) # The number of numbers that will be shown after the comma.

churn = pd.read_csv("../input/churn-for-bank-customers/churn.csv", index_col = 0)
churn.head() # first five row of the dataset
# checking dataset

print ("Rows     : " ,churn.shape[0])
print ("Columns  : " ,churn.shape[1])
print ("\nFeatures : \n" ,churn.columns.tolist())
print ("\nMissing values :  ", churn.isnull().sum().values.sum())
print ("\nUnique values :  \n",churn.nunique())
churn.describe().T
churn["Exited"].value_counts()
#Separating churn and non churn customers
exited     = churn[churn["Exited"] == 1]
not_exited = churn[churn["Exited"] == 0]
df = churn.drop(['CustomerId', 'Surname'], axis = 1)
df.head()
fig, axarr = plt.subplots(2, 3, figsize=(18, 6))
sns.countplot(x = 'Geography', hue = 'Exited',data = df, ax = axarr[0][0])
sns.countplot(x = 'Gender', hue = 'Exited',data = df, ax = axarr[0][1])
sns.countplot(x = 'HasCrCard', hue = 'Exited',data = df, ax = axarr[0][2])
sns.countplot(x = 'IsActiveMember', hue = 'Exited',data = df, ax = axarr[1][0])
sns.countplot(x = 'NumOfProducts', hue = 'Exited',data = df, ax = axarr[1][1])
sns.countplot(x = 'Tenure', hue = 'Exited',data = df, ax = axarr[1][2])

_, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.swarmplot(x = "NumOfProducts", y = "Age", hue="Exited", data = df, ax= ax[0])
sns.swarmplot(x = "HasCrCard", y = "Age", data = df, hue="Exited", ax = ax[1])
sns.swarmplot(x = "IsActiveMember", y = "Age", hue="Exited", data = df, ax = ax[2])
facet = sns.FacetGrid(df, hue = "Exited", aspect = 3)
facet.map(sns.kdeplot, "Age", shade = True)
facet.set(xlim = (0, df["Age"].max()))
facet.add_legend()

plt.show();
_, ax =  plt.subplots(1, 2, figsize = (15, 7))
cmap = sns.cubehelix_palette(light = 1, as_cmap = True)
sns.scatterplot(x = "Age", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax = ax[0])
sns.scatterplot(x = "CreditScore", y = "Balance", hue = "Exited", cmap = cmap, sizes = (10, 200), data = df, ax = ax[1]);
plt.figure(figsize = (10, 10))
sns.swarmplot(x = "HasCrCard", y = "Age", data = df, hue = "Exited")
corr = df.corr()
corr.style.background_gradient(cmap = 'coolwarm')
# NumOfProducts variable is converted to string values.
NumOfProd = []
for i in df['NumOfProducts']:
    if i == 1:
        NumOfProd.append('A')
    elif i == 2:
        NumOfProd.append('B')
    elif i == 3:
        NumOfProd.append('C')
    else:
        NumOfProd.append('D')
        
df['NumOfProducts'] = NumOfProd
df.head()
dummies = pd.get_dummies(df[['Geography', 'Gender', 'NumOfProducts']], drop_first = True) 
X_ = df.drop(['Geography', 'Gender', 'NumOfProducts'], axis = 1)
df_1 = pd.concat([X_, dummies], axis = 1)
df_1.head()
df_1.Balance = df_1.Balance + 1 # To get rid of the problem of dividing by 0
df_1['SalBal'] = df_1.EstimatedSalary / df_1.Balance #The ratio of variables EstimatedSalary and Balance is assigned as a new variable
df_1.head()

df_1.head()
# Standardization on four features
X_s = pd.DataFrame(df_1[['CreditScore', 'Balance', 'EstimatedSalary', 'SalBal']], 
                   columns = ['CreditScore', 'Balance', 'EstimatedSalary', 'SalBal'])

MinMax = MinMaxScaler(feature_range = (0, 1)).fit(X_s)
X_s = MinMax.transform(X_s)
X_st = pd.DataFrame(X_s, columns = ['CreditScore', 'Balance', 'EstimatedSalary', 'SalBal'])
X_st.index = X_st.index + 1
X_st.head()
# We define the dataset with standardized variables as df_2.
df_2 = df_1.drop(['CreditScore', 'Balance', 'EstimatedSalary', 'SalBal'], axis = 1)
df_2 = pd.concat([df_2, X_st], axis = 1, ignore_index = False)
df_2.head()
# credit scores are divided into 6 classes.
CreditScoreClass = []
for cs in churn.CreditScore:
    if 400 <= cs < 500:
        CreditScoreClass.append(1)
    elif 500 <= cs < 700:
        CreditScoreClass.append(2)
    elif  700 <= cs < 800:
        CreditScoreClass.append(3)
    elif  800 <=  cs < 850:
        CreditScoreClass.append(4)
    elif  850 <= cs: 
        CreditScoreClass.append(5)
    elif 400 > cs :
        CreditScoreClass.append(0)

df_2['CreditScoreClass'] = CreditScoreClass
df_2.drop('CreditScore', axis = 1, inplace = True)
df_2.head()
y = df_2['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LGBMClassifier(),
    XGBClassifier()]

result = []
results = pd.DataFrame(columns = ["Models","Accuracy"])

for model in models:
    names = model.__class__.__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores = cross_val_score(model, X_test, y_test, cv = 10, scoring = 'accuracy')
    result = pd.DataFrame([[names, acc * 100, 
                            np.mean(scores) * 100]], 
                          columns = ["Models", "Accuracy", "Avg_Accuracy"])
    results = results.append(result)
results
avg_accuracies={}
accuracies={}
roc_auc={}
pr_auc={}
def cv_score(name, model, folds):
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    avg_result = []
    for sc in scores:
        scores = cross_val_score(model, X_test, y_test, cv = folds, scoring = sc)
        avg_result.append(np.average(scores))
    df_avg_score = pd.DataFrame(avg_result)
    df_avg_score = df_avg_score.rename(index={0: 'Accuracy',
                                             1:'Precision',
                                             2:'Recall',
                                             3:'F1 score',
                                             4:'Roc auc'}, columns = {0: 'Average'})
    avg_accuracies[name] = np.round(df_avg_score.loc['Accuracy'] * 100, 2)
    values = [np.round(df_avg_score.loc['Accuracy'] * 100, 2),
            np.round(df_avg_score.loc['Precision'] * 100, 2),
            np.round(df_avg_score.loc['Recall'] * 100, 2),
            np.round(df_avg_score.loc['F1 score'] * 100, 2),
            np.round(df_avg_score.loc['Roc auc'] * 100, 2)]
    plt.figure(figsize = (15, 8))
    sns.set_palette('mako')
    ax = sns.barplot(x = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'Roc auc'], y = values)
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel('Percentage %', labelpad = 10)
    plt.xlabel('Scoring Parameters', labelpad = 10)
    plt.title('Cross Validation ' + str(folds) + '-Folds Average Scores', pad = 20)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 1.02))
    plt.show()
def conf_matrix(ytest, pred):
    plt.figure(figsize = (15, 8))
    global cm1
    cm1 = confusion_matrix(ytest, pred)
    ax = sns.heatmap(cm1, annot = True, cmap = 'Blues')
    plt.title('Confusion Matrix', pad = 30)
def metrics_score(cm):
    total = sum(sum(cm))
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    specificity = cm[0,0] / (cm[0, 1] + cm[0, 0])
    values = [np.round(accuracy * 100, 2),
            np.round(precision * 100, 2),
            np.round(sensitivity * 100, 2),
            np.round(f1 * 100, 2),
            np.round(specificity * 100, 2)]
    plt.figure(figsize = (15, 8))
    sns.set_palette('magma')
    ax = sns.barplot(x = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'Specificity'], y = values)
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel('Percentage %', labelpad = 10)
    plt.xlabel('Scoring Parameter', labelpad = 10)
    plt.title('Metrics Scores', pad = 20)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 1.02))
    plt.show()
def plot_roc_curve(fpr, tpr):
    plt.figure(figsize = (8, 6))
    plt.plot(fpr, tpr, color = 'Orange', label = 'ROC')
    plt.plot([0, 1], [0, 1], color = 'black', linestyle = '--')
    plt.ylabel('True Positive Rate', labelpad = 10)
    plt.xlabel('False Positive Rate', labelpad = 10)
    plt.title('Receiver Operating Characteristic (ROC) Curve', pad = 20)
    plt.legend()
    plt.show()
def plot_precision_recall_curve(recall, precision):
    plt.figure(figsize = (8,6))
    plt.plot(recall, precision, color = 'orange', label = 'PRC')
    plt.ylabel('Precision', labelpad = 10)
    plt.xlabel('Recall', labelpad = 10)
    plt.title('Precision Recall Curve', pad = 20)
    plt.legend()
    plt.show()
y = df_2['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
prediction1 = log_model.predict(X_test)
accuracy1 = log_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy1 * 100)
accuracies['Linear Regression'] = np.round(accuracy1 * 100, 2)
conf_matrix(y_test, prediction1)
metrics_score(cm1)
cv_score('Linear Regression', log_model, 5)
probs = log_model.predict_proba(X_test)
probs = probs[:, 1]
auc1 = roc_auc_score(y_test, probs)
roc_auc['Linear Regression'] = np.round(auc1, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc1)
fpr1, tpr1, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr1, tpr1)
precision1, recall1, _ = precision_recall_curve(y_test, probs)
auc_score1 = auc(recall1, precision1)
pr_auc['Linear Regression'] = np.round(auc_score1, 2)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score1)
plot_precision_recall_curve(recall1, precision1)
y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)
prediction2 = KNN_model.predict(X_test)
accuracy2 = KNN_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy2 * 100)
accuracies['KNeighbors Classifier'] = np.round(accuracy2 * 100, 2)
conf_matrix(y_test, prediction2)
metrics_score(cm1)
cv_score('KNeighbors Classifier', KNN_model, 5)
probs = KNN_model.predict_proba(X_test)
probs = probs[:, 1]
auc2 = roc_auc_score(y_test, probs)
roc_auc['KNeighbors Classifier'] = np.round(auc2, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc2)
fpr2, tpr2, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr2, tpr2)
precision2, recall2, _ = precision_recall_curve(y_test, probs)
auc_score2 = auc(recall2, precision2)
pr_auc['KNeighbors Classifier'] = np.round(auc_score2, 2)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score2)
plot_precision_recall_curve(recall2, precision2)
y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
SVC_model = SVC(probability = True)
SVC_model.fit(X_train, y_train)
prediction3 = SVC_model.predict(X_test)
accuracy3 = SVC_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy3 * 100)
accuracies['Support Vector Machine Classifier'] = np.round(accuracy3 * 100, 2)
conf_matrix(y_test, prediction3)
metrics_score(cm1);
cv_score('Support Vector Machine Classifier', SVC_model, 5)
probs = SVC_model.predict_proba(X_test)
probs = probs[:, 1]
auc3 = roc_auc_score(y_test, probs)
roc_auc['Support Vector Machine Classifier'] = np.round(auc3, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc3)
fpr3, tpr3, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr3, tpr3)
precision3, recall3, _ = precision_recall_curve(y_test, probs)
auc_score3 = auc(recall3, precision3)
pr_auc['Support Vector Machine Classifier'] = np.round(auc_score3, 2)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score3)
plot_precision_recall_curve(recall3, precision3)
y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
CART_model = DecisionTreeClassifier(max_depth = 10, min_samples_split = 50)
CART_model.fit(X_train, y_train)
prediction4 = CART_model.predict(X_test)
accuracy4 = CART_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy4 * 100)
accuracies['Classification and Regression Tree'] = np.round(accuracy4 * 100, 2)
conf_matrix(y_test, prediction4)
metrics_score(cm1)
cv_score('Classification and Regression Tree', CART_model, 5)
probs = CART_model.predict_proba(X_test)
probs = probs[:, 1]
auc4 = roc_auc_score(y_test, probs)
roc_auc['Desicion Tree Classifier']=np.round(auc4, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc4)
fpr4, tpr4, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr4, tpr4)
precision4, recall4, _ = precision_recall_curve(y_test, probs)
auc_score4 = auc(recall4, precision4)
pr_auc['Desicion Tree Classifier'] = np.round(auc_score4, 2)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score4)
plot_precision_recall_curve(recall4, precision4)
y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
rf_model = RandomForestClassifier(max_features = 3, min_samples_split = 10, n_estimators = 200)
rf_model.fit(X_train, y_train)
prediction5 = rf_model.predict(X_test)
accuracy5 = rf_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy5 * 100)
#rf_params = {"n_estimators": [100, 200, 500, 1000], "max_features": [3, 5, 7, 8], "min_samples_split": [2, 5, 10, 20]}
#rf_cv_model = GridSearchCV(rf, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#rf_cv_model.best_params_
accuracies['Random Forests'] = np.round(accuracy5 * 100, 2)
conf_matrix(y_test, prediction5)
metrics_score(cm1)
cv_score('Random Forests', rf_model, 5)
probs = rf_model.predict_proba(X_test)
probs = probs[:, 1]
auc5 = roc_auc_score(y_test, probs)
roc_auc['Random Forests Classifier']=np.round(auc5, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc5)
fpr5, tpr5, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr5, tpr5)
precision5, recall5, _ = precision_recall_curve(y_test, probs)
auc_score5 = auc(recall5, precision5)
pr_auc['Random Forests'] = np.round(auc_score5,3)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score5)
plot_precision_recall_curve(recall5, precision5)
feature_imp = pd.Series(rf_model.feature_importances_,
                        index = X_train.columns).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Important Scores')
plt.ylabel('Features')
plt.title("Feature Important Range")
plt.show()
y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
gbm_model = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 5, n_estimators = 300)
gbm_model.fit(X_train, y_train)
prediction6 = gbm_model.predict(X_test)
accuracy6 = gbm_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy6 * 100)
#gbm_params = {"learning_rate": [0.1, 0.01, 0.001, 0.05],"n_estimators": [100, 300, 500, 1000], "max_depth":[2, 3, 5, 8]}
#gbm_cv_model= GridSearchCV(gbm_model, gbm_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#gbm_cv_model.best_params_
accuracies['Gradient Boosting Machines'] = np.round(accuracy6 * 100, 2)
conf_matrix(y_test, prediction6)
metrics_score(cm1)
cv_score('Gradient Boosting Machines', gbm_model, 5)
probs = gbm_model.predict_proba(X_test)
probs = probs[:, 1]
auc6 = roc_auc_score(y_test, probs)
roc_auc['Gradient Boosting Machine Classifier'] = np.round(auc6, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc6)
fpr6, tpr6, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr6, tpr6)
precision6, recall6, _ = precision_recall_curve(y_test, probs)
auc_score6 = auc(recall6, precision6)
pr_auc['Gradient Boosting Machine Classifier'] = np.round(auc_score6, 2)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score6)
plot_precision_recall_curve(recall6, precision6)
feature_imp = pd.Series(gbm_model.feature_importances_,
                        index = X_train.columns).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Important Scores')
plt.ylabel('Features')
plt.title("Feature Important Range")
plt.show()
y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
xgb_model = XGBClassifier(learning_rate = 0.01, max_depth = 5, n_estimators = 1000, subsample = 0.8)
xgb_model.fit(X_train, y_train)
prediction7 = xgb_model.predict(X_test)
accuracy7 = xgb_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy7 * 100)
#xgb_params = {"n_estimators": [100, 500, 1000], "subsample":[0.5, 0.8 ,1], "max_depth":[3, 5, 7], "learning_rate":[0.1, 0.001, 0.01, 0.05]}
#xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#xgb_cv_model.best_params_
accuracies['XGBoost Classifier'] = np.round(accuracy7 * 100, 2)
conf_matrix(y_test, prediction7)
metrics_score(cm1)
cv_score('XGBoost Classifier', xgb_model, 5)
probs = xgb_model.predict_proba(X_test)
probs = probs[:, 1]
auc7 = roc_auc_score(y_test, probs)
roc_auc['XGB Machine Classifier']=np.round(auc7, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc7)
fpr7, tpr7, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr7, tpr7)
precision7, recall7, _ = precision_recall_curve(y_test, probs)
auc_score7 = auc(recall7, precision7)
pr_auc['XGB Machine Classifier'] = np.round(auc_score7, 2)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score7)
plot_precision_recall_curve(recall7, precision7)
feature_imp = pd.Series(gbm_model.feature_importances_,
                        index = X_train.columns).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Important Scores')
plt.ylabel('Features')
plt.title("Feature Important Range")
plt.show()
y = df_1['Exited']
X = df_2.drop('Exited', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12345)
lgbm_model = LGBMClassifier(learning_rate = 0.1, max_depth = 2, n_estimators = 500)
lgbm_model.fit(X_train, y_train)
prediction8 = lgbm_model.predict(X_test)
accuracy8 = lgbm_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy8 * 100)
#lgbm_params = {"learning_rate": [0.001, 0.01, 0.1], "n_estimators": [200, 500, 100], "max_depth":[1,2,5,8]}
#lgbm_cv_model = GridSearchCV(lgbm_model,lgbm_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#lgbm_cv_model.best_params_
accuracies['LightGBM Classifier'] = np.round(accuracy8 * 100, 2)
conf_matrix(y_test, prediction8)
metrics_score(cm1)
cv_score('LightGBM Classifier', lgbm_model, 5)
probs = lgbm_model.predict_proba(X_test)
probs = probs[:, 1]
auc8 = roc_auc_score(y_test, probs)
roc_auc['LightGBM Classifier'] = np.round(auc8, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc8)
fpr8, tpr8, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr8, tpr8)
precision8, recall8, _ = precision_recall_curve(y_test, probs)
auc_score8 = auc(recall8, precision8)
pr_auc['LightGBM Classifier'] = np.round(auc_score8, 2)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score8)
plot_precision_recall_curve(recall8, precision8)
feature_imp = pd.Series(gbm_model.feature_importances_,
                        index = X_train.columns).sort_values(ascending = False)

sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Important Scores')
plt.ylabel('Features')
plt.title("Feature Important Range")
plt.show()

models_tuned = [
    log_model,
    KNN_model,
    SVC_model,
    CART_model,
    rf_model,
    gbm_model,
    lgbm_model,
    xgb_model]

result = []
results = pd.DataFrame(columns = ["Models","Accuracy"])

for model in models_tuned:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    scores = cross_val_score(model, X_test, y_test, cv = 10, scoring = 'accuracy')
    result = pd.DataFrame([[names, acc * 100, 
                            np.mean(scores) * 100]], 
                          columns = ["Models", "Accuracy", "Avg_Accuracy"])
    results = results.append(result)
results
plt.figure(figsize = (15, 8))
sns.set_palette('cividis')
ax = sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel('Percentage %', labelpad = 10)
plt.xlabel('Algorithms', labelpad = 10)
plt.title('Accuracy Scores Comparison', pad = 20)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 1.02))
plt.show()
plt.figure(figsize = (15, 8))
sns.set_palette('viridis')
ax=sns.barplot(x = list(avg_accuracies.keys()), y = list(avg_accuracies.values()))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel('Percentage %', labelpad = 10)
plt.xlabel('Algorithms', labelpad = 10)
plt.title('Average Accuracy Scores Comparison', pad = 20)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),xytext=(p.get_x() + 0.3, p.get_height() + 1.02))
plt.show()
plt.figure(figsize = (8, 6))
sns.set_palette('Set1')
plt.plot(fpr1, tpr1, label = 'Linear Regression')
plt.plot(fpr2, tpr2, label = 'KNeiihbors Classifier')
plt.plot(fpr3, tpr3, label = 'SVM')
plt.plot(fpr4, tpr4, label = 'Decision Tree')
plt.plot(fpr5, tpr5, label = 'Random Forests')
plt.plot(fpr6, tpr6, label = 'Gradient Boosting MachineC')
plt.plot(fpr7, tpr7, label = 'XGBoost')
plt.plot(fpr8, tpr8, label = 'LightGBM')
plt.plot([0, 1], [0, 1], linestyle = '--')
plt.ylabel('True Positive Rate', labelpad = 10)
plt.xlabel('False Positive Rate', labelpad = 10)
plt.title('Receiver Operating Characteristic (ROC) Curves', pad = 20)
plt.legend()
plt.show()
plt.figure(figsize = (15, 8))
sns.set_palette('magma')
ax = sns.barplot(x = list(roc_auc.keys()), y = list(roc_auc.values()))
#plt.yticks(np.arange(0,100,10))
plt.ylabel('Score', labelpad = 10)
plt.xlabel('Algorithms', labelpad = 10)
plt.title('Area under the ROC Curves (AUC)', pad = 20)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 0.01))
plt.show()
plt.figure(figsize = (8, 6))
sns.set_palette('Set1')
plt.plot(recall1, precision1, label = 'Linear Regression PRC')
plt.plot(recall2, precision2, label = 'KNN PRC')
plt.plot(recall3, precision3, label = 'SVM PRC')
plt.plot(recall4, precision4, label = 'CART PRC')
plt.plot(recall5, precision5, label = 'Random Forests PRC')
plt.plot(recall6, precision6, label = 'GBM PRC')
plt.plot(recall7, precision7, label = 'XGB PRC')
plt.plot(recall8, precision8, label = 'LGBM PRC')
plt.ylabel('Precision', labelpad = 10)
plt.xlabel('Recall', labelpad = 10)
plt.title('Precision Recall Curves', pad = 20)
plt.legend()
plt.show()
plt.figure(figsize = (15, 8))
sns.set_palette('mako')
ax = sns.barplot(x = list(pr_auc.keys()), y = list(pr_auc.values()))
plt.ylabel('Score', labelpad = 10)
plt.xlabel('Algorithms', labelpad = 10)
plt.title('Area under the PR Curves (AUCPR)', pad = 20)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()), xytext = (p.get_x() + 0.3, p.get_height() + 0.01))
plt.show()