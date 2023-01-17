import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import missingno as msno

import warnings
warnings.simplefilter(action = 'ignore')
diabetes = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

diabetes.head()
diabetes.info()
diabetes.describe([0.10, 0.25, 0.40, 0.50,0.70, 0.90,0.95, 0.99]).T
df = diabetes.copy()
df["Outcome"].value_counts()
df.corr()
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
df.head()
df.groupby("Outcome").agg({"Pregnancies":"mean"})
df.groupby("Outcome").agg({"Age":"mean"})
df.groupby("Outcome").agg({"Age":"max"})
df.groupby("Outcome").agg({"Insulin": "mean"})
df.groupby("Outcome").agg({"Insulin": "max"})
df.groupby("Outcome").agg({"Glucose": "mean"})
df.groupby("Outcome").agg({"Glucose": "max"})
df.groupby("Outcome").agg({"BMI": "mean"})
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.isnull().sum()
msno.bar(df);
def median_target(sfy):   
    temp = df[df[sfy].notnull()]
    temp = temp[[sfy, 'Outcome']].groupby(['Outcome'])[[sfy]].median().reset_index()
    return temp

columns = df.columns
columns = columns.drop("Outcome")
for i in columns:
    median_target(i)
    df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
    df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]
df.head()
df.isnull().sum()
sns.set(font_scale=0.7) 
fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2,figsize=(7,12))
fig.tight_layout()
for ax,col in zip(axes.flatten(),df.columns):
    sns.boxplot(x=df[col],ax=ax)
for feature in df:
    
    Q1 = df[feature].quantile(0.05)
    Q3 = df[feature].quantile(0.95)
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    
    if df[(df[feature] > upper)].any(axis=None):
        print(feature,"yes")
    else:
        print(feature, "no")
Q1 = df.Insulin.quantile(0.25)
Q3 = df.Insulin.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Insulin"] > upper,"Insulin"] = upper
Q1 = df.SkinThickness.quantile(0.25)
Q3 = df.SkinThickness.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["SkinThickness"] > upper,"SkinThickness"] = upper
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_
df_scores = pd.DataFrame(np.sort(df_scores))
df_scores.plot(stacked=True, xlim=[0,60], style='.-'); # first 20 rows
    
df_scores[0:20]
df_scores.iloc[4,:]
threshold = np.sort(df_scores)[4]
new_df = df[np.array(df_scores > threshold)]
new_df.info()
df = new_df
df.describe().T

NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]
df.head()
def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"
    
df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))
df.head()
NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")
df["NewGlucose"] = NewGlucose
df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]
df.head()
df = pd.get_dummies(df, columns =["NewBMI","NewInsulinScore", "NewGlucose"], drop_first = True)
df.head()
categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]
categorical_df.head()
X_ = df.drop(['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                     'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis = 1)

X_.head()
df = pd.concat([X_,categorical_df], axis = 1)
df.head()
y = df['Outcome']
X = df.drop('Outcome', axis = 1)
X.head()
MinMax = MinMaxScaler(feature_range = (0, 1)).fit(X)
X_st = MinMax.transform(X)
X_st = pd.DataFrame(X_st, columns = X.columns)
X_st.head()
sdf = pd.concat([X_st, y], axis = 1)
sdf.describe().T

avg_accuracies={}
accuracies={}
roc_auc={}
pr_auc={}
def cal_score(name,model,folds):
    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    avg_result = []
    for sc in scores:
        scores = cross_val_score(model, X_st, y, cv = folds, scoring = sc)
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
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, color='orange', label='PRC')
    plt.ylabel('Precision',labelpad=10)
    plt.xlabel('Recall',labelpad=10)
    plt.title('Precision Recall Curve',pad=20)
    plt.legend()
    plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
prediction1 = log_model.predict(X_test)
accuracy1 = log_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy1 * 100)
accuracies['Linear Regression'] = np.round(accuracy1 * 100, 2)
conf_matrix(y_test, prediction1)
metrics_score(cm1)
cal_score('Linear Regression', log_model, 5)
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)
prediction2 = KNN_model.predict(X_test)
accuracy2 = KNN_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy2 * 100)
accuracies['KNeighbors Classifier'] = np.round(accuracy2 * 100, 2)
conf_matrix(y_test, prediction2)
metrics_score(cm1)
cal_score('KNeighbors Classifier', KNN_model, 5)
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
SVC_model = SVC(probability = True)
SVC_model.fit(X_train, y_train)
prediction3 = SVC_model.predict(X_test)
accuracy3 = SVC_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy3 * 100)
accuracies['Support Vector Machine Classifier'] = np.round(accuracy3 * 100, 2)
conf_matrix(y_test, prediction3)
metrics_score(cm1)
cal_score('Support Vector Machine Classifier', SVC_model, 5)
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
CART_model = DecisionTreeClassifier(max_depth = 10, min_samples_split = 50)
CART_model.fit(X_train, y_train)
prediction4 = CART_model.predict(X_test)
accuracy4 = CART_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy4 * 100)
accuracies['Classification and Regression Tree'] = np.round(accuracy4 * 100, 2)
conf_matrix(y_test, prediction4)
metrics_score(cm1)
cal_score('Classification and Regression Tree', CART_model, 5)
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
rf_model = RandomForestClassifier(max_features = 8, min_samples_split = 12, n_estimators = 120)
rf_model.fit(X_train, y_train)
prediction5 = rf_model.predict(X_test)
accuracy5 = rf_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy5 * 100)
accuracies['Random Forests'] = np.round(accuracy5 * 100, 2)
conf_matrix(y_test, prediction5)
metrics_score(cm1)
cal_score('Random Forests', rf_model, 5)
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
gbm_model = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, n_estimators = 500)
gbm_model.fit(X_train, y_train)
prediction6 = gbm_model.predict(X_test)
accuracy6 = gbm_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy6 * 100)
accuracies['Gradient Boosting Machines'] = np.round(accuracy6 * 100, 2)
conf_matrix(y_test, prediction6)
metrics_score(cm1)
cal_score('Gradient Boosting Machines', gbm_model, 5)
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
xgb_model = XGBClassifier(learning_rate = 0.01,max_depth = 3, n_estimators = 500, subsample = 1 )
xgb_model.fit(X_train, y_train)
prediction7 = xgb_model.predict(X_test)
accuracy7 = xgb_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy7 * 100)
accuracies['XGBoost Classifier'] = np.round(accuracy7 * 100, 2)
conf_matrix(y_test, prediction7)
metrics_score(cm1)
cal_score('XGBoost Classifier', xgb_model, 5)
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
lgbm_model = LGBMClassifier(learning_rate = 0.1, max_depth = 1, n_estimators = 200)
lgbm_model.fit(X_train, y_train)
prediction8 = lgbm_model.predict(X_test)
accuracy8 = lgbm_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy8 * 100)
accuracies['LightGBM Classifier'] = np.round(accuracy8 * 100, 2)
conf_matrix(y_test, prediction8)
metrics_score(cm1)
cal_score('LightGBM Classifier', lgbm_model, 5)
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
catboost_model = CatBoostClassifier(depth = 8, iterations = 100, learning_rate = 0.1)
catboost_model.fit(X_train, y_train)
prediction9 = catboost_model.predict(X_test)
accuracy9 = catboost_model.score(X_test, y_test) 
print ('Model Accuracy:', accuracy9 * 100)
accuracies['CatBoost Classifier'] = np.round(accuracy9 * 100, 2)
conf_matrix(y_test, prediction9)
metrics_score(cm1)
cal_score('CatBoost Classifier', catboost_model, 5)
probs = catboost_model.predict_proba(X_test)
probs = probs[:, 1]
auc9 = roc_auc_score(y_test, probs)
roc_auc['CatBoost Classifier']=np.round(auc9, 2)
print('Area under the ROC Curve (AUC): %.2f' % auc9)
fpr9, tpr9, _ = roc_curve(y_test, probs)
plot_roc_curve(fpr9, tpr9)
precision9, recall9, _ = precision_recall_curve(y_test, probs)
auc_score9 = auc(recall9, precision9)
pr_auc['CatBoost Classifier'] = np.round(auc_score9, 2)
print('Area under the PR Curve (AUCPR): %.2f' % auc_score9)
plot_precision_recall_curve(recall9, precision9)
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
    catboost_model,
    lgbm_model,
    xgb_model]

result = []
results = pd.DataFrame(columns = ["Models","Accuracy"])

for model in models_tuned:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    result = pd.DataFrame([[names, acc * 100]], columns = ["Models", "Accuracy"])
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
plt.plot(fpr9, tpr9, label = 'CatBosst')
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
plt.plot(recall5, precision5, label = 'LGBM PRC')
plt.plot(recall6, precision6, label = 'CatBoost PRC')
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
X_train, X_test, y_train, y_test = train_test_split(X_st, y, test_size = 0.20, random_state = 5)
rf_model = RandomForestClassifier(max_features = 8, min_samples_split = 12, n_estimators = 120)
rf_model.fit(X_train, y_train)
prediction5 = rf_model.predict(X_test)
accuracy5 = rf_model.score(X_test, y_test) 
print ('Model Accuracy:',accuracy5 * 100)
conf_matrix(y_test, prediction5)
metrics_score(cm1)
cal_score('Random Forests', rf_model, 10)

