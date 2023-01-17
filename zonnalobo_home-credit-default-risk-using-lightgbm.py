import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
pd.set_option('Display.max_columns',500)
pd.set_option('Display.max_rows',500)
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
app_train.head()
app_train.isnull().sum()
app_train[app_train.isnull().any(axis=1)].shape
app_train[~app_train.isnull().any(axis=1)].shape
app_train.select_dtypes(include=object).dtypes
app_test.select_dtypes(include=object).isnull().sum()
app_train.select_dtypes(include=object).isnull().sum()
app_train.NAME_TYPE_SUITE = app_train.NAME_TYPE_SUITE.fillna('Unaccompanied')
app_test.NAME_TYPE_SUITE = app_test.NAME_TYPE_SUITE.fillna('Unaccompanied')
app_train.OCCUPATION_TYPE = app_train.OCCUPATION_TYPE.fillna('Others')
app_test.OCCUPATION_TYPE = app_test.OCCUPATION_TYPE.fillna('Others')
app_train.FONDKAPREMONT_MODE = app_train.FONDKAPREMONT_MODE.fillna('not specified')
app_test.FONDKAPREMONT_MODE = app_test.FONDKAPREMONT_MODE.fillna('not specified')
app_train.HOUSETYPE_MODE = app_train.HOUSETYPE_MODE.fillna('Others')
app_test.HOUSETYPE_MODE = app_test.HOUSETYPE_MODE.fillna('Others')
app_train.WALLSMATERIAL_MODE = app_train.WALLSMATERIAL_MODE.fillna('Others')
app_test.WALLSMATERIAL_MODE = app_test.WALLSMATERIAL_MODE.fillna('Others')
app_train = app_train.drop(columns=['EMERGENCYSTATE_MODE'])
app_test = app_test.drop(columns=['EMERGENCYSTATE_MODE'])
min_ext_1 = app_train.EXT_SOURCE_1.min()
min_ext_2 = app_train.EXT_SOURCE_2.min()
min_ext_3 = app_train.EXT_SOURCE_3.min()
app_train.EXT_SOURCE_1 = app_train.EXT_SOURCE_1.fillna(min_ext_1)
app_train.EXT_SOURCE_2 = app_train.EXT_SOURCE_2.fillna(min_ext_2)
app_train.EXT_SOURCE_3 = app_train.EXT_SOURCE_3.fillna(min_ext_3)

app_test.EXT_SOURCE_1 = app_test.EXT_SOURCE_1.fillna(min_ext_1)
app_test.EXT_SOURCE_2 = app_test.EXT_SOURCE_2.fillna(min_ext_2)
app_test.EXT_SOURCE_3 = app_test.EXT_SOURCE_3.fillna(min_ext_3)

app_train = app_train.fillna(0)
app_test = app_test.fillna(0)
print(app_train.isnull().sum().sum())
print(app_test.isnull().sum().sum())
x_cat = app_train[app_train.select_dtypes(include=object).columns].columns
x_num = app_train[app_train.select_dtypes(exclude=object).columns].columns.drop(['SK_ID_CURR','TARGET'])
def plot_hist(x):
    plt.rcParams["figure.figsize"] = (10,8)
    ax = sns.countplot(x=x,data=app_train)
    plt.xlabel(str(x))
    plt.title('Histogram of '+str(x))
    plt.xticks(rotation=70)
    plt.show()
plot_hist('TARGET')
for x in x_cat:
    plot_hist(x)
def plot_dist(x):
    plt.rcParams["figure.figsize"] = (10,8)
    ax = sns.distplot(app_train[x])
    plt.xlabel(str(x))
    plt.title('Distribution of '+str(x))
    plt.show()
for x in x_num:
    plot_dist(x)
def plot_hist(x):
    plt.rcParams["figure.figsize"] = (10,8)
    ax = sns.countplot(x=x,data=app_test)
    plt.xlabel(str(x))
    plt.title('Histogram of '+str(x))
    plt.xticks(rotation=70)
    plt.show()
for x in x_cat:
    plot_hist(x)
def plot_dist(x):
    plt.rcParams["figure.figsize"] = (10,8)
    ax = sns.distplot(app_test[x])
    plt.xlabel(str(x))
    plt.title('Distribution of '+str(x))
    plt.show()
for x in x_num:
    plot_dist(x)
categorical = app_train[app_train.select_dtypes(include=object).columns]
x_cat = categorical.columns
categorical.head()
numerical = app_train[app_train.select_dtypes(exclude=object).columns]
numerical = numerical.drop(columns=['SK_ID_CURR','TARGET'])
x_num = numerical.columns
numerical.head()
target = app_train.TARGET
target.head()
scaller = MinMaxScaler()
app_train[x_num] = scaller.fit_transform(app_train[x_num])
app_test[x_num] = scaller.transform(app_test[x_num])
app_train[x_num].head()
for x in x_cat:
    lb = LabelEncoder()
    app_train[x] = lb.fit_transform(app_train[x])
    app_test[x] = lb.transform(app_test[x])
app_train[x_cat].head()
lb = LabelBinarizer()
app_train['TARGET'] = lb.fit_transform(app_train.TARGET)
app_train.TARGET.head()
app_train.head()
x_call = app_train.columns[2:]
app_test.head()
corr = app_train[x_num].corr()
cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
    .set_caption("Hover to magify")\
    .set_precision(2)\
    .set_table_styles(magnify())
fig,ax = plt.subplots(figsize=(12,10))
corr = app_train[x_call].corr()
hm = sns.heatmap(corr,ax=ax,vmin=-1,vmax=1,annot=False,cmap='coolwarm',square=True,fmt='.2f',linewidths=.05)
for x in x_call:
    msg = "%s : %.3f" % (x,np.corrcoef(app_train[x],app_train.TARGET)[0,1])
    print(msg)
train_df, test_df = train_test_split(app_train,test_size=0.33,shuffle=True,stratify=app_train.TARGET,
                                     random_state=217)
from sklearn.model_selection import StratifiedKFold,cross_validate,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,log_loss,roc_curve
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score,brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.utils import compute_sample_weight,compute_class_weight
from sklearn.calibration import calibration_curve,CalibratedClassifierCV
from sklearn import model_selection

x_calls = train_df.columns[2:]
scorer = ('accuracy','roc_auc','f1_weighted','average_precision')
models = []
models.append(('LR', LogisticRegression(class_weight='balanced')))
models.append(('CART', DecisionTreeClassifier(class_weight='balanced')))
models.append(('NB', GaussianNB()))
models.append(('RFC', RandomForestClassifier(class_weight='balanced')))
models.append(('ETC', ExtraTreesClassifier(class_weight='balanced')))
models.append(('XGBC', XGBClassifier(scale_pos_weight=189399/16633)))
models.append(('GBM', LGBMClassifier(class_weight='balanced')))
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=217, shuffle=True)
    cv_results = cross_validate(model, train_df[x_calls], train_df.TARGET,cv=kfold, scoring=scorer)
    cv_results1=cv_results['test_accuracy']
    cv_results2=cv_results['test_roc_auc']
    cv_results3=cv_results['test_f1_weighted']
    cv_results4=cv_results['test_average_precision']
    msg = "%s by Accuracy: %f(%f), by ROC_AUC: %f(%f), by F1-score: %f(%f), PR_AUC: %f(%f)" % (name, np.mean(cv_results1),
        np.std(cv_results1),np.mean(cv_results2),np.std(cv_results2),np.mean(cv_results3),np.std(cv_results3),
        np.mean(cv_results4),np.std(cv_results4))
    print(msg)
model_gbm = LGBMClassifier(boosting_type='gbdt', class_weight='balanced',
        colsample_bytree=1.0, importance_type='split',
        learning_rate=0.09275695087706179, max_depth=3669,
        min_child_samples=60, min_child_weight=0.001, min_data=6,
        min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=64,
        objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
        silent=True, sub_feature=0.7757070409332384, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0)
model_gbm.fit(train_df[x_calls], train_df.TARGET)
predictions = model_gbm.predict(test_df[x_calls])
prob = model_gbm.predict_proba(test_df[x_calls])[:,1]
test_df['TARGET_hat']=predictions
test_df['TARGET_prob']=prob
Y_validation = test_df.TARGET
print("Accuracy Score: %f" % accuracy_score(Y_validation, predictions))
print("ROC_AUC: %f" % roc_auc_score(Y_validation, prob,average='weighted'))
print("PR_AUC: %f" % average_precision_score(Y_validation, prob,average='weighted'))
print("F1: %f" % f1_score(Y_validation, predictions,average='weighted'))
print("Recall: %f" % recall_score(Y_validation, predictions,average='weighted'))
print("Precision: %f" % precision_score(Y_validation, predictions,average='weighted'))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(test_df.TARGET, test_df.TARGET_prob)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
def plot_calibration_curve(est, name, X_train, y_train, X_test, y_test):
    isotonic = CalibratedClassifierCV(est, cv='prefit', method='isotonic')
    sigmoid = CalibratedClassifierCV(est, cv='prefit', method='sigmoid')
    lr = LogisticRegression(C=1., solver='lbfgs',class_weight='balanced')
    fig = plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),(est, name),(isotonic, name + ' + Isotonic'),(sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_train.max())
        L2_score = log_loss(y_test, prob_pos)
        print("%s:" % name)
        print("\tBrier: %.3f" % (clf_score))
        print("\tLog Loss: %.3f" % (L2_score))
        print("\tAUC: %.3f" % roc_auc_score(y_test, prob_pos,average='weighted'))
        print("\tF1: %.3f\n" % f1_score(y_test, y_pred,average='weighted'))
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label="%s (%1.3f)" % (name, clf_score))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,histtype="step", lw=2)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
plot_calibration_curve(model_gbm,'LGBM',train_df[x_calls], train_df['TARGET'],
                       test_df[x_calls], test_df['TARGET'])
model_fix = LGBMClassifier(boosting_type='gbdt', class_weight='balanced',
        colsample_bytree=1.0, importance_type='split',
        learning_rate=0.09275695087706179, max_depth=3669,
        min_child_samples=60, min_child_weight=0.001, min_data=6,
        min_split_gain=0.0, n_estimators=100, n_jobs=-1, num_leaves=64,
        objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
        silent=True, sub_feature=0.7757070409332384, subsample=1.0,
        subsample_for_bin=200000, subsample_freq=0)
model_fix.fit(app_train[x_calls],app_train.TARGET)
importances = model_fix.feature_importances_
indices = np.argsort(importances)[::-1]
def variable_importance(importance, indices,x):
    print("Feature ranking:")
    importances = []
    for f in range(len(x)):
        i = f
        t=0
        print("%d. The feature '%s' has a Mean Decrease in Gini of %f" % (f + 1,x[indices[i]],importance[indices[f]]))
        importances.append([x[indices[i]],importance[indices[f]]])
    importances = pd.DataFrame(importances,columns=['Features','Gini'])
    return importances

importance = variable_importance(importances, indices,x_calls)
model_fix = CalibratedClassifierCV(model_fix, cv='prefit', method='sigmoid')
model_fix.fit(app_train[x_calls],app_train.TARGET)
TARGET = model_fix.predict_proba(app_test[x_calls])[:,1]
submission = pd.DataFrame({'SK_ID_CURR':app_test['SK_ID_CURR'],'TARGET':TARGET})
submission.to_csv('submission.csv', index=False)
submission.head()