import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

import eli5

from eli5.sklearn import PermutationImportance

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import RobustScaler

import shap

from pdpbox import pdp, get_dataset, info_plots
raw_data = pd.read_csv('../input/creditcard.csv')

raw_data.head()
rob_scaler = RobustScaler()

raw_data['Time_Int'] = raw_data['Time'].diff() + raw_data['Time'].diff().mean()

raw_data['Time_Int'][0] =  raw_data['Time'].diff().mean()

raw_data['Scl_Amount'] = rob_scaler.fit_transform(raw_data['Amount'].values.reshape(-1,1))

raw_data.drop(['Time', 'Amount'], axis=1, inplace=True)

raw_data.head()
raw_data.describe().transpose()
print(raw_data.groupby(['Class']).Class.count())

sns.set_style('dark')

plt.figure(figsize = (10,5))

sns.countplot(raw_data['Class'], 

              alpha =.60, 

              palette= ['lightgreen','red'])

plt.title('Fraud vs Non Fraud')

plt.ylabel('# Cases')

plt.show()
X = raw_data.drop('Class', axis = 1)

y = raw_data['Class']

cols = X.columns.tolist()
sns.set_style('dark')

fig = plt.figure(figsize= (20,40))

fig.subplots_adjust(hspace = 0.30, wspace = 0.30)

k=0

for i in range(1,len(raw_data.columns)+1):

    ax = fig.add_subplot(11,3,i)

    sns.boxplot(x = 'Class', 

                y = X.columns[k], 

                data = raw_data, 

                palette = 'Blues')

    k = k + 1

    if k == len(X.columns): break

plt.show()
sns.set_style('dark')

fig = plt.figure(figsize = (20,40))

fig.subplots_adjust(hspace = 0.30, 

                    wspace = 0.30)

k=0

for i in range(1, len(X.columns) + 1):

    ax = fig.add_subplot(11, 3, i)

    sns.distplot(X[X.columns[k]], 

                 color = 'teal')

    k = k + 1

    if k == len(X.columns): break

plt.show()
fig, ax = plt.subplots(figsize=(10, 8))

corr = X.corr()

sns.heatmap(corr, 

            mask = np.zeros_like(corr, 

                                 dtype=np.bool), 

            cmap = sns.diverging_palette(275, 

                                         150, 

                                         as_cmap=True), 

            square = True, 

            ax = ax)

plt.title('Correlation matrix of the imbalanced data')
sm = SMOTE(random_state=101)

X_sm, y_sm = sm.fit_sample(X, y.ravel())

bal_data = pd.DataFrame(X_sm)

bal_data.columns = cols

bal_data['Class'] = y_sm

print(bal_data.groupby(['Class']).Class.count())

sns.set_style('dark')

plt.figure(figsize = (10,5))

sns.countplot(bal_data['Class'], 

              alpha =.60, 

              palette= ['lightgreen','red'])

plt.title('Fraud vs Non Fraud')

plt.ylabel('# Cases')

plt.show()
bal_data = bal_data.drop('Class', axis = 1)

fig, ax = plt.subplots(figsize=(10, 8))

corr = bal_data.corr()

sns.heatmap(corr, 

            mask = np.zeros_like(corr, 

                                 dtype=np.bool), 

            cmap = sns.diverging_palette(275, 

                                         150, 

                                         as_cmap=True), 

            square = True, 

            ax = ax)

plt.title('Correlation matrix of the balanced data (SMOTE)')
un_sam = RandomUnderSampler(random_state=101)

X_un_sam, y_un_sam = un_sam.fit_sample(X, y.ravel())

bal_data = pd.DataFrame(X_un_sam)

bal_data.columns = cols

bal_data['Class'] = y_un_sam

print(bal_data.groupby(['Class']).Class.count())

sns.set_style('dark')

plt.figure(figsize = (10,5))

sns.countplot(bal_data['Class'], 

              alpha =.60, 

              palette= ['lightgreen','red'])

plt.title('Fraud vs Non Fraud')

plt.ylabel('# Cases')

plt.show()
bal_data = bal_data.drop('Class', axis = 1)

fig, ax = plt.subplots(figsize=(10, 8))

corr = bal_data.corr()

sns.heatmap(corr, 

            mask = np.zeros_like(corr, 

                                 dtype=np.bool), 

            cmap = sns.diverging_palette(275, 

                                         150, 

                                         as_cmap=True), 

            square = True, 

            ax = ax)

plt.title('Correlation matrix of the balanced data (under sampling)')
X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size = .30, 

                                                    random_state = 101)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

rf = RandomForestClassifier(n_jobs = -1, 

                            random_state = 101)

rf.fit(X_train_res, y_train_res)

feat_imp = rf.feature_importances_

sns.set(style="dark")

fig, ax = plt.subplots(figsize=(10, 8))

var_imp = pd.DataFrame({'feature':cols, 

                        'importance':feat_imp})

var_imp = var_imp.sort_values(ascending=False, 

                              by='importance')

ax = sns.barplot(x='importance', 

                 y='feature', 

                 data=var_imp)
var_imp['feature_imp_cumsum'] = var_imp['importance'].cumsum()

var_imp
top_features = SelectFromModel(rf, threshold=0.01)

top_features.fit(X_train_res, y_train_res)

rf_features= X_train.columns[(top_features.get_support())]

rf_features = rf_features.tolist()
perm = PermutationImportance(rf.fit(X_train_res, 

                                    y_train_res), 

                             random_state=1).fit(X_train_res,

                                                 y_train_res)

eli5.show_weights(perm, 

                  feature_names = X_train.columns.tolist(), 

                  top=(30))
pi_features = eli5.explain_weights_df(perm, feature_names = X_train.columns.tolist())

pi_features = pi_features.loc[pi_features['weight'] >= 0.01]['feature'].tolist()
print("\nFeatures from random forest", rf_features)

print("\nFeatures from permutation importance", pi_features)
X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size=0.3, 

                                                    random_state=101)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

lreg = LogisticRegression()

lreg_model_all = lreg.fit(X_train_res, y_train_res)

y_pred_lreg_all = lreg_model_all.predict(X_test)



print('Confusion Matrix')

print('__'*10)

print(confusion_matrix(y_test, 

                       y_pred_lreg_all))

print('__'*30)

print('\nClassification Metrics')

print('__'*30)

print(classification_report(y_test, 

                            y_pred_lreg_all))

print('__'*30)

logreg_accuracy = round(accuracy_score(y_test, 

                                       y_pred_lreg_all) * 100,2)

print('Accuracy', logreg_accuracy,'%')
fpr_lreg_all, tpr_lreg_all, thresholds = roc_curve(y_test, 

                                                   y_pred_lreg_all)

roc_auc_lreg_all = auc(fpr_lreg_all,

                       tpr_lreg_all)

plt.title('Receiver Operating Characteristic - LReg (All features)')

plt.plot(fpr_lreg_all, 

         tpr_lreg_all, 

         'b',

         label='AUC = %0.3f'% roc_auc_lreg_all)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
X_rf = X[rf_features]

X_train, X_test, y_train, y_test = train_test_split(X_rf,

                                                    y, 

                                                    test_size=0.3, 

                                                    random_state=101)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

lreg = LogisticRegression()

lreg_model_rf = lreg.fit(X_train_res, y_train_res)

y_pred_lreg_rf = lreg_model_rf.predict(X_test)



print('Confusion Matrix')

print('__'*10)

print(confusion_matrix(y_test, 

                       y_pred_lreg_rf))

print('__'*30)

print('\nClassification Metrics')

print('__'*30)

print(classification_report(y_test, 

                            y_pred_lreg_rf))

print('__'*30)

logreg_accuracy = round(accuracy_score(y_test, 

                                       y_pred_lreg_rf) * 100,2)

print('Accuracy', logreg_accuracy,'%')
fpr_lreg_rf, tpr_lreg_rf, thresholds_rf = roc_curve(y_test, 

                                                   y_pred_lreg_rf)

roc_auc_lreg_rf = auc(fpr_lreg_rf,

                       tpr_lreg_rf)

plt.title('Receiver Operating Characteristic - LReg (All features + RF Features)')

plt.plot(fpr_lreg_rf, 

         tpr_lreg_rf, 

         'g',

         label='AUC = %0.3f'% roc_auc_lreg_rf)

plt.plot(fpr_lreg_all, 

         tpr_lreg_all, 

         'b',

         label='AUC = %0.3f'% roc_auc_lreg_all)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
X_pf = X[pi_features]

X_train, X_test, y_train, y_test = train_test_split(X_pf, 

                                                    y, 

                                                    test_size=0.3, 

                                                    random_state=101)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

lreg_model_pi = lreg.fit(X_train_res, y_train_res)

y_pred_lreg_pi = lreg_model_pi.predict(X_test)



print('Confusion Matrix')

print('__'*10)

print(confusion_matrix(y_test, 

                       y_pred_lreg_pi))

print('__'*30)

print('\nClassification Metrics')

print('__'*30)

print(classification_report(y_test, 

                            y_pred_lreg_pi))

print('__'*30)

logreg_accuracy = round(accuracy_score(y_test, 

                                       y_pred_lreg_pi) * 100,2)

print('Accuracy', logreg_accuracy,'%')
fpr_lreg_pi, tpr_lreg_pi, thresholds_pi = roc_curve(y_test, 

                                                   y_pred_lreg_pi)

roc_auc_lreg_pi = auc(fpr_lreg_pi,

                       tpr_lreg_pi)

plt.title('Receiver Operating Characteristic - LReg (All features + RF Features + PI features)')

plt.plot(fpr_lreg_rf, 

         tpr_lreg_rf, 

         'g',

         label='AUC = %0.3f'% roc_auc_lreg_rf)

plt.plot(fpr_lreg_all, 

         tpr_lreg_all, 

         'b',

         label='AUC = %0.3f'% roc_auc_lreg_all)

plt.plot(fpr_lreg_pi, 

         tpr_lreg_pi, 

         'y',

         label='AUC = %0.3f'% roc_auc_lreg_pi)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
rf_model_all = rf.fit(X_train_res, y_train_res)

y_pred_rf_all = rf_model_all.predict(X_test.as_matrix())



print('Confusion Matrix')

print('__'*10)

print(confusion_matrix(y_test, 

                       y_pred_rf_all))

print('__'*30)

print('\nClassification Metrics')

print('__'*30)

print(classification_report(y_test, 

                            y_pred_rf_all))

print('__'*30)

logreg_accuracy = round(accuracy_score(y_test, 

                                       y_pred_rf_all) * 100,2)

print('Accuracy', logreg_accuracy,'%')
fpr_rf_all, tpr_rf_all, thresholds = roc_curve(y_test, 

                                                   y_pred_rf_all)

roc_auc_rf_all = auc(fpr_rf_all,

                       tpr_rf_all)

plt.title('Receiver Operating Characteristic - RF (All features)')

plt.plot(fpr_rf_all, 

         tpr_rf_all, 

         'b',

         label='AUC = %0.3f'% roc_auc_rf_all)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
X_rf = X[rf_features]

X_train, X_test, y_train, y_test = train_test_split(X_rf, 

                                                    y, 

                                                    test_size=0.3, 

                                                    random_state=101)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

rf_model_rf = rf.fit(X_train_res, y_train_res)

y_pred_rf_rf = rf_model_rf.predict(X_test.as_matrix())



print('Confusion Matrix')

print('__'*10)

print(confusion_matrix(y_test, 

                       y_pred_rf_rf))

print('__'*30)

print('\nClassification Metrics')

print('__'*30)

print(classification_report(y_test, 

                            y_pred_rf_rf))

print('__'*30)

logreg_accuracy = round(accuracy_score(y_test, 

                                       y_pred_rf_rf) * 100,2)

print('Accuracy', logreg_accuracy,'%')
fpr_rf_rf, tpr_rf_rf, thresholds_rf = roc_curve(y_test, 

                                                   y_pred_rf_rf)

roc_auc_rf_rf = auc(fpr_rf_rf,

                       tpr_rf_rf)

plt.title('Receiver Operating Characteristic - RF (All features + RF features)')

plt.plot(fpr_rf_rf, 

         tpr_rf_rf, 

         'g',

         label='AUC = %0.3f'% roc_auc_rf_rf)

plt.plot(fpr_rf_all, 

         tpr_rf_all, 

         'b',

         label='AUC = %0.3f'% roc_auc_rf_all)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
X_pf = X[pi_features]

X_train, X_test, y_train, y_test = train_test_split(X_pf, 

                                                    y, 

                                                    test_size=0.3, 

                                                    random_state=101)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

rf_model_pi = rf.fit(X_train_res, y_train_res)

y_pred_rf_pi = rf_model_pi.predict(X_test.as_matrix())



print('Confusion Matrix')

print('__'*10)

print(confusion_matrix(y_test, 

                       y_pred_rf_pi))

print('__'*30)

print('\nClassification Metrics')

print('__'*30)

print(classification_report(y_test, 

                            y_pred_rf_pi))

print('__'*30)

logreg_accuracy = round(accuracy_score(y_test, 

                                       y_pred_rf_pi) * 100,2)

print('Accuracy', logreg_accuracy,'%')
fpr_rf_pi, tpr_rf_pi, thresholds_pi = roc_curve(y_test, 

                                                   y_pred_rf_pi)

roc_auc_rf_pi = auc(fpr_rf_pi,

                       tpr_rf_pi)

plt.title('Receiver Operating Characteristic - RF (All features + RF features + PI features)')

plt.plot(fpr_rf_rf, 

         tpr_rf_rf, 

         'g',

         label='AUC = %0.3f'% roc_auc_rf_rf)

plt.plot(fpr_rf_all, 

         tpr_rf_all, 

         'b',

         label='AUC = %0.3f'% roc_auc_rf_all)

plt.plot(fpr_rf_pi, 

         tpr_rf_pi, 

         'y',

         label='AUC = %0.3f'% roc_auc_rf_pi)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
pdp_V4 = pdp.pdp_isolate(model = rf_model_pi,

                         dataset = X_test, 

                         model_features = pi_features, 

                         feature='V4', 

                         num_grid_points = 20)



pdp.pdp_plot(pdp_V4, 

             'V4', 

             plot_pts_dist=True)

plt.show()
pdp_V14 = pdp.pdp_isolate(model = rf_model_pi,

                         dataset = X_test, 

                         model_features = pi_features, 

                         feature='V14', 

                         num_grid_points = 20)



pdp.pdp_plot(pdp_V14, 

             'V14', 

             plot_pts_dist=True)

plt.show()
explainer = shap.TreeExplainer(rf_model_pi)
data_row = 443

data_row = X_test.iloc[data_row]

print("Predicted probability\nNon-fraud - Fraud\n", rf_model_pi.predict_proba(data_row.values.reshape(1, -1)))

shap_values = explainer.shap_values(data_row)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_row)
data_row = 442

data_row = X_test.iloc[data_row]

print("Predicted probability\nNon-fraud - Fraud\n", rf_model_pi.predict_proba(data_row.values.reshape(1, -1)))

shap_values = explainer.shap_values(data_row)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_row)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values[1], X_test)
shap.dependence_plot('V16', shap_values[1], X_test, interaction_index = 'V4')