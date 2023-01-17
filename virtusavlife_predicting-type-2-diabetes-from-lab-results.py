import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE,ADASYN
import lightgbm as lgm
from boruta import BorutaPy
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,auc,log_loss
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as se
from sklearn.utils import shuffle
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
medications = pd.read_csv('medications.csv',encoding='unicode_escape')
labs = pd.read_csv('labs.csv',encoding='unicode_escape')
medications.info()
labs.info()
medications.head()
labs.head()
medications.sort_values('RXDDAYS',ascending=False,inplace=True)
medications.reset_index(inplace=True,drop=True)
medications = medications.drop_duplicates('SEQN',keep='first')
len(medications)
medications.sort_values('SEQN',inplace=True)
medications.reset_index(inplace=True,drop=True)
medications['is_diabetic'] = np.where(medications['RXDRSC1'].str.contains('E11',case=False,na=False),'1','0')
medications = medications.iloc[:,[0,13]]
diabetes = pd.merge(labs,medications,how='left',on='SEQN')
diabetes.info()
diabetes.head()
diabetes.isna().sum()
diabetes.isna().mean()
diabetes = diabetes.loc[:,diabetes.isna().mean() <= .5]
len(diabetes.columns)
diabetes.head()
diabetes = diabetes.fillna(diabetes.mean())
diabetes.head()
x = diabetes.iloc[:,:-1]
y = diabetes.iloc[:,[-1]]
fields_ranking = pd.DataFrame(x.columns.tolist(), columns=['features'])
random_forest_classifier = RandomForestClassifier()
boruta = BorutaPy(random_forest_classifier,n_estimators='auto')
boruta.fit(x.values,y.values)
fields_ranking['rank'] = boruta.ranking_
fields_ranking.sort_values('rank',ascending=True,inplace=True)
fields_ranking.reset_index(inplace=True,drop=True)
print(boruta.n_features_)
fields_ranking.head(15)
y['is_diabetic'] = y['is_diabetic'].astype(int)
data = lgm.Dataset(x,label=y)
params = {"max_depth":15, "learning_rate":0.1, "num_leaves":900, "n_estimators":100}
lgm_model = lgm.train(params=params, train_set=data, categorical_feature='auto')
lgm_plot = lgm.plot_importance(lgm_model, max_num_features=152, figsize=(50,30))
plt.show()
this = list(lgm_model.feature_importance())
effe_cols = []
for i in this:
    if i > 110:
        effe_cols.append(lgm_model.feature_name()[this.index(i)])
        this[this.index(i)]=0
diabetes = diabetes[effe_cols]
diabetes['is_diabetic'] = y['is_diabetic']
del diabetes['SEQN']
diabetes.head()
glycohemoglobin_level_t2d = pd.DataFrame(diabetes[diabetes['is_diabetic']==1].iloc[:,10],columns=['LBXGH'])
glycohemoglobin_level_t2d.sort_values('LBXGH',ascending=False,inplace=True)
glycohemoglobin_level_t2d.reset_index(inplace=True,drop=True)
glycohemoglobin_level_t2d = pd.DataFrame([glycohemoglobin_level_t2d['LBXGH'].mean()],columns=['LBXGH_Level'])
glycohemoglobin_level_normal = pd.DataFrame(diabetes[diabetes['is_diabetic']==0].iloc[:,10],columns=['LBXGH'])
glycohemoglobin_level_normal.sort_values('LBXGH',ascending=False,inplace=True)
glycohemoglobin_level_normal.reset_index(inplace=True,drop=True)
glycohemoglobin_level_normal = pd.DataFrame([glycohemoglobin_level_normal['LBXGH'].mean()],columns=['LBXGH_Level'])
f, (ax1,ax2) = plt.subplots(1, 2)
glycohemoglobin_t2d_plot = se.barplot(y=glycohemoglobin_level_t2d,ax=ax1)
glycohemoglobin_normal_plot = se.barplot(y=glycohemoglobin_level_normal,ax=ax2)
ax1.set_ylabel('LBXGH Level')
ax1.set_xlabel('Patient having Type2Diabetes')
ax2.set_ylabel('LBXGH Level')
ax2.set_xlabel('Patient does not having Type2Diabetes')
f.suptitle('Average Glycohemoglobin level')
f.subplots_adjust(wspace=0.8)
se.despine(bottom=True)
glucose_refrigerated_level_t2d = pd.DataFrame(diabetes[diabetes['is_diabetic']==1].iloc[:,4],columns=['LBXSGL'])
glucose_refrigerated_level_t2d.sort_values('LBXSGL',ascending=False,inplace=True)
glucose_refrigerated_level_t2d.reset_index(inplace=True,drop=True)
glucose_refrigerated_level_t2d = pd.DataFrame([glucose_refrigerated_level_t2d['LBXSGL'].mean()],columns=['LBXSGL_Level'])
glucose_refrigerated_level_normal = pd.DataFrame(diabetes[diabetes['is_diabetic']==0].iloc[:,4],columns=['LBXSGL'])
glucose_refrigerated_level_normal.sort_values('LBXSGL',ascending=False,inplace=True)
glucose_refrigerated_level_normal.reset_index(inplace=True,drop=True)
glucose_refrigerated_level_normal = pd.DataFrame([glucose_refrigerated_level_normal['LBXSGL'].mean()],columns=['LBXSGL_Level'])
f, (ax3,ax4) = plt.subplots(1, 2)
glucose_refrigerated_t2d_plot = se.barplot(y=glucose_refrigerated_level_t2d,ax=ax3)
glucose_refrigerated_normal_plot = se.barplot(y=glucose_refrigerated_level_normal,ax=ax4)
ax3.set_ylabel('LBXSGL Level')
ax3.set_xlabel('Patient having Type2Diabetes')
ax4.set_ylabel('LBXSGL Level')
ax4.set_xlabel('Patient does not having Type2Diabetes')
f.suptitle('Average Glucose Refrigerated Serum level')
f.subplots_adjust(wspace=0.8)
se.despine(bottom=True)
pairplot_t2d = se.pairplot(diabetes.iloc[:,[0,2,4,10,14]], hue='is_diabetic', diag_kind='kde', plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'}, size=4)
diabetes.plot(kind='density', subplots=True, layout=(4,4), sharex=False,figsize=(30,10))
plt.show()
diabetes['is_diabetic'] = diabetes['is_diabetic'].astype(int)
feature_corr = diabetes.corr()
plt.subplots(figsize=(10,5))
feature_map = se.heatmap(feature_corr, cmap='Accent', center=0, vmax=.3, square=True)
plt.show()
diabetes.is_diabetic.value_counts()
x = diabetes.iloc[:,:-1]
y = diabetes.iloc[:,[-1]]
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train = x_train.values
y_train = y_train.values
synthetic_features_SMOTE = SMOTE(random_state=0,ratio={0:9592,1:1000})
X_SMOTE, Y_SMOTE = synthetic_features_SMOTE.fit_sample(x_train,y_train.ravel())
print(np.unique(Y_SMOTE,return_counts=True))
print("Length of X is", len(X_SMOTE))
print("Length of Y is", len(Y_SMOTE))
synthetic_features_ADASYN = ADASYN(random_state=0, ratio={0:9592,1:1200})
X_ADASYN, Y_ADASYN = synthetic_features_ADASYN.fit_sample(x_train,y_train.ravel())
print(np.unique(Y_ADASYN,return_counts=True))
print("Length of X is", len(X_ADASYN))
print("Length of Y is", len(Y_ADASYN))
lr = LogisticRegression()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
xgbc = XGBClassifier()
model_list = []
model_list.append(('LR',lr))
model_list.append(('DTC',dtc))
model_list.append(('RFC',rfc))
model_list.append(('ABC',abc))
model_list.append(('GBC',gbc))
model_list.append(('xgbc',xgbc))
modelname = []
modelaccuracy = []
model_roc_auc_score = []
for model_name,select_model in model_list:
    fit_model = select_model.fit(X_ADASYN,Y_ADASYN)
    model_predict = fit_model.predict(x_test.values)
    model_accuracy = accuracy_score(y_test,model_predict)
    roc_auc_score_model = roc_auc_score(y_test,model_predict)
    model_msg = "%s %f %f" % (model_name, model_accuracy, roc_auc_score_model)
    modelname.append(model_name)
    modelaccuracy.append(model_accuracy)
    model_roc_auc_score.append(roc_auc_score_model)
    print(model_msg)
bar_plot = se.barplot(x=modelname,y=modelaccuracy)
bar_plot.set_ylim(0,1)
bar_plot.set_ylabel("Accuracy")
bar_plot.set_xlabel("Model")
bar_plot.set_title("Accuracy of a model")
models = []
models.append(('LR',LogisticRegression()))
models.append(('DTC',DecisionTreeClassifier()))
models.append(('RFC',RandomForestClassifier()))
models.append(('ABC',AdaBoostClassifier()))
models.append(('GBC',GradientBoostingClassifier()))
models.append(('xgbc',XGBClassifier()))
model_results = []
model_name = []
model_score = []
for name, model in models:
    kfold_validation = model_selection.KFold(n_splits=10, random_state=0)
    cv_results = model_selection.cross_val_score(model, X_ADASYN, Y_ADASYN.ravel(), cv=kfold_validation, scoring='accuracy')
    model_results.append(cv_results)
    model_name.append(name)
    model_score.append(cv_results.mean())
    msg = "%s %f %f" % (name, cv_results.mean(), cv_results.std())
    print(msg)
boxplot_cv = se.boxplot(x=model_name, y=model_results)
boxplot_cv.set_ylim(0.5,1)
rfc_tunning = RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(start=10, stop=110, num=11)]
max_depth.append(None)
min_samples_split = [2 ,5 ,10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True,False]
random_grid = {'n_estimators':n_estimators, 'max_features':max_features, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf, 'bootstrap':bootstrap}
randomsearch = RandomizedSearchCV(estimator=rfc_tunning, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
randomsearch.fit(X_ADASYN, Y_ADASYN)
randomsearch.best_params_
param_grid = {
    'bootstrap': [False],
    'max_depth': [30, 40, 50, 60],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [1000, 1200, 1600, 2000]
}
gridsearch_tunning = GridSearchCV(estimator=rfc_tunning, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
gridsearch_tunning.fit(X_ADASYN, Y_ADASYN)
gridsearch_tunning.best_params_
rfc_model = RandomForestClassifier(n_estimators=1000, max_depth=50, max_features='auto', bootstrap=False, min_samples_leaf=1, min_samples_split=2)
rfc_fit = rfc_model.fit(X_ADASYN, Y_ADASYN)
rfc_predict = rfc_fit.predict(x_test)
rfc_accuracy = accuracy_score(y_test,rfc_predict)
rfc_roc_score = roc_auc_score(y_test,rfc_predict)
rfc_predict_train_data = rfc_fit.predict(x_train)
rfc_accuracy_train_data = accuracy_score(y_train,rfc_predict_train_data)
rfc_roc_score_train_data = roc_auc_score(y_train, rfc_predict_train_data)
print("Accuracy on test data is %f" % (rfc_accuracy))
print("ROC_AUC Score for test data is %f" % (rfc_roc_score))
print("Accuracy on train data is %f" % (rfc_accuracy_train_data))
print("ROC_AUC Score for train data is %f" % (rfc_roc_score_train_data))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,rfc_predict)
roc_auc = auc(false_positive_rate,true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
cm = confusion_matrix(y_test, rfc_predict)
cm
abc_tuning = AdaBoostClassifier()
param_grid = {
    'n_estimators': [5,10],
    'learning_rate': [0.01,0.1,1]
}
randomsearch = RandomizedSearchCV(estimator=abc_tuning, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
randomsearch.fit(X_ADASYN, Y_ADASYN)
randomsearch.best_params_
gridsearch_tunning = GridSearchCV(estimator=abc_tuning, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
gridsearch_tunning.fit(X_ADASYN, Y_ADASYN)
gridsearch_tunning.best_params_
dc_cl = DecisionTreeClassifier(max_depth=10, max_features=2,max_leaf_nodes=2,min_samples_leaf=1)
adboost_model = AdaBoostClassifier(learning_rate=1,n_estimators=5,base_estimator=dc_cl,random_state=0)
ad_fit = adboost_model.fit(X_ADASYN, Y_ADASYN)
ad_predict = ad_fit.predict(x_test)
ad_accuracy = accuracy_score(y_test,ad_predict)
ad_roc_auc = roc_auc_score(y_test,ad_predict)
ad_predict_train = ad_fit.predict(x_train)
ad_accuracy_train = accuracy_score(y_train,ad_predict_train)
ad_roc_auc_train = roc_auc_score(y_train,ad_predict_train)
print("Accuracy on test data is %f" % (ad_accuracy))
print("ROC_AUC Score for test data is %f" % (ad_roc_auc))
print("Accuracy on train data is %f" % (ad_accuracy_train))
print("ROC_AUC Score for train data is %f" % (ad_roc_auc_train))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,ad_predict)
roc_auc = auc(false_positive_rate,true_positive_rate)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
cm =confusion_matrix(y_test,ad_predict)
cm
max_depth = [int(x) for x in np.linspace(start=1, stop=32)]
max_features = list(range(1,diabetes.shape[1]))
min_samples_leaf = [1,2,3,4]
min_samples_split = [2,5,10]
param_grid_data = {
    'max_depth':max_depth,
    'max_features':max_features,
    'min_samples_leaf':min_samples_leaf,
    'min_samples_split':min_samples_split
}
gridsearch_tunning = GridSearchCV(estimator=dtc_tunning, param_grid=param_grid_data, cv=3, n_jobs=-1, verbose=2)
gridsearch_tunning.fit(X_ADASYN, Y_ADASYN)
gridsearch_tunning.best_params_
decision_model = DecisionTreeClassifier(max_depth=30,max_features='auto',min_samples_leaf=1,min_samples_split=2,random_state=0,)
decision_fit = decision_model.fit(X_ADASYN, Y_ADASYN)
decision_predict = decision_fit.predict(x_test)
decision_accuracy = accuracy_score(y_test,decision_predict)
decision_roc_auc = roc_auc_score(y_test,decision_predict)
decision_predict_train = decision_fit.predict(x_train)
decision_accuracy_train = accuracy_score(y_train,decision_predict_train)
decision_roc_auc_train = roc_auc_score(y_train,decision_predict_train)
print("Accuracy on test data is %f" % (decision_accuracy))
print("ROC_AUC Score for test data is %f" % (decision_roc_auc))
print("Accuracy on train data is %f" % (decision_accuracy_train))
print("ROC_AUC Score for train data is %f" % (decision_roc_auc_train))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,decision_predict)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.title("Receiver Operating Characteristic")
plt.plot(false_positive_rate, true_positive_rate, 'b' ,label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
cm = confusion_matrix(y_test,decision_predict)
cm
model_path = os.path.join(os.path.pardir,'Type2Diabetes','adaboost.pkl')
model_pickle = open(model_path,'wb')
pickle.dump(ad_fit,model_pickle)
model_pickle.close()
model_lr = os.path.join(os.path.pardir,'Type2Diabetes','lr.pkl')
model_pickle_lr = open(model_lr,'wb')
pickle.dump(lr,model_pickle_lr)
model_pickle_lr.close()
type2Diabetes = pd.DataFrame()
print("Enter Albumin, urine (ug/mL)")
type2Diabetes.loc[0,'AlbuminUrine'] = float(input())
print("Enter Urinary creatinine (mg/dL)")
type2Diabetes.loc[0,'UrinaryCreatinine'] = float(input())
print("Enter Albumin creatinine ratio (mg/g)")
type2Diabetes.loc[0,'AlbuminCreatinine'] = float(input())
print("Enter Creatinine (mg/dL)")
type2Diabetes.loc[0,'Creatinine'] = float(input())
print("Glucose, refrigerated serum (mg/dL)")
type2Diabetes.loc[0,'Glucose'] = float(input())
print("Iron, refrigerated serum (ug/dL)")
type2Diabetes.loc[0,'Iron'] = float(input())
print("Lactate dehydrogenase (U/L)")
type2Diabetes.loc[0,'Lactate'] = float(input())
print("Osmolality (mmol/Kg)")
type2Diabetes.loc[0,'Osmolality'] = float(input())
print("Red cell distribution width (%)")
type2Diabetes.loc[0,'RedCell'] = float(input())
print("Platelet count (1000 cells/uL)")
type2Diabetes.loc[0,'Platelet'] = float(input())
print("Glycohemoglobin (%)")
type2Diabetes.loc[0,'Glycohemoglobin'] = float(input())
print("The volume of urine collection #1 (mL)")
type2Diabetes.loc[0,'UrineCollection'] = float(input())
print("Urine #1 Flow Rate (mL/min)")
type2Diabetes.loc[0,'UrineFlow'] = float(input())
print("Vitamin B12(pg/mL)")
type2Diabetes.loc[0,'VitaminB12'] = float(input())
def total_weight_find(col_values, col_intercept):
    total_weight = 0
    for i in range(0, 14):
        total_weight = total_weight + (col_values[i]*col_intercept[0][i])
    return total_weight
predicted_data = ad_fit.predict(type2Diabetes.values.astype('float'))
if predicted_data==0:
    print("isDiabetic = No")
else:
    print("isDiabetic = Yes")
col_list = type2Diabetes.values.ravel().tolist()
intercept_values = lr.coef_.tolist()
total_weigh_data = total_weight_find(col_list, intercept_values)
w_glyco = ((col_list[10] * intercept_values[0][10]) / total_weigh_data)
w_gluco = ((col_list[4] * intercept_values[0][4]) / total_weigh_data)
w_albumin = ((col_list[0] * intercept_values[0][0]) / total_weigh_data)
w_alcr = ((col_list[2]*intercept_values[0][2]) / total_weigh_data)
w_uc = ((col_list[1]*intercept_values[0][1])/total_weigh_data)
w_cr = ((col_list[3]*intercept_values[0][3])/total_weigh_data)
w_ic = ((col_list[5]*intercept_values[0][5])/total_weigh_data)
w_ld = ((col_list[6]*intercept_values[0][6])/total_weigh_data)
w_os = ((col_list[7]*intercept_values[0][7])/total_weigh_data)
w_rd= ((col_list[8]*intercept_values[0][8])/total_weigh_data)
w_pl = ((col_list[9]*intercept_values[0][9])/total_weigh_data)
w_ucoll = ((col_list[11]*intercept_values[0][11])/total_weigh_data)
w_urflo = ((col_list[12]*intercept_values[0][12])/total_weigh_data)
w_vita = ((col_list[13]*intercept_values[0][13])/total_weigh_data)
total_pv = abs(w_glyco)+abs(w_gluco)+abs(w_alcr)+abs(w_albumin)+abs(w_uc)+abs(w_cr)+abs(w_ic)+abs(w_ld)+abs(w_os)+abs(w_rd)+abs(w_pl)+abs(w_ucoll)+abs(w_urflo)+abs(w_vita)
p_glyco = (w_glyco/total_pv)*100
p_gluco = (w_gluco/total_pv)*100
p_albumin = (w_albumin/total_pv)*100
p_alcr = (w_alcr/total_pv)*100
p_uc = (w_uc/total_pv)*100
p_cr = (w_cr/total_pv)*100
p_ic = (w_ic/total_pv)*100
p_ld = (w_ld/total_pv)*100
p_os = (w_os/total_pv)*100
p_rd = (w_rd/total_pv)*100
p_pl = (w_pl/total_pv)*100
p_ucoll = (w_ucoll/total_pv)*100
p_urflo = (w_urflo/total_pv)*100
p_vita = (w_vita/total_pv)*100
vb = [abs(round(p_albumin,2)),abs(round(p_uc,2)),abs(round(p_alcr,2)),abs(round(p_cr,2)),abs(round(p_gluco,2)),abs(round(p_ic,2)),abs(round(p_ld,2)),abs(round(p_os,2)),abs(round(p_rd,2)),abs(round(p_pl,2)),abs(round(p_glyco,2)),abs(round(p_ucoll,2)),abs(round(p_urflo,2)),abs(round(p_vita,2))]
plt.figure(figsize=(35,10))
dr = se.barplot(y=vb,x=type2Diabetes.columns)
k = 0
for p in dr.patches:
    dr.text(p.get_x()+p.get_width()/2.,
    p.get_height()+0.02,
    '{:1.1f}%'.format(vb[k]),
    ha="center",color='black',fontsize=30)
    k = k+1
plt.savefig('t2dColumnProbability.png')
plt.title("Features Probability")
