import pandas as pd

import numpy as np

import time

import matplotlib.pyplot as plt

import seaborn as sns

import graphviz



from imblearn.over_sampling import RandomOverSampler

from imblearn.pipeline import Pipeline



from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import GridSearchCV, train_test_split



from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet

import lightgbm as lgb

import xgboost as xgb

import catboost as cat



from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier



from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

from sklearn.tree import export_graphviz

import warnings 

warnings.filterwarnings('ignore')
def plot_roc_curve(fprs, tprs):

    

    tprs_interp = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)

    f, ax = plt.subplots(figsize=(8, 8))

    

    # Plotting ROC for each fold and computing AUC scores

    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):

        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))

        tprs_interp[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))

        

    # Plotting ROC for random guessing

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')

    

    mean_tpr = np.mean(tprs_interp, axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)

    

    # Plotting the mean ROC

    ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc), lw=2, alpha=0.8)

    

    # Plotting the standard deviation around the mean ROC Curve

    std_tpr = np.std(tprs_interp, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')

    

    ax.set_xlabel('False Positive Rate', size=15, labelpad=20)

    ax.set_ylabel('True Positive Rate', size=15, labelpad=20)

    ax.tick_params(axis='x', labelsize=15)

    ax.tick_params(axis='y', labelsize=15)

    ax.set_xlim([-0.05, 1.05])

    ax.set_ylim([-0.05, 1.05])



    ax.set_title('ROC Curves of Folds', size=20, y=1.02)

    ax.legend(loc='lower right', prop={'size': 13})

    

    plt.show()
def plot_feature_importances(feature_importances, title, feature_names):

    feature_importances = 100.0*(feature_importances/max(feature_importances))

    index_sorted = np.flipud(np.argsort(feature_importances))

    pos = np.arange(index_sorted.shape[0])+0.5

    

    fig, ax = plt.subplots(figsize=(16,4))

    plt.bar(pos,feature_importances[index_sorted])

    for tick in ax.get_xticklabels():

        tick.set_rotation(90)

    plt.xticks(pos,feature_names[index_sorted])

    plt.ylabel('Relative Importance')

    plt.title(title)

    plt.show() 
df_train = pd.read_csv('/kaggle/input/rs6-attrition-predict/train.csv')

df_test = pd.read_csv('/kaggle/input/rs6-attrition-predict/test.csv')
def extract_features(df, is_train=False):

    # target

    y = pd.DataFrame()

    if is_train:

        attrition_dict = {'No':0,'Yes':1}

        df['Attrition'] = df['Attrition'].map(lambda x: attrition_dict[x])

        y = df.Attrition

        df.drop(['Attrition'], axis=1, inplace=True)

    else:

        y = df.user_id

    df.drop(['user_id'], inplace=True, axis=1)

    # BusinessTravel

    businesstravel_dict = {'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently':2}

    df['BusinessTravel'] = df['BusinessTravel'].map(lambda x: businesstravel_dict[x])

    # Department

    department_dict = {'Sales':0, 'Research & Development':1, 'Human Resources':2}

    df['Department'] = df['Department'].map(lambda x: department_dict[x])

    # EducationField

    educationfield_dict = {'Life Sciences':0, 'Medical':1, 'Marketing':2, 'Technical Degree':3, 'Human Resources':4, 'Other':5}

    df['EducationField'] = df['EducationField'].map(lambda x: educationfield_dict[x])

    # Gender

    gender_dict = {'Male':0, 'Female': 1}

    df['Gender'] = df['Gender'].map(lambda x: gender_dict[x])

    # JobRole

    jobrole_dict = {'Sales Executive':0, 

                    'Research Scientist':1, 

                    'Laboratory Technician':2, 

                    'Manufacturing Director':3, 

                    'Healthcare Representative':4,

                    'Manager':5, 

                    'Sales Representative':6,

                    'Research Director':7,

                    'Human Resources':8

                   }

    df['JobRole'] = df['JobRole'].map(lambda x: jobrole_dict[x])

    # MaritalStatus

    maritalstatus_dict = {'Single':0, 'Married':1, 'Divorced':2}

    df['MaritalStatus'] = df['MaritalStatus'].map(lambda x: maritalstatus_dict[x])

    # Over18

    df.drop(['Over18'], inplace=True, axis=1)

    # EmployeeNumber

    df.drop(['EmployeeNumber'], inplace=True, axis=1)

    # OverTime

    overtime_dict = {'Yes':0, 'No':1}

    df['OverTime'] = df['OverTime'].map(lambda x: overtime_dict[x])

    return y, df
target, train = extract_features(df_train, True)

user_id, test = extract_features(df_test, False)

del df_train

del df_test
def get_optimizer_params(model, model_params, train=train, test=test):

    gridsearch = GridSearchCV(model, model_params, scoring='roc_auc', cv=5)

    gridsearch.fit(train, target)

    best_score = gridsearch.best_score_

    print("Best score: %0.3f" % best_score)

    print("Best parameters set:")

    best_parameters = gridsearch.best_estimator_.get_params()

    for param_name in sorted(gridsearch.param_grid.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))

#     print(best_parameters)

    return best_parameters



def get_model_result(model, model_params, model_name, scaler=None, test_size=0.5, train=train, test=test, feature_importance=False, gridsearch=True):

    if scaler is not None:

        train = scaler.fit_transform(train)

        test = scaler.fit_transform(test)



    if gridsearch is True:

        best_params = get_optimizer_params(model, model_params)

        model.set_params(**best_params)

    X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=test_size, random_state=2020, stratify=target)

    model.fit(X_train, y_train)

    prob_y_val = model.predict_proba(X_val)[:,1] if hasattr(model, 'predict_proba') else model.predict(X_val)

    trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_val, prob_y_val)

    best_score = auc(trn_fpr, trn_tpr)

    best_parameters = model.get_params()

    plot_roc_curve([list(trn_fpr)], [list(trn_tpr)])

        

        

    if feature_importance is True:

        plot_feature_importances(model.feature_importances_, 'Importance of Features', train.columns)

    result = pd.DataFrame()

    result['user_id'] = user_id

    result['Attrition'] = pd.DataFrame(model.predict_proba(test)[:,1] if hasattr(model, 'predict_proba') else model.predict(test))

    result[['user_id', 'Attrition']].to_csv(f'result-{model_name}.csv', index=False, float_format='%.8f')

    print('result of predict:\n', result.head())

    return best_score, best_parameters
data = pd.concat([train, test]).corr() ** 2

data = np.tril(data, k=-1)

data[data==0] = np.nan
figure, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(np.sqrt(data), annot=False, cmap='viridis', ax=ax)
data = train.corrwith(target).agg('square')



figure, ax = plt.subplots(figsize=(10, 10))

data.agg('sqrt').plot.bar(ax=ax)

del data
model = Ridge()

model_params = {'alpha': [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 7], 'tol': [0.0001, 0.001, 0.01, 0.1]}

model_params = {'alpha': [2], 'tol': [0.0001]}

start = time.time()

best_score_ridge, best_params_ridge = get_model_result(model, model_params, 'ridge', StandardScaler())

print(time.time()-start)

del model
model = Lasso()

model_params = {'alpha': np.logspace(-10, -6, 50)}

model_params = {'alpha': [1e-10]}

start = time.time()

best_score_lasso, best_params_lasso = get_model_result(model, model_params, 'lasso', StandardScaler())

print(time.time() - start)

del model
model = ElasticNet()

model_params = {'alpha': np.logspace(-10, -4, 10), 'l1_ratio': np.logspace(-10, -4, 10)}

model_params = {'alpha': [0.0001], 'l1_ratio': [1e-10]}

start = time.time()

best_score_elasticnet, best_params_elasticnet = get_model_result(model, model_params, 'elasticnet', StandardScaler())

print(time.time() - start)

del model 
model = LogisticRegression(class_weight='balanced')

model_params = {

    'penalty': ['l1', 'l2'], 

    'C': [0.4, 0.5, 0.6],

    'solver':['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']

}

model_params = {

    'penalty': ['l1'], 

    'C': [0.5],

    'solver':['liblinear']

}

start = time.time()

best_score_lgr,best_params_lgr = get_model_result(model, model_params,'logisticregression', StandardScaler()) 

print(time.time() - start)

del model 
def sigmoid(x):

    return 1/(1+np.exp(-x))

fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4))

z = np.linspace(-3,3,num=100)

ax[0].plot(z,-np.log(sigmoid(z)),label='logistic',lw=2, color='b')

ax[1].plot(z,-np.log(1-sigmoid(z)),label='logistic',lw=2,color='b')

ax[0].plot(z,np.maximum(0,1-z),label='SVM',lw=2, color='r',linestyle='--')

ax[1].plot(z,np.maximum(0,1+z),label='SVM',lw=2,color='r',linestyle='--')

ax[0].set_title('y=1')

ax[1].set_title('y=0')

ax[0].set_xlabel('z')

ax[1].set_xlabel('z')

ax[0].set_ylabel('individual loss')

ax[1].set_ylabel('individual loss')

ax[0].legend()

ax[1].legend()

plt.show()
x = np.linspace(-4,4,num=100)

l = 0

gamma1=0.5

f1 = np.exp(-gamma1*(x-l)*(x-l))

gamma2=5

f2 = np.exp(-gamma2*(x-l)*(x-l))

plt.plot(x,f1,label=r'$\gamma = 0.5$')

plt.plot(x,f2,label=r'$\gamma = 5$')

plt.legend(fontsize = 14)

plt.xlabel('x',fontsize = 14)

plt.ylabel('similarity', fontsize = 14)

plt.arrow(0,0.2,0,-0.18, head_width=0.2, head_length=0.05,lw=1,color='indianred')

plt.text(-0.7,0.22,'landmark', color='indianred', fontsize=14)

plt.show()
# model = SVC()

# model_params = {

#     'C':[0.1,1,5,10,50,100, 200],

#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

#     'gamma':[1,0.1,0.01,0.001]

# }

# best_score_svc, best_params_svc = get_model_result(model, model_params, 'SVC', StandardScaler())
tree = DecisionTreeClassifier(max_depth=2, random_state=2020)

tree.fit(train, target)

dot_data = export_graphviz(tree,

                out_file=None,

                feature_names=train.columns,

                class_names=['Yes', 'No'],

                rounded=True,

                filled=True)



graph = graphviz.Source(dot_data)

graph.render() 

graph
model = DecisionTreeClassifier(random_state=2020)

model_params = {

    'max_features': [0.8, 1.0], 

    'max_depth': [8, 9, 10],

    'min_samples_leaf':[30, 40, 50]

} 

model_params = {

    'max_features': [1.0], 

    'max_depth': [9],

    'min_samples_leaf':[30]

}

start = time.time()

best_score_dtc, best_params_dtc = get_model_result(model, model_params, 'dtc', None, feature_importance=True)

print(time.time() - start)

del model  
model = RandomForestClassifier(random_state=2020)

model_params = {

    'n_estimators':[500],

    'n_jobs':[-1],

    'max_features': [0.5, 0.6], 

    'max_depth': [8, 9, 10],

    'min_samples_leaf':[8, 10],

#     'random_state':[2020]

}

model_params = {

    'n_estimators':[200],

    'max_features': [0.6], 

    'max_depth': [10],

    'min_samples_leaf':[10],

}

start = time.time()

best_score_rfc, best_params_rfc = get_model_result(model, model_params, 'rfc', feature_importance=True)

print(time.time() - start)

del model 
# rfc = RandomForestClassifier(best_params_rfc)

# svm = SVC(best_params_svc)

# lr = LogisticRegression(best_params_logisticregression)

# best_score_voting, best_params_voting = get_model_result(VotingClassifier(estimators = [('rf',rfc), ('svm',svm), ('log', lr)], voting='soft'), 'voting', StandardScaler(), gridsearch=False)
best_score_bagging, best_params_bagging = get_model_result(BaggingClassifier(LogisticRegression(**best_params_lgr),n_estimators=500, random_state=2020), 'bagging', StandardScaler(), gridsearch=False)

best_score_pasting, best_params_pasting = get_model_result(BaggingClassifier(LogisticRegression(**best_params_lgr),n_estimators=500, bootstrap_features=True, max_features=1.0, random_state=2020), 'pasting', StandardScaler(), gridsearch=False)
model = ExtraTreesClassifier(random_state=2020)

model_params = {

    'n_estimators':[500],

    'n_jobs':[-1], 

    'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 

    'max_depth': [10,11,12,13,14],

    'min_samples_leaf':[1,10,100],

#     'random_state':[0]

} 

model_params = {

    'n_estimators':[500],

    'max_features': [0.5], 

    'max_depth': [12],

    'min_samples_leaf':[10],  

}

start = time.time()

best_score_etf, best_params_etf = get_model_result(model, model_params, 'etc',feature_importance=True)

print(time.time() - start)

del model
model = xgb.XGBClassifier(random_state=2020,tree_method='gpu_hist', silent=1, booster='gbtree', objective='binary:logistic')

model_params = {

    'booster':['gbtree'],

    'colsample_bytree': [0.5, 0.8],

    'subsample': [0,3, 0.5],

    'learning_rate': [0.075, 0.01],

    'objective': ['binary:logistic'],

    'max_depth': [ 7, 8, 9],

    'num_parallel_tree': [0.1, 1, 10],

    'min_child_weight': [0.2, 0.8],

}

model_params = {

    'colsample_bytree': [0.5],

    'subsample': [0.5],

    'learning_rate': [0.075],

    'max_depth': [9],

    'num_parallel_tree': [1],

    'min_child_weight': [0.2],

}



start = time.time()

best_score_xgboost, best_params_xgboost = get_model_result(model, model_params, 'xgboost')

print(time.time()-start)

del model  
model = lgb.LGBMClassifier(random_state=2020, device='gpu', gpu_platform_id=0, gpu_device_id=0, silent=1)

model_params = {

    'n_estimators': [200, 300, 400],

    'learning_rate': [0.01, 0.1, 0.5],

    'num_leaves':[10,100,400],

    'colsample_bytree':[0.5,0.8, 1.0],

    'subsample':[0.3,0.5,0.9],

    'max_depth':[7, 10, 15],

    'reg_alpha':[0.01, 0.2, 0.5],

    'reg_lambda':[0.01, 0.3, 0.8],

    'min_split_gain':[0.01, 0.1],

    'min_child_weight': [1,2,4],

}

model_params = {

    'n_estimators': [600],

    'learning_rate': [0.1],

    'num_leaves':[120],

    'colsample_bytree':[0.5],

    'subsample':[0.9],

    'max_depth':[15],

    'reg_alpha':[0.01, 0.2],

    'reg_lambda':[0.4],

    'min_split_gain':[0.01],

}

start = time.time()

best_score_lightgbm, best_params_lightgbm = get_model_result(model, model_params, 'lightgbm')

print(time.time()-start)

del model 


model = cat.CatBoostClassifier(random_state=2020, task_type='GPU', allow_writing_files=False, silent=True, eval_metric='AUC', bootstrap_type='Bernoulli')

model_params = {

    'iterations': [200,600],

    'learning_rate': [0.05, 0.3, 0.5],

    'depth': [6, 7, 9, 10],

    'l2_leaf_reg': [30, 40, 50],

    'bootstrap_type': ['Bernoulli'],

    'subsample': [0.5, 0.7, 1.0],

    'scale_pos_weight': [4, 5, 10],

}

model_params = {

    'iterations': [600],

    'learning_rate': [ 0.5],

    'depth': [10],

    'l2_leaf_reg': [40],

    'subsample': [0.5],

    'scale_pos_weight': [8],

}

start = time.time()

best_score_catboost, best_params_catboost = get_model_result(model, model_params, 'catboost')

print(time.time() - start)

del model 
auc_value = [best_score_ridge, best_score_elasticnet, best_score_lgr, best_score_dtc, best_score_rfc, best_score_etf, best_score_xgboost, best_score_lightgbm, best_score_catboost, best_score_bagging, best_score_pasting]

auc_label = ['ridge', 'elasticnet', 'lgr', 'dtc', 'rfc', 'etf', 'xgboost', 'lightgbm', 'catboost', 'bagging', 'pasting']

# auc_time = [best_score_ridge, best_score_elasticnet, best_score_lgr, best_score_dtc, best_score_rfc, best_score_etf, best_score_xgboost, best_score_lightgbm, best_score_catboost]

figure, ax = plt.subplots(figsize=(16,4))



plt.bar(range(len(auc_value)), auc_value, tick_label=auc_label)

for tick in ax.get_xticklabels():

    tick.set_rotation(90)

plt.title('Different AUC in Dataset by ML')

plt.show()