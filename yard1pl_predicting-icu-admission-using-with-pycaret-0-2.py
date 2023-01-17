!pip install https://github.com/Yard1/pycaret/archive/one_hot_for_boolean_categories.zip
!pip install --upgrade --no-deps --force-reinstall https://github.com/Yard1/pycaret/archive/one_hot_for_boolean_categories.zip
import pandas as pd
import numpy as np
from pycaret.classification import *

import random as rn
SEED = 440
!PYTHONHASHSEED=0
np.random.seed(SEED)
rn.seed(SEED)
raw_data = pd.read_excel('../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
raw_data
raw_data['AGE_PERCENTIL'] = raw_data['AGE_PERCENTIL'].replace(r'[^0-9]', '', regex=True)
def get_na_table(df):
    na_counts = pd.DataFrame([(x, df[x].isna().sum(), df[x].isna().sum()/len(df)) for x in df.columns])
    na_counts.columns = ['Variable', 'NAs', 'Percentage']
    return na_counts[na_counts['NAs'] > 0].sort_values(by=['NAs'], ascending=False)
get_na_table(raw_data)
# https://www.kaggle.com/fernandoramacciotti/early-icu-detection-only-0-2-window
raw_data = raw_data.sort_values(by=['PATIENT_VISIT_IDENTIFIER', 'WINDOW']).groupby('PATIENT_VISIT_IDENTIFIER', as_index=False).fillna(method='ffill').fillna(method='bfill')

get_na_table(raw_data)
# https://www.kaggle.com/fernandoramacciotti/early-icu-detection-only-0-2-window

# dropping the patients which were admitted to ICU in the first window
data = raw_data.loc[~((raw_data['WINDOW'] == '0-2') & (raw_data['ICU'] == 1))]


# getting the patients that were eventually admitted to ICU
icu_above_2 = data.groupby('PATIENT_VISIT_IDENTIFIER').agg({'ICU': max}).reset_index().rename(columns={'ICU': 'ICU_NEW'})
    
data = data.merge(icu_above_2, on=['PATIENT_VISIT_IDENTIFIER'], how='left')
data = data.loc[data['WINDOW'] == '0-2']
pd.crosstab(data['WINDOW'], data['ICU_NEW'])
modifiers = ['_DIFF', '_MEAN', '_MEAN_REL', '_MIN', '_MAX', '_MEDIAN']
lab_columns = [x for x in data.columns if any(y in x for y in modifiers)]
vital_signs = [x.replace('_MEAN', '') for x in lab_columns if '_MEAN' in x]
vital_signs = {x:[y for y in lab_columns if y.startswith(x)] for x in vital_signs}
for k, v in vital_signs.items():
    display(data[v].corr())
lab_columns_to_ignore = [x for x in lab_columns if not (x.endswith('_DIFF_REL') or x.endswith('_MEDIAN'))]
data['MEAN_AERTIAL_PRESSURE'] = (2*data['BLOODPRESSURE_DIASTOLIC_MEAN'])/3 + (data['BLOODPRESSURE_SISTOLIC_MEAN'])/3
data['BLOOD_PRESSURE_MEDIAN'] = data['BLOODPRESSURE_SISTOLIC_MEDIAN'] - data['BLOODPRESSURE_DIASTOLIC_MEDIAN']
experiment = setup(
    data, 
    target='ICU_NEW',
    ignore_features=['PATIENT_VISIT_IDENTIFIER', 'ICU', 'WINDOW']+lab_columns_to_ignore,
    ordinal_features={'AGE_PERCENTIL': sorted(list(data.AGE_PERCENTIL.unique()))}, # converting AGE_PERCENTIL to an ordinal feature instead of categorical
    fix_imbalance=True, # fixing train-test split imbalances
    feature_selection=True, feature_selection_threshold=0.95, # conservative important feature selection
    remove_perfect_collinearity=True, # in case we missed any perfectly collinear features
    session_id=SEED, # seed for reproductibility
    silent=True # for kaggle compatibility
    )
transformed_data = get_config('X')
transformed_data
display(sorted(transformed_data.columns))
# helper function to get a nice model name
def get_model_name(e) :
    mn = str(e).split("(")[0]

    if 'catboost' in str(e):
        mn = 'CatBoostClassifier'
    
    model_dict_logging = {'ExtraTreesClassifier' : 'Extra Trees Classifier',
                        'GradientBoostingClassifier' : 'Gradient Boosting Classifier', 
                        'RandomForestClassifier' : 'Random Forest Classifier',
                        'LGBMClassifier' : 'Light Gradient Boosting Machine',
                        'XGBClassifier' : 'Extreme Gradient Boosting',
                        'AdaBoostClassifier' : 'Ada Boost Classifier', 
                        'DecisionTreeClassifier' : 'Decision Tree Classifier', 
                        'RidgeClassifier' : 'Ridge Classifier',
                        'LogisticRegression' : 'Logistic Regression',
                        'KNeighborsClassifier' : 'K Neighbors Classifier',
                        'GaussianNB' : 'Naive Bayes',
                        'SGDClassifier' : 'SVM - Linear Kernel',
                        'SVC' : 'SVM - Radial Kernel',
                        'GaussianProcessClassifier' : 'Gaussian Process Classifier',
                        'MLPClassifier' : 'MLP Classifier',
                        'QuadraticDiscriminantAnalysis' : 'Quadratic Discriminant Analysis',
                        'LinearDiscriminantAnalysis' : 'Linear Discriminant Analysis',
                        'CatBoostClassifier' : 'CatBoost Classifier',
                        'BaggingClassifier' : 'Bagging Classifier',
                        'VotingClassifier' : 'Voting Classifier'} 

    return model_dict_logging.get(mn)
models = compare_models(sort='F1', n_select=8)
compare_cv_results = [get_config('display_container')[-1]]
models.append(create_model('rf'))
compare_cv_results.append(get_config('display_container')[-1])
tuned_models_auc_f1 = []
tuned_models_f1_auc = []
tuned_models_auc_f1_cv = []
tuned_models_f1_auc_cv = []
for model in models:
    model_tuned = tune_model(tune_model(model, optimize='AUC', n_iter=60, choose_better=True), optimize='F1', n_iter=20, choose_better = True) 
    tuned_models_auc_f1.append(model_tuned)
    tuned_models_auc_f1_cv.append(get_config('display_container')[-1])
for model in models:
    model_tuned = tune_model(tune_model(model, optimize='F1', n_iter=60, choose_better=True), optimize='AUC', n_iter=20, choose_better = True) 
    tuned_models_f1_auc.append(model_tuned)
    tuned_models_f1_auc_cv.append(get_config('display_container')[-1])
cv_results = compare_cv_results[0].iloc[0:8, :].reset_index().rename({'index': 'Index'}, axis=1).drop('TT (Sec)', axis=1)
cv_results.insert(2, 'Optimized for', '',)
cv_results.loc[len(cv_results)] = [9, get_model_name(models[-1]), ''] + list(compare_cv_results[-1].loc['Mean', :])

tuned_models_auc_f1_mean = []
for i, x in enumerate(tuned_models_auc_f1):
    df = pd.DataFrame(tuned_models_auc_f1_cv[i].loc['Mean', :].to_frame().T.reset_index(drop=True), columns=['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC'])
    df.insert(0, 'Index', i)
    df.insert(1, 'Model', get_model_name(tuned_models_auc_f1[i]))
    df.insert(2, 'Optimized for', 'AUC-F1')
    tuned_models_auc_f1_mean.append(df)
tuned_models_f1_auc_mean = []
for i, x in enumerate(tuned_models_f1_auc):
    df = pd.DataFrame(tuned_models_f1_auc_cv[i].loc['Mean', :].to_frame().T.reset_index(drop=True), columns=['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC'])
    df.insert(0, 'Index', i)
    df.insert(1, 'Model', get_model_name(tuned_models_f1_auc[i]))
    df.insert(2, 'Optimized for', 'F1-AUC')
    tuned_models_f1_auc_mean.append(df)
    
cv_results = pd.concat([cv_results] + tuned_models_auc_f1_mean + tuned_models_f1_auc_mean).reset_index(drop=True)
cv_results.sort_values(by='F1',ascending=False)
metrics = []

for i, model in enumerate(models):
    x = predict_model(model)
    real = x["ICU_NEW"]
    predicted = x["Label"]
    out = get_config('display_container')[-1]
    out_metrics = [i, get_model_name(model), ''] + [out.iloc[0,x] for x in range(1,8)]
    metrics.append(out_metrics)
for i, model in enumerate(tuned_models_auc_f1):
    x = predict_model(model)
    real = x["ICU_NEW"]
    predicted = x["Label"]
    out = get_config('display_container')[-1]
    out_metrics = [i, get_model_name(model), 'AUC-F1'] + [out.iloc[0,x] for x in range(1,8)]
    metrics.append(out_metrics)
for i, model in enumerate(tuned_models_f1_auc):
    x = predict_model(model)
    real = x["ICU_NEW"]
    predicted = x["Label"]
    out = get_config('display_container')[-1]
    out_metrics = [i, get_model_name(model), 'F1-AUC'] + [out.iloc[0,x] for x in range(1,8)]
    metrics.append(out_metrics)

test_results = pd.DataFrame(metrics, columns=['Index','Model', 'Optimized for', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC']).reset_index(drop=True)
test_results.sort_values(by='F1',ascending=False)
cv_test_average = test_results.copy()
value_columns = cv_test_average.columns[-7:]
for x in value_columns:
    cv_test_average[x] = np.absolute(cv_test_average[x] - cv_results[x])/cv_results[x]
cv_test_average['Mean Difference'] = cv_test_average.drop('Index', axis=1).mean(numeric_only=True, axis=1)
cv_test_average['Max Difference'] = cv_test_average.drop('Index', axis=1).max(numeric_only=True, axis=1)
cv_test_average.sort_values(by='Mean Difference', ascending=True)
blend_stack_cv_results = []
blend_stack_models = []
models_to_blend = [tuned_models_f1_auc[8], tuned_models_f1_auc[1], models[0]]

blend = blend_models(models_to_blend, method='soft')
blend_stack_models.append(blend)
blend_stack_cv_results.append(get_config('display_container')[-1])
models_to_stack = [tuned_models_f1_auc[8], tuned_models_f1_auc[1], models[0], tuned_models_f1_auc[4]]

stack_catboost = stack_models(models_to_stack, meta_model = models[4])
blend_stack_models.append(stack_catboost)
blend_stack_cv_results.append(get_config('display_container')[-1])
stack_rf = stack_models(models_to_stack, meta_model = models[-1])
blend_stack_models.append(stack_rf)
blend_stack_cv_results.append(get_config('display_container')[-1])
names = ['Voting Classifier', 'Stack CatBoost', 'Stack Random Forest']
blend_stack_cv_results_mean = []
for i, x in enumerate(blend_stack_cv_results):
    df = pd.DataFrame(blend_stack_cv_results[i].loc['Mean', :].to_frame().T.reset_index(drop=True), columns=['Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC'])
    df.insert(0, 'Index', i)
    df.insert(1, 'Model', names[i])
    df.insert(2, 'Optimized for', '')
    blend_stack_cv_results_mean.append(df)
    
blend_stack_cv_results_mean = pd.concat(blend_stack_cv_results_mean).reset_index(drop=True)
blend_stack_cv_results_mean.sort_values(by='F1',ascending=False)
blend_stack_test_results = []
for i, model in enumerate(blend_stack_models):
    x = predict_model(model)
    real = x["ICU_NEW"]
    predicted = x["Label"]
    out = get_config('display_container')[-1]
    out_metrics = [i, names[i], ''] + [out.iloc[0,x] for x in range(1,8)]
    blend_stack_test_results.append(out_metrics)

blend_stack_test_results = pd.DataFrame(blend_stack_test_results, columns=['Index','Model', 'Optimized for', 'Accuracy', 'AUC', 'Recall', 'Prec.', 'F1', 'Kappa', 'MCC']).reset_index(drop=True)
blend_stack_test_results.sort_values(by='F1',ascending=False)
blend_stack_cv_test_average = blend_stack_test_results.copy()
value_columns = blend_stack_cv_test_average.columns[-7:]
for x in value_columns:
    blend_stack_cv_test_average[x] = np.absolute(blend_stack_cv_test_average[x] - blend_stack_cv_results_mean[x])/blend_stack_cv_results_mean[x]
blend_stack_cv_test_average['Mean Difference'] = blend_stack_cv_test_average.drop('Index', axis=1).mean(numeric_only=True, axis=1)
blend_stack_cv_test_average['Max Difference'] = blend_stack_cv_test_average.drop('Index', axis=1).max(numeric_only=True, axis=1)
blend_stack_cv_test_average.sort_values(by='Mean Difference', ascending=True)
summary_cv_results = pd.concat([blend_stack_cv_results_mean, cv_results.sort_values(by='F1',ascending=False).iloc[0:10,:]]).reset_index(drop=True)
display(summary_cv_results.sort_values(by='F1',ascending=False))
display(summary_cv_results.sort_values(by='AUC',ascending=False))
plot_model(blend, plot='confusion_matrix')
plot_model(blend, plot='class_report')
interpret_model(tuned_models_auc_f1[4])
interpret_model(models[0])
plot_model(models[0], plot='confusion_matrix')
plot_model(models[0], plot='class_report')
plot_model(models[0], plot='auc')
plot_model(tuned_models_f1_auc[1], plot='confusion_matrix')
plot_model(tuned_models_f1_auc[1], plot='class_report')
plot_model(tuned_models_f1_auc[1], plot='auc')
interpret_model(tuned_models_f1_auc[-1])
plot_model(tuned_models_f1_auc[-1], plot='confusion_matrix')
plot_model(tuned_models_f1_auc[-1], plot='class_report')
plot_model(tuned_models_f1_auc[-1], plot='auc')
!rm -rf *.pkl
from datetime import date
final_selected_models = models_to_stack + [blend]
for model in final_selected_models:
    save_model(finalize_model(model), f'covid19_{get_model_name(model).replace(" ", "_")}_{date.today()}')
    
!rm -rf catboost_info
!rm -rf cb_model.json