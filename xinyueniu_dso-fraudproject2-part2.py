import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

final_d= pd.read_csv("../input/all-final-data/var_final.csv")

out_of_time = pd.read_csv("../input/outoftime/all_df_after_1101.csv")

final_d.head()
final_d.columns
#y=final_data['Fraud']

x_name_20=['AC1_ACC30', 'Amount_avg_card_merchant_7',

       'Amount_median_card_merchant_7', 'Amount_avg_card_14',

       'Amount_avg_card_30', 'Amount_median_card_30',

       'Amount_max_card_state_14', 'Amount_avg_card_state_14',

       'Amount_max_card_zip_30', 'Amount_median_card_zip_30',

       'Amount_avg_card_1', 'Amount_median_card_merchant_1',

       'Amount_max_merchant_7', 'Amount_max_card_merchant_1',

       'Amount_avg_merchant_7', 'Amount_median_card_merchant_30',

       'Amount_median_card_merchant_14', 'Amount_sum_card_merchant_7',

       'Amount_avg_merchant_1', 'Amount_sum_card_3','Fraud']

x_name_25=['AC1_ACC30', 'Amount_avg_card_merchant_7',

       'Amount_median_card_merchant_7', 'Amount_avg_card_14',

       'Amount_avg_card_30', 'Amount_median_card_30',

       'Amount_max_card_state_14', 'Amount_avg_card_state_14',

       'Amount_max_card_zip_30', 'Amount_median_card_zip_30',

       'Amount_avg_card_1', 'Amount_median_card_merchant_1',

       'Amount_max_merchant_7', 'Amount_max_card_merchant_1',

       'Amount_avg_merchant_7', 'Amount_median_card_merchant_30',

       'Amount_median_card_merchant_14', 'Amount_sum_card_merchant_7',

       'Amount_avg_merchant_1', 'Amount_sum_card_3','Amount_max_card_merchant_7', 'Amount_avg_card_merchant_3',

       'Amount_avg_card_zip_1', 'Amount_max_card_state_1',

       'Amount_max_card_zip_14','Fraud']

x_name_30=['AC1_ACC30', 'Amount_avg_card_merchant_7',

       'Amount_median_card_merchant_7', 'Amount_avg_card_14',

       'Amount_avg_card_30', 'Amount_median_card_30',

       'Amount_max_card_state_14', 'Amount_avg_card_state_14',

       'Amount_max_card_zip_30', 'Amount_median_card_zip_30',

       'Amount_avg_card_1', 'Amount_median_card_merchant_1',

       'Amount_max_merchant_7', 'Amount_max_card_merchant_1',

       'Amount_avg_merchant_7', 'Amount_median_card_merchant_30',

       'Amount_median_card_merchant_14', 'Amount_sum_card_merchant_7',

       'Amount_avg_merchant_1', 'Amount_sum_card_3','Amount_max_card_merchant_7', 'Amount_avg_card_merchant_3',

       'Amount_avg_card_zip_1', 'Amount_max_card_state_1',

       'Amount_max_card_zip_14','Amount_max_card_merchant_14',

       'Amount_sum_merchant_3', 'Amount_max_card_state_30',

       'Amount_max_card_1', 'Amount_sum_card_merchant_3','Fraud']

x_name_35=['AC1_ACC30', 'Amount_avg_card_merchant_7',

       'Amount_median_card_merchant_7', 'Amount_avg_card_14',

       'Amount_avg_card_30', 'Amount_median_card_30',

       'Amount_max_card_state_14', 'Amount_avg_card_state_14',

       'Amount_max_card_zip_30', 'Amount_median_card_zip_30',

       'Amount_avg_card_1', 'Amount_median_card_merchant_1',

       'Amount_max_merchant_7', 'Amount_max_card_merchant_1',

       'Amount_avg_merchant_7', 'Amount_median_card_merchant_30',

       'Amount_median_card_merchant_14', 'Amount_sum_card_merchant_7',

       'Amount_avg_merchant_1', 'Amount_sum_card_3','Amount_max_card_merchant_7', 'Amount_avg_card_merchant_3',

       'Amount_avg_card_zip_1', 'Amount_max_card_state_1',

       'Amount_max_card_zip_14','Amount_max_card_merchant_14',

       'Amount_sum_merchant_3', 'Amount_max_card_state_30',

       'Amount_max_card_1', 'Amount_sum_card_merchant_3','Amount_max_merchant_3', 'Amount_sum_merchant_7',

       'Amount_median_card_1', 'Amount_avg_card_state_30',

       'Amount_sum_card_merchant_1','Fraud']

x_name_40=['AC1_ACC30', 'Amount_avg_card_merchant_7',

       'Amount_median_card_merchant_7', 'Amount_avg_card_14',

       'Amount_avg_card_30', 'Amount_median_card_30',

       'Amount_max_card_state_14', 'Amount_avg_card_state_14',

       'Amount_max_card_zip_30', 'Amount_median_card_zip_30',

       'Amount_avg_card_1', 'Amount_median_card_merchant_1',

       'Amount_max_merchant_7', 'Amount_max_card_merchant_1',

       'Amount_avg_merchant_7', 'Amount_median_card_merchant_30',

       'Amount_median_card_merchant_14', 'Amount_sum_card_merchant_7',

       'Amount_avg_merchant_1', 'Amount_sum_card_3','Amount_max_card_merchant_7', 'Amount_avg_card_merchant_3',

       'Amount_avg_card_zip_1', 'Amount_max_card_state_1',

       'Amount_max_card_zip_14','Amount_max_card_merchant_14',

       'Amount_sum_merchant_3', 'Amount_max_card_state_30',

       'Amount_max_card_1', 'Amount_sum_card_merchant_3','Amount_max_merchant_3', 'Amount_sum_merchant_7',

       'Amount_median_card_1', 'Amount_avg_card_state_30',

       'Amount_sum_card_merchant_1','AC1_ANC30', 'Amount_median_card_zip_1',

       'AC1_ANC7', 'Amount_median_merchant_1', 'Amount_avg_card_zip_7','Fraud']

final_data_20=final_d[x_name_20]

final_data_25=final_d[x_name_25]

final_data_30=final_d[x_name_30]

final_data_35=final_d[x_name_35]

final_data_40=final_d[x_name_40]
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

import random

num_test = 0.3
def fit_classification(model,final_data,x_name,num_test=0.3):

    #cv_model = GridSearchCV(model, cv_parameters)

    model_fdr=[]

    for i in range(10):

        random.seed(i)

        y=final_data['Fraud']

        x=final_data[x_name]

        y_out_of_time=out_of_time['Fraud']

        x_out_of_time=out_of_time[x_name]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=num_test, random_state=i)

        #cv_model = GridSearchCV(model, cv_parameters)

        model.fit(x_train, y_train)

        y_traing_prob=model.predict_proba(x_train)[:,1]

        y_test_prob=model.predict_proba(x_test)[:,1]

        y_out_of_time_prob=model.predict_proba(x_out_of_time)[:,1]

        

        y_traing = pd.DataFrame({'y_traing_real': y_train,'y_traing_prob': y_traing_prob})

        y_traing=y_traing.sort_values('y_traing_prob',ascending=False)

        y_test= pd.DataFrame({'y_test_real': y_test,'y_test_prob': y_test_prob})

        y_test=y_test.sort_values('y_test_prob',ascending=False)

        y_out_of_time = pd.DataFrame({'y_out_of_time_real': y_out_of_time,'y_out_of_time_prob': y_out_of_time_prob})

        y_out_of_time=y_out_of_time.sort_values('y_out_of_time_prob',ascending=False)

        

        y_train_cutpoint=int(y_traing.shape[0]*0.03)

        train_num=y_traing['y_traing_real'][:y_train_cutpoint].sum()

        train_denum=y_traing['y_traing_real'].sum()

        fdr_traing=train_num/train_denum

        print(y_train_cutpoint,train_num,train_denum)

        y_test_cutpoint=int(y_test.shape[0]*0.03)

        test_num=y_test['y_test_real'][:y_test_cutpoint].sum()

        test_denum=y_test['y_test_real'].sum()

        fdr_test=test_num/test_denum

        print(y_test_cutpoint,test_num,test_denum)

        y_out_of_time_cutpoint=int(y_out_of_time.shape[0]*0.03)

        fdr_out_of_time=y_out_of_time['y_out_of_time_real'][:y_out_of_time_cutpoint].sum()/y_out_of_time['y_out_of_time_real'].sum()

        

        fdr_3=[fdr_traing,fdr_test,fdr_out_of_time]

        model_fdr.append(fdr_3)

    

    model_fdr = pd.DataFrame(model_fdr,columns=['traing','test','out of time'],

                             index=['iteration1','iteration2','iteration3','iteration4','iteration5','iteration6','iteration7','iteration8','iteration9','iteration10'])

    mean_train=model_fdr['traing'].mean()

    mean_test= model_fdr['test'].mean()

    mean_oot=model_fdr['out of time'].mean()

    model_fdr.loc['mean']=[mean_train,mean_test,mean_oot]

    return model_fdr
final_data_20['Fraud'].sum()
int(final_data_20.shape[0]*0.3)
random_forest = RandomForestClassifier(max_depth=8,n_estimators=80)



random_forest_20 = fit_classification(model=random_forest, final_data=final_data_20,x_name=x_name_20,num_test=0.3)

random_forest_20
from sklearn.linear_model import LogisticRegression

l2_logistic = LogisticRegression(penalty = 'l2')

l2_logistic_20 = fit_classification(l2_logistic, final_data_20,x_name_20,num_test=0.3)

l2_logistic_25 = fit_classification(l2_logistic, final_data_25,x_name_25,num_test=0.3)

l2_logistic_30 = fit_classification(l2_logistic, final_data_30,x_name_30,num_test=0.3)

l2_logistic_35 = fit_classification(l2_logistic, final_data_35,x_name_35,num_test=0.3)

l2_logistic_40 = fit_classification(l2_logistic, final_data_40,x_name_40,num_test=0.3)

l2_logistic = l2_logistic_20.merge(l2_logistic_25, how='outer', left_index=True, right_index=True).merge(l2_logistic_30, how='outer', left_index=True, right_index=True).merge(l2_logistic_35, how='outer', left_index=True, right_index=True).merge(l2_logistic_40, how='outer', left_index=True, right_index=True)

l2_logistic.columns = ['train_20v', 'test_20v','oot_20v','train_25v', 'test_25v','oot_25v','train_30v', 'test_30v','oot_30v','train_35v', 'test_35v','oot_35v','train_40v', 'test_40v','oot_40v']

l2_logistic
random_forest = RandomForestClassifier(max_depth=8,n_estimators=80)



random_forest_20 = fit_classification(model=random_forest, final_data=final_data_20,x_name=x_name_20,num_test=0.3)

random_forest_25 = fit_classification(model=random_forest, final_data=final_data_25,x_name=x_name_25,num_test=0.3)

random_forest_30 = fit_classification(model=random_forest, final_data=final_data_30,x_name=x_name_30,num_test=0.3)

random_forest_35 = fit_classification(model=random_forest, final_data=final_data_35,x_name=x_name_35,num_test=0.3)

random_forest_40 = fit_classification(model=random_forest, final_data=final_data_40,x_name=x_name_40,num_test=0.3)

random_forest = random_forest_20.merge(random_forest_25, how='outer', left_index=True, right_index=True).merge(random_forest_30, how='outer', left_index=True, right_index=True).merge(random_forest_35, how='outer', left_index=True, right_index=True).merge(random_forest_40, how='outer', left_index=True, right_index=True)

random_forest.columns = ['train_20v', 'test_20v','oot_20v','train_25v', 'test_25v','oot_25v','train_30v', 'test_30v','oot_30v','train_35v', 'test_35v','oot_35v','train_40v', 'test_40v','oot_40v']

random_forest.to_csv('random_forest_dfr3.csv')

random_forest
from xgboost.sklearn import XGBClassifier

XGB_Classifier = XGBClassifier(

    max_depth=8,

    learning_rate=0.08,

    n_estimators=80,

    objective="rank:pairwise",

    gamma=0,

    min_child_weight=1,

    max_delta_step=0,

    subsample=1,

    colsample_bytree=1,

    colsample_bylevel=1,

    reg_alpha=0,

    reg_lambda=1,

    scale_pos_weight=1,

    base_score=0.5,

    missing=None,

    silent=True,

    nthread=-1,

    seed=1)

#XGB = fit_classification(XGB_Classifier, final_data,num_test=0.3)



XGB_Classifier_20 = fit_classification(model=XGB_Classifier, final_data=final_data_20,x_name=x_name_20,num_test=0.3)

XGB_Classifier_25 = fit_classification(model=XGB_Classifier, final_data=final_data_25,x_name=x_name_25,num_test=0.3)

XGB_Classifier_30 = fit_classification(model=XGB_Classifier, final_data=final_data_30,x_name=x_name_30,num_test=0.3)

XGB_Classifier_35 = fit_classification(model=XGB_Classifier, final_data=final_data_35,x_name=x_name_35,num_test=0.3)

XGB_Classifier_40 = fit_classification(model=XGB_Classifier, final_data=final_data_40,x_name=x_name_40,num_test=0.3)

XGB_Classifier = XGB_Classifier_20.merge(XGB_Classifier_25, how='outer', left_index=True, right_index=True).merge(XGB_Classifier_30, how='outer', left_index=True, right_index=True).merge(XGB_Classifier_35, how='outer', left_index=True, right_index=True).merge(XGB_Classifier_40, how='outer', left_index=True, right_index=True)

XGB_Classifier.columns = ['train_20v', 'test_20v','oot_20v','train_25v', 'test_25v','oot_25v','train_30v', 'test_30v','oot_30v','train_35v', 'test_35v','oot_35v','train_40v', 'test_40v','oot_40v']

XGB_Classifier.to_csv('XGB_Classifier_dfr3.csv')

XGB_Classifier