import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

final_d= pd.read_csv("../input/new-data/var_final.csv")

out_of_time = pd.read_csv("../input/outoftime/all_df_after_1101.csv")

final_d.head()
out_of_time.head()
final_d.columns
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

final_data_20=final_d[x_name_20]
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

import random

num_test = 0.3
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(max_depth=8,n_estimators=50)



random.seed(1)

y=final_data_20['Fraud']

x=final_data_20[x_name_20]

y_out_of_time=out_of_time['Fraud']

x_out_of_time=out_of_time[x_name_20]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=num_test, random_state=1)

#cv_model = GridSearchCV(model, cv_parameters)

random_forest.fit(x_train, y_train)

y_traing_prob=random_forest.predict_proba(x_train)[:,1]

y_traing_pred=random_forest.predict(x_train)

y_test_prob=random_forest.predict_proba(x_test)[:,1]

y_test_pred=random_forest.predict(x_test)

y_out_of_time_prob=random_forest.predict_proba(x_out_of_time)[:,1]

y_out_of_time_pred=random_forest.predict(x_out_of_time)
y_traing = pd.DataFrame({'y_traing_real': y_train,'y_traing_prob': y_traing_prob,'y_traing_pred':y_traing_pred})

y_traing=y_traing.sort_values('y_traing_prob',ascending=False)

y_test= pd.DataFrame({'y_test_real': y_test,'y_test_prob': y_test_prob,'y_test_pred':y_test_pred})

y_test=y_test.sort_values('y_test_prob',ascending=False)

y_out_of_time = pd.DataFrame({'y_out_of_time_real': y_out_of_time,'y_out_of_time_prob': y_out_of_time_prob,'y_out_of_time_pred':y_out_of_time_pred})

y_out_of_time=y_out_of_time.sort_values('y_out_of_time_prob',ascending=False)

        
y_out_of_time
y_out_of_time_cutpoint=int(y_out_of_time.shape[0]*0.03)

y_out_of_time_tem=y_out_of_time.head(y_out_of_time_cutpoint)

y_out_of_time_tem
y_out_of_time['y_out_of_time_real'].sum()
y_out_of_time_tem['y_out_of_time_real'].sum()
dfr=y_out_of_time['y_out_of_time_real'].sum()/y_out_of_time_tem['y_out_of_time_real'].sum()

dfr