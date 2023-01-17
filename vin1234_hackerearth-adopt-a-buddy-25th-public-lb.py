# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



%matplotlib inline
# To print multiple output in a cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
train=pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/train.csv')



test=pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/test.csv')
train.head(20)

train.shape



print('-----------'*5)

test.head()

test.shape
# test[test['length(m)']==0]



train[train['length(m)']==0]



# train[train['height(cm)']==0]
train['breed_category'].value_counts()



train['pet_category'].value_counts()
class data_fabrication:

#     def __init__(self,train_data,test_data):

#         self.train_data=train_data

#         self.test_data=test_data

        

    def merge(self,train_data,test_data):



        """

        create a new columns to segregate the train and test data



        """

        train_data['train_or_test']='train'

        test_data['train_or_test']='test'

        df=pd.concat([train_data,test_data])

        return df



        """

        Now let's create a method to deal with pet_id 

        By simply removing the ```alphabatics and _``` character

        """

    

    def Id(self,train_data,test_data):

        df=self.merge(train,test)

        df['U_pet_id']=df['pet_id'].str.replace('ANSL_','').astype('int')

        

        # No improvement with these variable 

        

#         df['week_pet_id'] = df['U_pet_id']%7

#         df['month_pet_id'] = df['U_pet_id']%30

#         df['year_pet_id'] = df['U_pet_id']%365

#         df['num_weeks_pet_id'] = df['U_pet_id']//7

#         df['num_year_pet_id'] = df['U_pet_id']//365

#         df['num_quarter_pet_id'] = df['U_pet_id']//90

#         df['quarter_pet_id'] = df['U_pet_id']%90  

        

        df.drop(['pet_id'],axis=1,inplace=True)

        return df

    

    #function to impute the missing values 

    def Impute(self,train_data,test_data):

        df=self.Id(train_data,test_data)

        df['condition']=df['condition'].fillna(3.0).astype('int')

        return df

        
# class for time series feature creation.



class Time_Series_Transformation():    

    

    def __init__(self,train,test):

        self.object1=data_fabrication()

        

    

    def create_week_date_featues_arrival(self): 

        

        df=self.object1.Impute(train,test)

        

        df['Year_arrival'] = pd.to_datetime(df['listing_date']).dt.year

        df['Month_arrival'] = pd.to_datetime(df['listing_date']).dt.month

        df['Day_arrival'] = pd.to_datetime(df['listing_date']).dt.day

        df['Dayofweek_arrival'] = pd.to_datetime(df['listing_date']).dt.dayofweek

        df['DayOfyear_arrival'] = pd.to_datetime(df['listing_date']).dt.dayofyear

        df['Week_arrival'] = pd.to_datetime(df['listing_date']).dt.week

        df['Quarter_arrival'] = pd.to_datetime(df['listing_date']).dt.quarter 

        df['Is_month_start_arrival'] = pd.to_datetime(df['listing_date']).dt.is_month_start

        df['Is_month_end_arrival'] = pd.to_datetime(df['listing_date']).dt.is_month_end

        df['Is_quarter_start_arrival'] = pd.to_datetime(df['listing_date']).dt.is_quarter_start

        df['Is_quarter_end_arrival'] = pd.to_datetime(df['listing_date']).dt.is_quarter_end

        df['Is_year_start_arrival'] = pd.to_datetime(df['listing_date']).dt.is_year_start

        df['Is_year_end_arrival'] = pd.to_datetime(df['listing_date']).dt.is_year_end

        df['Semester_arrival'] = np.where(df['listing_date'].isin([1,2]),1,2)

        df['Is_weekend_arrival'] = np.where(df['listing_date'].isin([5,6]),1,0)

        df['Is_weekday_arrival'] = np.where(df['listing_date'].isin([0,1,2,3,4]),1,0)

        df['Days_in_month_arrival'] = pd.to_datetime(df['listing_date']).dt.days_in_month

        df['Hour_arrival'] = pd.to_datetime(df['listing_date']).dt.hour

        df.drop(['listing_date'],axis=1,inplace=True)

        return df

    

    def create_week_date_featues_issue(self):

        

        df=self.create_week_date_featues_arrival()

        

        df['Year_issue'] = pd.to_datetime(df['issue_date']).dt.year

        df['Month_issue'] = pd.to_datetime(df['issue_date']).dt.month

        df['Day_issue'] = pd.to_datetime(df['issue_date']).dt.day

        df['Dayofweek_issue'] = pd.to_datetime(df['issue_date']).dt.dayofweek

        df['DayOfyear_issue'] = pd.to_datetime(df['issue_date']).dt.dayofyear

        df['Week_issue'] = pd.to_datetime(df['issue_date']).dt.week

        df['Quarter_issue'] = pd.to_datetime(df['issue_date']).dt.quarter 

        df['Is_month_start_issue'] = pd.to_datetime(df['issue_date']).dt.is_month_start

        df['Is_month_end_issue'] = pd.to_datetime(df['issue_date']).dt.is_month_end

        df['Is_quarter_start_issue'] = pd.to_datetime(df['issue_date']).dt.is_quarter_start

        df['Is_quarter_end_issue'] = pd.to_datetime(df['issue_date']).dt.is_quarter_end

        df['Is_year_start_issue'] = pd.to_datetime(df['issue_date']).dt.is_year_start

        df['Is_year_end_issue'] = pd.to_datetime(df['issue_date']).dt.is_year_end

        df['Semester_issue'] = np.where(df['issue_date'].isin([1,2]),1,2)

        df['Is_weekend_issue'] = np.where(df['issue_date'].isin([5,6]),1,0)

        df['Is_weekday_issue'] = np.where(df['issue_date'].isin([0,1,2,3,4]),1,0)

        df['Days_in_month_issue'] = pd.to_datetime(df['issue_date']).dt.days_in_month

        df.drop(['issue_date'],axis=1,inplace=True)

      

        return df

    

    



    # With below function we gonna calculate

    # The different bettween arriaval date and issued date



    def time_based_feature(self):

        df=self.create_week_date_featues_issue()

        df['year_took']=df['Year_arrival']-df['Year_issue']

        df['months_took']=df['Month_arrival']-df['Month_issue']

#         df['months_took']=df['year_took']*12

        df['days_took']=df['Day_arrival']-df['Day_issue']

    

#         df['days_took']=df['year_took']*365



#         df['quarter_took']=df['Quarter_arrival']-df['Quarter_issue']

    

#         df['semester_took']=df['Semester_arrival']-df['Semester_issue']

        return df

    

    def dummies_creation(self,columns):

        df=self.time_based_feature()

        for i in columns:

            df = pd.get_dummies(df, columns=[i])            

        return df

                
# start with the first class

new=Time_Series_Transformation(train,test)



col=['Is_month_start_issue','Is_month_end_issue','Is_quarter_start_issue','Is_quarter_end_issue','Is_year_start_issue','Is_year_end_issue',

    'Is_month_start_arrival','Is_month_end_arrival','Is_quarter_start_arrival','Is_quarter_end_arrival','Is_year_start_arrival','Is_year_end_arrival',

    'condition','color_type','Year_arrival']



df=new.dummies_creation(col)
# df.head()

# df.shape
# let;s try without dropping the U_pet_id(outputs a drop in the score)



# df.drop(['U_pet_id'],axis=1,inplace=True)
# seperate the train and test data





train_d=df.loc[df.train_or_test.isin(['train'])]

test_d=df.loc[df.train_or_test.isin(['test'])]



train_d.drop(columns={'train_or_test'},axis=1,inplace=True)

test_d.drop(columns={'train_or_test'},axis=1,inplace=True)

test_d.drop(columns={'breed_category','pet_category'},axis=1,inplace=True)
import lightgbm as lgb



from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import classification_report







class model_building:

        

#     def __init__(self,parameters,label_col1,label_col2):

#         self.parameters=parameters

#         self.label_col1='pet_category'

#         self.label_col2='breed_category'

                

    def create_model(self,df,column,drop_column):

        print('This time we are training model for {}'.format(column))

        df=df.drop([drop_column],axis=1)

        

        label_col=column

        df_train, df_eval = train_test_split(df, test_size=0.30,random_state=42, 

                                             shuffle=True,stratify=df[label_col])

        

        feature_cols = df.columns.tolist()



        feature_cols.remove(column)

        

        params={'learning_rate': 0.03,'max_depth': 9,'n_estimators': 5000,

                'objective': 'multiclass','boosting_type': 'gbdt','subsample': 0.7,

                'random_state': 42,'colsample_bytree': 0.7,'min_data_in_leaf': 31,

                'reg_alpha': 1.7,'reg_lambda': 1.11}        

        

        clf = lgb.LGBMClassifier(**params)

        

        clf.fit(df_train[feature_cols], df_train[label_col], early_stopping_rounds=100, 

                eval_set=[(df_train[feature_cols], df_train[label_col]), (df_eval[feature_cols], df_eval[label_col])], 

                eval_metric='multi_logloss', verbose=True)

        

        

        

        eval_score = accuracy_score(df_eval[label_col], clf.predict(df_eval[feature_cols]))

        

        print('\n')

        print('------------'*5)

        print('Eval ACC: {}'.format(eval_score))

        

        df_train = pd.concat((df_train, df_eval))

        

        clf = lgb.LGBMClassifier(**params)



        clf.fit(df_train[feature_cols], df_train[label_col], eval_metric='multi_error', verbose=False)



        # eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))

        eval_score_acc = accuracy_score(df_train[label_col], clf.predict(df_train[feature_cols]))



        print('ACC: {}'.format(eval_score_acc))   

        

        return clf,feature_cols



               

    def create_prediction(self,df,column,drop_column):

        

        clf,feature_col=self.create_model(df,column,drop_column)

        self.test=test_d

        prediction = clf.predict(self.test[feature_col])

        return prediction

# 'breed_category'

# 'pet_category'



model=model_building()

# model=model.create_model(train_d,'pet_category','breed_category')



prediction=model.create_prediction(train_d,'pet_category','breed_category')



model2=model_building()

prediction2=model2.create_prediction(train_d,'breed_category','pet_category')
# prediction2

# train_d.shape
# y1=train_d['pet_category']

# y2=train_d['breed_category']



# # x=train_d.drop(['breed_category','pet_category'],axis=1)



# train_d1=train_d.drop(['breed_category'],axis=1)



# train_d2=train_d.drop(['pet_category'],axis=1)
# # Model training importations



# import lightgbm as lgb



# from sklearn.model_selection import train_test_split, KFold, StratifiedKFold



# from sklearn.metrics import mean_squared_log_error, mean_squared_error

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import roc_auc_score

# from sklearn.metrics import accuracy_score

# from sklearn.metrics import log_loss

# from sklearn.metrics import classification_report

# import lightgbm as lgb

# label_col1='pet_category'

# label_col2='breed_category'
# df_train1, df_eval1 = train_test_split(train_d1, test_size=0.30, random_state=42, shuffle=True, stratify=train_d1[label_col1])
# # for firt model

# feature_cols1 = train_d1.columns.tolist()



# feature_cols1.remove('pet_category')



# #for second model



# feature_cols2 = train_d2.columns.tolist()



# feature_cols2.remove('breed_category')

# params = {}

# params['learning_rate'] = 0.25

# params['max_depth'] = 10

# params['n_estimators'] = 10000

# params['objective'] = 'multiclass'

# params['boosting_type'] = 'gbdt'

# params['subsample'] = 0.7

# params['random_state'] = 42

# params['colsample_bytree']=0.7

# params['min_data_in_leaf'] = 25

# params['reg_alpha'] = 1.7

# params['reg_lambda'] = 1.11
# params
# clf = lgb.LGBMClassifier(**params)
# clf.fit(df_train1[feature_cols1], df_train1[label_col1], early_stopping_rounds=1000, eval_set=[(df_train1[feature_cols1], df_train1[label_col1]), (df_eval1[feature_cols1], df_eval1[label_col1])], eval_metric='multi_error', verbose=True)

# eval_score = accuracy_score(df_eval1[label_col1], clf.predict(df_eval1[feature_cols1]))



# print('Eval ACC: {}'.format(eval_score))
# df_train1 = pd.concat((df_train1, df_eval1))
# clf = lgb.LGBMClassifier(**params)



# clf.fit(df_train1[feature_cols1], df_train1[label_col1], eval_metric='multi_error', verbose=False)



# # eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))

# eval_score_acc = accuracy_score(df_train1[label_col1], clf.predict(df_train1[feature_cols1]))



# print('ACC: {}'.format(eval_score_acc))
# y_pred1 = clf.predict(test_d[feature_cols1])
# train_d2=train_d.drop(['pet_category'],axis=1)



# let's add the predicted feature and the observe our final result



# train_d2['pet_category']=train['pet_category']
# label_col2='breed_category'
# df_train2, df_eval2 = train_test_split(train_d2, test_size=0.30, random_state=42, shuffle=True, stratify=train_d2[label_col2])
# feature_cols2 = train_d2.columns.tolist()



# feature_cols2.remove('breed_category')
# clf2 = lgb.LGBMClassifier(**params)
# clf2.fit(df_train2[feature_cols2], df_train2[label_col2], early_stopping_rounds=1000, eval_set=[(df_train2[feature_cols2], df_train2[label_col2]), (df_eval2[feature_cols2], df_eval2[label_col2])], eval_metric='multi_error', verbose=True)

# eval_score2 = accuracy_score(df_eval2[label_col2], clf2.predict(df_eval2[feature_cols2]))



# print('Eval ACC: {}'.format(eval_score2))
# df_train2 = pd.concat((df_train2, df_eval2))
# clf2 = lgb.LGBMClassifier(**params)



# clf2.fit(df_train2[feature_cols2], df_train2[label_col2], eval_metric='multi_error', verbose=False)



# # eval_score_auc = roc_auc_score(df_train[label_col], clf.predict(df_train[feature_cols]))

# eval_score_acc = accuracy_score(df_train2[label_col2], clf2.predict(df_train2[feature_cols2]))



# print('ACC: {}'.format(eval_score_acc))
# add first predicted values inthe test data 

# test_d['pet_category']=y_pred1
# y_pred2 = clf2.predict(test_d[feature_cols2])
# train.head()
# submission=pd.DataFrame({'pet_id':test['pet_id'],'breed_category':y_pred2,'pet_category':y_pred1})



submission=pd.DataFrame({'pet_id':test['pet_id'],'breed_category':prediction2,'pet_category':prediction})
submission.head()
# submission.to_csv('submission28.csv',index=False)
submission.shape