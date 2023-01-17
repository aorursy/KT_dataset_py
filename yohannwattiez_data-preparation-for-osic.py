import tensorflow as tf

import pandas as pd

import numpy as np

import os

import random

import pickle
#Import raw data and copy

train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
ID='Patient_Week'
#Take mean/first of several tests on same weeks

train_temp = train.groupby(['Patient', 'Weeks']).mean()[['FVC', 'Percent']]

train_temp[['Age', 'Sex', 'SmokingStatus']] = train.groupby(['Patient', 'Weeks']).first()[['Age', 'Sex', 'SmokingStatus']]

train = train_temp.reset_index()
train['Min_week'] = train.groupby('Patient')['Weeks'].transform('min')

base = train.loc[train.Weeks == train.Min_week]

base = base[['Patient','FVC']].copy()

base.columns = ['Patient','Base_FVC']

train = train.merge(base, on='Patient', how='left')

train['Base_week'] = train['Weeks'] - train['Min_week']
# #Data Augmentation

# rename_cols = {'Weeks_x':'Base_week', 'FVC_x': 'Base_FVC', 'Percent_x': 'Base_percent', 'Age_x': 'Age', 'SmokingStatus_x': 'SmokingStatus', 'Sex_x':'Sex', 'Weeks_y':'Weeks', 'FVC_y': 'FVC'}

# drop_cols = ['Age_y', 'Sex_y', 'SmokingStatus_y', 'Percent_y']

# train = train.merge(train, how='left', left_on='Patient', right_on='Patient').rename(columns=rename_cols).drop(columns=drop_cols)

# train[ID] = train['Patient'].astype(str) + '_' + train['Weeks'].astype(str)

# train = train[['Patient', 'Base_week', 'Base_FVC', 'Base_percent', 'Age', 'Sex', 'SmokingStatus', 'Weeks', 'Patient_Week', 'FVC']].reset_index(drop=True)

# #Keep base week atm

# # train = train[train['Weeks']!=train['Base_week']]
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder



#Override OneHotEncoder to have the column names created automatically (Ex-smoker, Never Smoked...) 

class OneHotEncoder(SklearnOneHotEncoder):

    def __init__(self, **kwargs):

        super(OneHotEncoder, self).__init__(**kwargs)

        self.fit_flag = False



    def fit(self, X, **kwargs):

        out = super().fit(X)

        self.fit_flag = True

        return out



    def transform(self, X, categories, index='', name='', **kwargs):

        sparse_matrix = super(OneHotEncoder, self).transform(X)

        new_columns = self.get_new_columns(X=X, name=name, categories=categories)

        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)

        return d_out



    def fit_transform(self, X, categories, index, name, **kwargs):

        self.fit(X)

        return self.transform(X, categories=categories, index=index, name=name)



    def get_new_columns(self, X, name, categories):

        new_columns = []

        for j in range(len(categories)):

            new_columns.append('{}_{}'.format(name, categories[j]))

        return new_columns



from sklearn.preprocessing import LabelEncoder

from sklearn.exceptions import NotFittedError



def standardisation(x, u, s):

    return (x-u)/s



def normalization(x, ma, mi):

    return (x-mi)/(ma-mi)



class data_preparation():

    def __init__(self, bool_normalization=True,bool_standard=False):

        self.enc_sex = LabelEncoder()

        self.enc_smok = LabelEncoder()

        self.onehotenc_smok = OneHotEncoder()

        self.standardisation = bool_standard

        self.normalization = bool_normalization

        

        

    def __call__(self, data_untransformed):

        data = data_untransformed.copy(deep=True)

        

        #For the test set/Already fitted

        try:

            data['Sex'] = self.enc_sex.transform(data['Sex'].values)

            data['SmokingStatus'] = self.enc_smok.transform(data['SmokingStatus'].values)

            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)

            

            #Standardisation

            if self.standardisation:

                data['Base_week'] = standardisation(data['Base_week'],self.base_week_mean,self.base_week_std)

                data['Base_FVC'] = standardisation(data['Base_FVC'],self.base_fvc_mean,self.base_fvc_std)

                data['Base_percent'] = standardisation(data['Base_percent'],self.base_percent_mean,self.base_percent_std)

                data['Age'] = standardisation(data['Age'],self.age_mean,self.age_std)

                data['Weeks'] = standardisation(data['Weeks'],self.weeks_mean,self.weeks_std)

            

            #Normalization

            if self.normalization:

                data['Base_week'] = normalization(data['Base_week'],self.base_week_max,self.base_week_min)

                data['Base_FVC'] = normalization(data['Base_FVC'],self.base_fvc_max,self.base_fvc_min)

                data['Percent'] = normalization(data['Percent'],self.base_percent_max,self.base_percent_min)

                data['Age'] = normalization(data['Age'],self.age_max,self.age_min)

                data['Weeks'] = normalization(data['Weeks'],self.weeks_max,self.weeks_min)

                data['Min_week'] = normalization(data['Min_week'],self.base_week_max,self.base_week_min)



        #For the train set/Not yet fitted    

        except NotFittedError:

            data['Sex'] = self.enc_sex.fit_transform(data['Sex'].values)

            data['SmokingStatus'] = self.enc_smok.fit_transform(data['SmokingStatus'].values)

            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.fit_transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)

            

            #Standardisation

            if self.standardisation:

                self.base_week_mean = data['Base_week'].mean()

                self.base_week_std = data['Base_week'].std()

                data['Base_week'] = standardisation(data['Base_week'],self.base_week_mean,self.base_week_std)



                self.base_fvc_mean = data['Base_FVC'].mean()

                self.base_fvc_std = data['Base_FVC'].std()

                data['Base_FVC'] = standardisation(data['Base_FVC'],self.base_fvc_mean,self.base_fvc_std)



                self.base_percent_mean = data['Base_percent'].mean()

                self.base_percent_std = data['Base_percent'].std()

                data['Base_percent'] = standardisation(data['Base_percent'],self.base_percent_mean,self.base_percent_std)



                self.age_mean = data['Age'].mean()

                self.age_std = data['Age'].std()

                data['Age'] = standardisation(data['Age'],self.age_mean,self.age_std)



                self.weeks_mean = data['Weeks'].mean()

                self.weeks_std = data['Weeks'].std()

                data['Weeks'] = standardisation(data['Weeks'],self.weeks_mean,self.weeks_std)



                

            #Normalization

            if self.normalization:

                self.base_week_min = data['Base_week'].min()

                self.base_week_max = data['Base_week'].max()

                data['Base_week'] = normalization(data['Base_week'],self.base_week_max,self.base_week_min)



                self.base_fvc_min = data['Base_FVC'].min()

                self.base_fvc_max = data['Base_FVC'].max()

                data['Base_FVC'] = normalization(data['Base_FVC'],self.base_fvc_max,self.base_fvc_min)



                self.base_percent_min = data['Percent'].min()

                self.base_percent_max = data['Percent'].max()

                data['Percent'] = normalization(data['Percent'],self.base_percent_max,self.base_percent_min)



                self.age_min = data['Age'].min()

                self.age_max = data['Age'].max()

                data['Age'] = normalization(data['Age'],self.age_max,self.age_min)



                self.weeks_min = data['Weeks'].min()

                self.weeks_max = data['Weeks'].max()

                data['Weeks'] = normalization(data['Weeks'],self.weeks_max,self.weeks_min)

                

                self.base_week_min = data['Min_week'].min()

                self.base_week_max = data['Min_week'].max()

                data['Min_week'] = normalization(data['Min_week'],self.base_week_max,self.base_week_min)



            

        return data
#Transform data

data_prep = data_preparation(bool_normalization=True)

train = data_prep(train)
#Keep the data_prep object for tabular X_prediction

pickefile = open('data_prep', 'wb')

pickle.dump(data_prep, pickefile)

pickefile.close()



train.to_csv('train.csv', index=False)
# list_files = pd.DataFrame(columns=['files'])

# list_files['Patient'] = [i for i in os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/')]

# list_files.loc[:,'files'] = [os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/'+i) for i in list_files['Patient']]

# list_files['num_files'] = list_files['files'].apply(lambda x:len(x))
# import pydicom



# def sort_function(x): #Get the files in the right order

#     return int(x.split('.')[0])



# size_image = pd.DataFrame(columns=['img_names','path', 'patient', 'size'])

# path = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/'

# for patient in os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/'):

#     list_img_names = sorted([i for i in os.listdir(path+'/'+patient)], key=sort_function)

#     df = pd.DataFrame({'img_names':list_img_names})

#     df['path'] = df.apply(lambda x:path+patient+'/'+x.loc['img_names'], axis=1)

#     df['patient'] = patient

#     df['size'] = df.apply(lambda x: (pydicom.dcmread(x.loc['path']).data_element('Rows').value, pydicom.dcmread(x.loc['path']).data_element('Rows').value), axis=1)

#     size_image = size_image.append(df)
# size_image.to_csv('size_image.csv', index=False)

# list_files.to_csv('list_files.csv', index=False)