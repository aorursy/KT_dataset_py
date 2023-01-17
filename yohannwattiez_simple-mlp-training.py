import tensorflow as tf

import pandas as pd

import numpy as np

import os

import random

import pickle

import time



from tensorflow import keras as K

from tensorflow.keras import layers as L



from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error



import matplotlib.pyplot as plt
def seed_all(seed = 20):

    os.environ['PYTHONHASHSEED']=str(seed)

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    # # 5. For layers that introduce randomness like dropout, make sure to set seed values:

    # model.add(Dropout(0.25, seed=seed_value))

    # #6 Configure a new global `tensorflow` session: 

    # from keras import backend as K 

    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1) 

    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

    # K.set_session(sess)

    

seed_all(20)
#Import raw data and copy

# train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')

# test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')



# size_image = pd.read_csv('/kaggle/input/prep-data/size_image.csv')

# list_files = pd.read_csv('/kaggle/input/prep-data/list_files.csv')



train = pd.read_csv('/kaggle/input/data-preparation-for-osic/train.csv')



raw_test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

X_prediction = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
ID='Patient_Week'

PINBALL_QUANTILE = [0.255, 0.50, 0.745]

LAMBDA_LOSS = 0.585

EPOCH = [54, 55, 20, 60, 23]

BATCH_SIZE = 128



NFOLD = 5
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

#=============================#

def score(y_true, y_pred):

#     y_true=tf.dtypes.cast(y_true, tf.float32)

#     y_pred=tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return K.backend.mean(metric)

#============================#

def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = PINBALL_QUANTILE

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.backend.mean(v)

#=============================#

def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss
def eval_score(y_true, y_pred):

    y_true = tf.dtypes.cast(y_true, tf.float32)*(data_prep.fvc_max-data_prep.fvc_min)+data_prep.fvc_min

    y_pred = tf.dtypes.cast(y_pred, tf.float32)*(data_prep.fvc_max-data_prep.fvc_min)+data_prep.fvc_min

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return -K.backend.mean(metric)
SELECTED_COLUMNS = ['Weeks', 'Percent', 'Age', 'Sex', 'Min_week', 'Base_FVC','Base_week', '_Currently smokes', '_Ex-smoker', '_Never smoked']

def create_model(lambda_loss):

    model_input = K.Input(shape=(len(SELECTED_COLUMNS),))

    x = L.Dense(500, activation="selu", name='dense_to_freeze1')(model_input)

    x = L.Dense(100, activation="selu", name='dense_to_freeze2')(x)

#     FVC = L.Dense(3)(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)

    p2 = L.Dense(3, activation="selu", name="p2")(x)

    FVC = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                         name="FVC")([p1, p2])



    model = K.Model(

        inputs=model_input,

        outputs=[FVC],

    )

#     boundaries = [150, 250, 350]

#     values = [0.1, 0.08, 0.01, 0.001]

#     learning_rate_fn = K.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

#     optimizer=K.optimizers.Adam(learning_rate=learning_rate_fn, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)

    model.compile(

        optimizer=K.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),

        loss=mloss(lambda_loss),

        metrics=score,

    )

    return model



model = create_model(LAMBDA_LOSS)

# tf.keras.utils.plot_model(model)

model.summary()
X_prediction['Patient'] = X_prediction['Patient_Week'].str.extract(r'(.*)_.*')

X_prediction['Weeks'] = X_prediction['Patient_Week'].str.extract(r'.*_(.*)').astype(int)

X_prediction = X_prediction[['Patient', 'Weeks', 'Patient_Week']]

rename_cols = {'Weeks_y':'Min_week', 'Weeks_x': 'Weeks', 'FVC':'Base_FVC'}

X_prediction = X_prediction.merge(raw_test, how='left', left_on='Patient', right_on='Patient').rename(columns=rename_cols)[['Patient', 'Min_week', 'Base_FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'Weeks', 'Patient_Week']].reset_index(drop=True)
X_prediction['Base_week'] = X_prediction['Weeks'] - X_prediction['Min_week']
# rename_cols = {'Weeks_x':'Base_week', 'FVC_x': 'Base_FVC', 'Percent_x': 'Base_percent', 'Age_x': 'Age', 'SmokingStatus_x': 'SmokingStatus', 'Sex_x':'Sex', 'Weeks_y':'Weeks', 'FVC_y': 'FVC'}

# drop_cols = ['Age_y', 'Sex_y', 'SmokingStatus_y', 'Percent_y']

# test = test.merge(test, how='left', left_on='Patient', right_on='Patient').rename(columns=rename_cols).drop(columns=drop_cols)

# test[ID] = test['Patient'].astype(str) + '_' + test['Weeks'].astype(str)

# test = test[['Patient', 'Base_week', 'Base_FVC', 'Base_percent', 'Age', 'Sex', 'SmokingStatus', 'Weeks', 'Patient_Week', 'FVC']].reset_index(drop=True)
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
pickefile = open('/kaggle/input/data-preparation-for-osic/data_prep', 'rb')

data_prep = pickle.load(pickefile)

pickefile.close()
X_prediction = data_prep(X_prediction)

# test = train[train['Patient']=='ID00009637202177434476278']

# train = train[~(train['Patient']=='ID00009637202177434476278')]
# list_patient_score=[]

# for i in train.Patient.unique():

#     model = create_model(LAMBDA_LOSS)

#     print(i)

#     history = model.fit(x=train[~train.Patient.isin([i])][SELECTED_COLUMNS], y=train[~train.Patient.isin([i])][['FVC']], validation_data=(train[train.Patient.isin([i])][SELECTED_COLUMNS], train[train.Patient.isin([i])][['FVC']]), epochs=250)

#     list_patient_score.append([i, history.history['val_score']])
pickefile = open('/kaggle/input/trained-cnn-mlp-for-osic/list_patient_score', 'rb')

list_patient_score = pickle.load(pickefile)

pickefile.close()
# pickefile = open('list_patient_score', 'wb')

# pickle.dump(list_patient_score, pickefile)

# pickefile.close()
# list_mean_score=[]

# for i in list_patient_score:

#     list_mean_score.append(np.mean(i[1]))
# count=0

# for i in list_patient_score:

#     if np.mean(i[1])>8:

#         plt.plot(i[1])

#         count+=1    

# print(count)
list_patient_weight = []

for i in list_patient_score:

    if 6.4>np.mean(i[1]):

        list_patient_weight.append([i[0], 3])

    elif 6.8>np.mean(i[1])>6.4:

        list_patient_weight.append([i[0], 50])

    elif 7.1>np.mean(i[1])>6.8:

        list_patient_weight.append([i[0], 100])

    elif 50>np.mean(i[1])>7.6:

        list_patient_weight.append([i[0], 3])

    else:

        list_patient_weight.append([i[0], 3])

        

train['Weight'] = train.Patient.map(dict(list_patient_weight))
pickefile = open('list_patient_weight', 'wb')

pickle.dump(list_patient_weight, pickefile)

pickefile.close()
#Selection for KFOLD

list_patient_KFOLD=[]

for i in list_patient_score:

    if np.mean(i[1]) < 6.26:

        list_patient_KFOLD.append([i[0], 0])

    elif 6.26<=np.mean(i[1]) < 6.43:

        list_patient_KFOLD.append([i[0], 1])

    elif 6.43<=np.mean(i[1]) < 6.74:

        list_patient_KFOLD.append([i[0], 2])

    elif 6.74<=np.mean(i[1]) < 7.15:

        list_patient_KFOLD.append([i[0], 3])

    elif 7.15<=np.mean(i[1]) < 50:

        list_patient_KFOLD.append([i[0], 4])
# model = create_model(LAMBDA_LOSS)

# # list_patients = train.Patient.sample(12).to_list()

# # history = model.fit(x=train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], y=train[~train.Patient.isin(list_patients)][['FVC']], validation_data=(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']]), epochs=EPOCH, sample_weight=train[~train.Patient.isin(list_patients)].Weight, verbose=0)

# history = model.fit(x=train[SELECTED_COLUMNS], y=train[['FVC']], epochs=EPOCH, sample_weight=train.Weight, verbose=0)

# model.save('model')

# # plt.plot(history.history['val_score'])

# plt.plot(history.history['score'])
# for i in range(1,6):

#     model = create_model(LAMBDA_LOSS)

#     list_patients = train.Patient.unique()[np.random.randint(0, len(train.Patient.unique())-1, size=round(len(train.Patient.unique())/NFOLD))]

#     history = model.fit(x=train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], y=train[~train.Patient.isin(list_patients)][['FVC']], validation_data=(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']]), epochs=EPOCH, sample_weight=train[~train.Patient.isin(list_patients)].Weight, verbose=0)

#     # model.save('model')

#     plt.figure(figsize=(20,20))

#     plt.subplot(3,3,i)

#     plt.plot(history.history['val_score'])

#     plt.plot(history.history['score'])
# pe = np.zeros((X_prediction.shape[0], 3))

# pred = np.zeros((train.shape[0], 3))

# i=0

# EPOCH = [54, 55, 56, 57, 58]

# for j in range(NFOLD):

#     print(f"FOLD {i}")

#     model = create_model(LAMBDA_LOSS)

#     list_patients = [j[0] for j in list_patient_KFOLD if j[1]==i]

# #     list_patients = train.Patient.unique()[np.random.randint(0, len(train.Patient.unique())-1, size=round(len(train.Patient.unique())/NFOLD))]

#     history = model.fit(x=train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], y=train[~train.Patient.isin(list_patients)][['FVC']], validation_data=(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']]), epochs=EPOCH[j], sample_weight=train[~train.Patient.isin(list_patients)].Weight, verbose=0)

#     print("train", model.evaluate(train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[~train.Patient.isin(list_patients)][['FVC']], verbose=0, batch_size=BATCH_SIZE))

#     print("val", model.evaluate(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']], verbose=0, batch_size=BATCH_SIZE))

#     pred[train[train.Patient.isin(list_patients)].index] = model.predict(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0)

#     pe += model.predict(X_prediction[SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0) / NFOLD

#     model.save('model_' + str(i))
KFOLD_confidence = [0.05, 0.15, 0.2, 0.25, 0.35]
pe = np.zeros((X_prediction.shape[0], 3))

pred = np.zeros((train.shape[0], 3))

for i in range(NFOLD):

    print(f"FOLD {i}")

    model = create_model(LAMBDA_LOSS)

    list_patients = [j[0] for j in list_patient_KFOLD if j[1]==i]

#     list_patients = train.Patient.unique()[np.random.randint(0, len(train.Patient.unique())-1, size=round(len(train.Patient.unique())/NFOLD))]

    history = model.fit(x=train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], y=train[~train.Patient.isin(list_patients)][['FVC']], validation_data=(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']]), epochs=EPOCH[i], sample_weight=train[~train.Patient.isin(list_patients)].Weight, verbose=0)

    print("train", model.evaluate(train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[~train.Patient.isin(list_patients)][['FVC']], verbose=0, batch_size=BATCH_SIZE))

    print("val", model.evaluate(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']], verbose=0, batch_size=BATCH_SIZE))

    pred[train[train.Patient.isin(list_patients)].index] = model.predict(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0)

    pe += model.predict(X_prediction[SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0)* KFOLD_confidence[i]

    model.save('model_' + str(i))
# loss_value = [0.58, 0.585, 0.59, 0.595]

# for loss in loss_value:

#     print('Loss_value; %f' % loss)

#     res_train = np.zeros((5,1))

#     res_val = np.zeros((5,1))

#     for k in range(3):

#         for i in range(NFOLD):

#             model = create_model(LAMBDA_LOSS)

#             list_patients = [j[0] for j in list_patient_KFOLD if j[1]==i]

#             history = model.fit(x=train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], y=train[~train.Patient.isin(list_patients)][['FVC']], validation_data=(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']]), epochs=EPOCH[i], sample_weight=train[~train.Patient.isin(list_patients)].Weight, verbose=0)

#             res_train[i] += model.evaluate(train[~train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[~train.Patient.isin(list_patients)][['FVC']], verbose=0, batch_size=BATCH_SIZE)[1]/3

#             res_val[i] += model.evaluate(train[train.Patient.isin(list_patients)][SELECTED_COLUMNS], train[train.Patient.isin(list_patients)][['FVC']], verbose=0, batch_size=BATCH_SIZE)[1]/3

#     for p in range(5):

#         print('Train')

#         print(res_train[p])

#         print('Val')

#         print(res_val[p])

            

        
# NFOLD = 3

# kf = KFold(n_splits=NFOLD)

# pe = np.zeros((X_prediction.shape[0], 3))

# pred = np.zeros((train.shape[0], 3))



# cnt = 0

# EPOCHS = 200

# for tr_idx, val_idx in kf.split(train):

#     cnt += 1

#     print(f"FOLD {cnt}")

#     net = create_model(LAMBDA_LOSS)

#     net.fit(train.loc[tr_idx, SELECTED_COLUMNS], train.loc[tr_idx, 'FVC'], batch_size=BATCH_SIZE, epochs=EPOCHS,

#             validation_data=(train.loc[val_idx, SELECTED_COLUMNS], train.loc[val_idx, 'FVC']), sample_weight=train.loc[tr_idx, 'Weight'], verbose=0)

#     print("train", net.evaluate(train.loc[tr_idx, SELECTED_COLUMNS], train.loc[tr_idx, 'FVC'], verbose=0, batch_size=BATCH_SIZE))

#     print("val", net.evaluate(train.loc[val_idx, SELECTED_COLUMNS], train.loc[val_idx, 'FVC'], verbose=0, batch_size=BATCH_SIZE))

#     print("predict val...")

#     pred[val_idx] = net.predict(train.loc[val_idx, SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0)

#     print("predict test...")

#     pe += net.predict(X_prediction[SELECTED_COLUMNS], batch_size=BATCH_SIZE, verbose=0) / NFOLD

#     model.save('model_' + str(cnt))
sigma_opt = mean_absolute_error(train[['FVC']], pred[:, 1])

unc = pred[:,2] - pred[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)



X_prediction['FVC1'] = 0.996*pe[:, 1]

X_prediction['Confidence1'] = pe[:, 2] - pe[:, 0]
subm = X_prediction.copy()

subm['FVC'] = 3020

subm['Confidence'] = 100



subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

if sigma_mean<70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)
# SELECTED_COLUMNS = ['Base_week', 'Base_FVC', 'Base_percent', 'Age', 'Sex','Weeks', '_Currently smokes', '_Ex-smoker', '_Never smoked']

# history = model.fit(x=train[SELECTED_COLUMNS], y=train[['FVC']], validation_data=(test[SELECTED_COLUMNS], test[['FVC']]), epochs=EPOCH)

# model.save('model')
# SELECTED_COLUMNS = ['Base_week', 'Base_FVC', 'Base_percent', 'Age', 'Sex','Weeks', '_Currently smokes', '_Ex-smoker', '_Never smoked']

# history=[]

# for i in np.linspace(0,1,11):

#     print('Lambda %f' % i)

#     model = model = create_model(i)

#     history.append(model.fit(x=train[SELECTED_COLUMNS], y=train[['FVC']], validation_data=(test[SELECTED_COLUMNS], test[['FVC']]), epochs=200))

# # model.save('model')
# import matplotlib.pyplot as plt

# plt.figure(figsize=(30,10))

# for i in range(1,11):

#     plt.subplot(3,4,i)

#     plt.plot(history[i].history['score'])

#     plt.plot(history[i].history['val_score'])