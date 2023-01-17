!pip install fastai2 -q
import os

import numpy as np

import pandas as pd



from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error



import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff



import missingno as msno

import tensorflow.keras.backend as K



import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Lambda



import matplotlib.pyplot as plt

%matplotlib inline
from fastai2.basics import *

from fastai2.callback.all import *

from fastai2.vision.all import *

from fastai2.medical.imaging import *



import pydicom
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train_df.head()
plt.subplot(121)

msno.bar(train_df)

plt.title("Missing values in Training Data", fontsize = 20, color = 'Red')



plt.subplot(122)

msno.bar(test_df)

plt.title("Missing values in Test Data", fontsize = 20, color = 'Blue')



plt.show()
print(f'Total unique patients are {train_df.Patient.nunique()} out of total {len(train_df.Patient)} patients')
go.Figure(go.Pie(labels = train_df.Sex.value_counts().keys().tolist(), 

                       values = train_df.Sex.value_counts().values.tolist(), 

                       marker = dict(colors=['red']), hoverinfo = "value", pull=[0, 0.1]), 

          layout = go.Layout(title = {'text':"Gender Distribution", 'x':0.5}, font=dict(family="Courier New, monospace",

                                                                                                size=18,

                                                                                                color="RebeccaPurple")))
go.Figure(go.Pie(labels = train_df.SmokingStatus.value_counts().keys().tolist(), 

                       values = train_df.SmokingStatus.value_counts().values.tolist(), 

                       marker = dict(colors=['pink', 'blue', 'purple']), hoverinfo = "value", hole = 0.3), 

          layout = go.Layout(title = {'text':"Smoking Status", 'x':0.425}, font=dict(family="Courier New, monospace",

                                                                                                size=18,

                                                                                                color="RebeccaPurple")))
train_df.groupby('Sex')['SmokingStatus'].value_counts()
fig = go.Figure(data=[

    go.Bar(name='Smoker', x= train_df.Sex.unique(), y = [train_df.groupby('Sex')['SmokingStatus'].value_counts().values[5],

                                                         train_df.groupby('Sex')['SmokingStatus'].value_counts().values[2]]),

    go.Bar(name='Non-Smoker', x= train_df.Sex.unique(), y = [train_df.groupby('Sex')['SmokingStatus'].value_counts().values[4],

                                                             train_df.groupby('Sex')['SmokingStatus'].value_counts().values[0]]),

    go.Bar(name='Ex-Smoker', x= train_df.Sex.unique(), y = [train_df.groupby('Sex')['SmokingStatus'].value_counts().values[3],

                                                             train_df.groupby('Sex')['SmokingStatus'].value_counts().values[1]]),

])

fig.update_layout(title = {'text':"Smoking Distribution by Sex", 'x':0.5}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),

                  barmode ='group')

fig.show()
count_df = pd.DataFrame(train_df['Patient'].value_counts())

count_df = count_df.reset_index()

count_df.rename(columns = {'index':'Patient ID', 'Patient':'No of Images'}, inplace = True)



fig = px.bar(count_df, x='Patient ID',y ='No of Images',color='No of Images')

fig.update_xaxes(showticklabels=False)

fig.update_layout(title = {'text':"Distribution of Images per Patient", 'x':0.5}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()
dicom_ids = os.listdir('../input/osic-pulmonary-fibrosis-progression/train/')

patient_sizes = [len(os.listdir('../input/osic-pulmonary-fibrosis-progression/train/' + d)) for d in dicom_ids]

dicom_df = pd.DataFrame({'Dicom_ID':dicom_ids, 'Dicom Files':patient_sizes})



fig = px.bar(dicom_df, x='Dicom_ID',y ='Dicom Files',color='Dicom Files')

fig.update_xaxes(showticklabels=False)

fig.update_layout(title = {'text':"Distribution of Dicom Files per Dicom ID", 'x':0.5}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()
fig = ff.create_distplot([train_df.Age.values], ['Age'], colors = ['red'])

fig.update_layout(title = {'text':"Age Distribution", 'x':0.5}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()



fig = ff.create_distplot([train_df.Weeks.values], ['Weeks'], colors = ['blue'])

fig.update_layout(title = {'text':"Weeks Distribution", 'x':0.5}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()
fig = ff.create_distplot([train_df.Percent.values], ['Percent'], colors = ['purple'])

fig.update_layout(title = {'text':"Percent Distribution", 'x':0.5}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()



fig = ff.create_distplot([train_df.FVC.values], ['FVC'], colors = ['green'])

fig.update_layout(title = {'text':"FVC Distribution", 'x':0.5}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()
fig = px.histogram(train_df, x='Age', color='SmokingStatus', marginal="box", 

                   color_discrete_map={'Ex-smoker':'green','Never smoked':'light green','Currently smokes':'orange'})

fig.update_traces(marker_line_color='cyan',marker_line_width=1, opacity=0.8)

fig.update_layout(title = {'text':"Smoking Status by Age", 'x':0.4}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()
fig = px.histogram(train_df, x='Age', color='Sex',marginal="box", color_discrete_map={'Male':'blue','Female':'light green'})

fig.update_traces(marker_line_color='cyan',marker_line_width=1, opacity=0.8)

fig.update_layout(title = {'text':"Sex Distribution by Age", 'x':0.45}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()
fig = px.histogram(train_df, x='FVC', color='Sex', marginal="rug", 

                   color_discrete_map={'Male':'DarkKhaki','Female':'MediumSpringGreen'})

fig.update_traces(marker_line_color='LightSlateGrey',marker_line_width=1, opacity=0.8)

fig.update_layout(title = {'text':"Gender Distribution in FVC", 'x':0.45}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()
fig = px.histogram(train_df, x='FVC', color='SmokingStatus', marginal="box",

                   color_discrete_map={'Ex-smoker':'#393E46','Never smoked':'MediumTurquoise','Currently smokes':'Linen'})

fig.update_traces(marker_line_color = 'black',marker_line_width = 1, opacity = 0.8)

fig.update_layout(title = {'text':"SmokingStatus Distribution in FVC", 'x':0.45}, 

                  font = dict(family="Courier New, monospace", size=18, color="RebeccaPurple"))

fig.show()
source_path = Path('../input/osic-pulmonary-fibrosis-progression')

source_files = os.listdir(source_path)

print(source_files)
train_path = source_path/'train'

train_files = get_dicom_files(train_path)

train_files
dicom_img = dcmread(train_files[0])

dicom_img
dicom_img.show()
tensor_dicom = pixels(dicom_img) #convert into tensor



print(f'RescaleIntercept: {dicom_img.RescaleIntercept:1f}\nRescaleSlope: {dicom_img.RescaleSlope:1f}\nMax pixel: '

      f'{tensor_dicom.max()}\nMin pixel: {tensor_dicom.min()}\nShape: {tensor_dicom.shape}')
tensor_dicom_scaled = scaled_px(dicom_img)

plt.hist(tensor_dicom_scaled.flatten(), color='c')
# Viewing Cancellous Bone Area

dicom_img.show(min_px = 300, max_px = 400, figsize=(10, 10))
# Fat Area

dicom_img.show(min_px = -120, max_px = -90, figsize=(10, 10))
# Water Based Area

dicom_img.show(max_px=None, min_px=0, figsize=(10, 10))
# Air Based Area

dicom_img.show(max_px=None, min_px=-1000, figsize=(10, 10))
train_df.shape
train_df[train_df.duplicated(subset = ['Patient','Weeks'])]
train_df.drop_duplicates(keep=False, inplace = True, subset=['Patient','Weeks'])
submission_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
submission_df
temp_sub_df = submission_df['Patient_Week'].str.split('_', expand = True)

temp_sub_df.rename(columns = {0: 'Patient', 1: 'Weeks'}, inplace = True)
submission_df = pd.concat([submission_df, temp_sub_df], axis = 1)

submission_df = submission_df[['Patient','Weeks','Confidence','Patient_Week']]
test_df.head()
submission_df = submission_df.merge(test_df.drop('Weeks', axis = 1), on = 'Patient')
train_df['data_type'] = 'Train'

test_df['data_type'] = 'Val'

submission_df['data_type'] = 'Test'

combined_df = train_df.append([test_df, submission_df])
data_type = ['Train', 'Val', 'Test']

for type in data_type:

    data = combined_df.query("data_type == @type")

    print(type, "shape in combined data is ", data.shape)
# Minimum Week for each patient

combined_df['Min_Weeks'] = combined_df['Weeks']

combined_df.loc[combined_df.data_type == 'Test','Min_Weeks'] = np.nan

combined_df['Min_Weeks'] = combined_df.groupby('Patient')['Min_Weeks'].transform('min')
base = combined_df.loc[combined_df.Weeks == combined_df.Min_Weeks]

base = base[['Patient','FVC']].rename(columns = {'FVC':'min_FVC'})

base.drop_duplicates(keep = 'first', inplace = True, subset = ['Patient'])
combined_df.Weeks = combined_df.Weeks.astype(int)

combined_df.Min_Weeks = combined_df.Min_Weeks.astype(float)
combined_df = combined_df.merge(base, on='Patient', how='left')

combined_df['Deviation_Weeks'] = combined_df['Weeks'] - combined_df['Min_Weeks']

del base
combined_df = pd.concat([combined_df, pd.get_dummies(combined_df[['Sex','SmokingStatus']])], axis = 1)
scaler = MinMaxScaler()

scaled = pd.DataFrame(scaler.fit_transform(combined_df[['Age','Percent','min_FVC','Deviation_Weeks']]), 

                      columns = ['scaled_Age', 'scaled_Percent', 'scaled_FVC', 'scaled_Deviation_Weeks'])

combined_df = pd.concat([combined_df, scaled], axis = 1)
combined_df
feature_columns = ['Sex_Male','Sex_Female','SmokingStatus_Ex-smoker','SmokingStatus_Never smoked','SmokingStatus_Currently smokes',

                   'scaled_Age','scaled_Percent','scaled_Deviation_Weeks','scaled_FVC']
train_df = combined_df.loc[combined_df.data_type == 'Train']

test_df = combined_df.loc[combined_df.data_type == 'Val']

submission_df = combined_df.loc[combined_df.data_type == 'Test']

del combined_df
train_df.shape, test_df.shape, submission_df.shape
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")



def score(y_true, y_pred):

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return K.mean(metric)



def qloss(y_true, y_pred):

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)



def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss



def make_model():

    x1 = Input((9,), name="Patient")

    x2 = Dense(100, activation="relu", name="d1")(x1)

    x3 = Dense(100, activation="relu", name="d2")(x2)

    

    p1 = Dense(3, activation="relu", name="p1")(x3)

    p2 = Dense(3, activation="relu", name="p2")(x3)

    

    preds = Lambda(lambda x3: x3[0] + tf.cumsum(x3[1], axis=1), 

                     name="preds")([p1, p2])

    

    model = Model(x1, preds, name="CNN")

   

    model.compile(loss = mloss(0.8), optimizer = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.005,

                                                                          amsgrad=False), metrics=[score])

    return model
model = make_model()

print(model.summary())

print(model.count_params())
y = train_df['FVC'].values

z = train_df[feature_columns].values

sub = submission_df[feature_columns].values

pe = np.zeros((sub.shape[0], 3))

pred = np.zeros((z.shape[0], 3))
NFOLD = 5

BATCH_SIZE=128

kf = KFold(n_splits=NFOLD)
%%time

cnt = 0

for tr_idx, val_idx in kf.split(z):

    cnt += 1

    print(f"FOLD {cnt}")

    model = make_model()

    model.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=800, 

            validation_data=(z[val_idx], y[val_idx]), verbose=0) #

    print("train", model.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))

    print("val", model.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))

    print("predict val...")

    pred[val_idx] = model.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)

    print("predict test...")

    pe += model.predict(sub, batch_size=BATCH_SIZE, verbose=0) / NFOLD
sigma_opt = mean_absolute_error(y, pred[:, 1])

unc = pred[:,2] - pred[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)
idxs = np.random.randint(0, y.shape[0], 100)

plt.plot(y[idxs], label="ground truth")

plt.plot(pred[idxs, 0], label="q25")

plt.plot(pred[idxs, 1], label="q50")

plt.plot(pred[idxs, 2], label="q75")

plt.legend(loc="best")

plt.show()
print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())
plt.hist(unc)

plt.title("uncertainty in prediction")

plt.show()
submission_df.head()
pe[:, 1]
submission_df['FVC1'] = pe[:, 1]

submission_df['Confidence1'] = pe[:, 2] - pe[:, 0]
subm = submission_df[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
subm.loc[~subm.FVC1.isnull()].head(10)
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

if sigma_mean<70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']
subm.head()
subm.describe().T
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)