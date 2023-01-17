# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import copy
df=pd.read_csv("/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv")
df.head()
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# derived features
df['i_s']=np.sqrt(df['i_d']**2+df['i_q']**2)
df['u_s']=np.sqrt(df['u_d']**2+df['u_q']**2)
df['s']=1.5*df['i_s']*df['u_s']
target_features = ['pm', 'stator_tooth', 'stator_yoke', 'stator_winding']
del df['torque']
inputs=['ambient','coolant','u_d','u_q','motor_speed','i_d','i_q','i_s','u_s','s','profile_id']
target=['pm','stator_yoke','stator_tooth','stator_winding','profile_id']
#keep a load profile seperate from training
val_ind=df['profile_id']==20
#train index
tr_ind=df['profile_id']!=20

#seperate training data
df_tr=df[tr_ind]
#seperate test data
df_val=df[val_ind].reset_index()
del df_val['index']
#spans = [6360, 3360, 1320, 9480]  # these values correspond to cutoff-frequencies in terms of low pass filters, or half-life in terms of EWMAs, respectively
#spans=[600,2200,4200,8800]
spans=[500,2161,4000,8890]
#spans=[1900]
max_span = max(spans)
enriched_profiles = []
for p_id, p_df in df_tr.groupby(['profile_id']):
    target_df = p_df.loc[:, target_features].reset_index(drop=True)
    # pop out features we do not want to calculate the EWMA from
    p_df = p_df.drop(target_features + ['profile_id'], axis=1).reset_index(drop=True)
    
    # prepad with first values repeated until max span in order to get unbiased EWMA during first observations
    prepadding = pd.DataFrame(np.zeros((max_span, len(p_df.columns))),
                              columns=p_df.columns)
    temperature_cols = [c for c in ['ambient', 'coolant'] if c in df]
    prepadding.loc[:, temperature_cols] = p_df.loc[0, temperature_cols].values

    # prepad
    prepadded_df = pd.concat([prepadding, p_df], axis=0, ignore_index=True)
    ewma = pd.concat([prepadded_df.ewm(span=s).mean().rename(columns=lambda c: f'{c}_ewma_{s}') for s in spans], axis=1).astype(np.float32)
    ewma = ewma.iloc[max_span:, :].reset_index(drop=True)  # remove prepadding
    assert len(p_df) == len(ewma) == len(target_df), f'{len(p_df)}, {len(ewma)}, and {len(target_df)} do not match'
    new_p_df = pd.concat([p_df, ewma, target_df], axis=1)
    new_p_df['profile_id'] = p_id
    enriched_profiles.append(new_p_df.dropna())
enriched_df = pd.concat(enriched_profiles, axis=0, ignore_index=True)  

# normalize
#save p_id
p_ids = enriched_df['profile_id']
#p_ids = enriched_df.pop('profile_id')
scaler = StandardScaler()
enriched_df = pd.DataFrame(scaler.fit_transform(enriched_df), columns=enriched_df.columns)
enriched_df['profile_id']=p_ids
enriched_df.head()
plt.plot(enriched_df['i_d'][:10000])
plt.plot(enriched_df['i_d_ewma_9480'][:10000])
plt.plot(enriched_df['i_d_ewma_1320'][:10000])
plt.plot(enriched_df['i_d_ewma_6360'][:10000])
plt.plot(enriched_df['i_d_ewma_3360'][:10000])
#plt.plot(enriched_df['i_d_ewma_620'][:10000])

target_rg=enriched_df[target_features]
for name in target_features:
    del enriched_df[name]
print(target_rg.shape,enriched_df.shape)
targ=df_val[target_features]
#spans=[1900]
#spans = [6360, 3360, 1320, 9480]  # these values correspond to cutoff-frequencies in terms of low pass filters, or half-life in terms of EWMAs, respectively
#spans=[600,2200,4200,8800]
spans=[500,2161,4000,8890]
max_span = max(spans)
enriched_profiles = []
for p_id, p_df in df_val.groupby(['profile_id']):
    target_df = p_df.loc[:, target_features].reset_index(drop=True)
    # pop out features we do not want to calculate the EWMA from
    p_df = p_df.drop(target_features + ['profile_id'], axis=1).reset_index(drop=True)
    
    # prepad with first values repeated until max span in order to get unbiased EWMA during first observations
    prepadding = pd.DataFrame(np.zeros((max_span, len(p_df.columns))),
                              columns=p_df.columns)
    temperature_cols = [c for c in ['ambient', 'coolant'] if c in df]
    prepadding.loc[:, temperature_cols] = p_df.loc[0, temperature_cols].values

    # prepad
    prepadded_df = pd.concat([prepadding, p_df], axis=0, ignore_index=True)
    ewma = pd.concat([prepadded_df.ewm(span=s).mean().rename(columns=lambda c: f'{c}_ewma_{s}') for s in spans], axis=1).astype(np.float32)
    ewma = ewma.iloc[max_span:, :].reset_index(drop=True)  # remove prepadding
    assert len(p_df) == len(ewma) == len(target_df), f'{len(p_df)}, {len(ewma)}, and {len(target_df)} do not match'
    new_p_df = pd.concat([p_df, ewma, target_df], axis=1)
    new_p_df['profile_id'] = p_id
    enriched_profiles.append(new_p_df.dropna())
enriched_df_val = pd.concat(enriched_profiles, axis=0, ignore_index=True)  

# normalize
p_ids = enriched_df_val.pop('profile_id')
scaler = StandardScaler()
enriched_df_val = pd.DataFrame(scaler.fit_transform(enriched_df_val), columns=enriched_df_val.columns)
print(enriched_df_val.shape)
for name in target_features:
    del enriched_df_val[name]
print(targ.shape,enriched_df_val.shape)
scale_val={}
for name in targ.columns:
    if name!='profile_id':
        a=min(targ[name])
        b=max(targ[name])
        scale_val[name+"min"]=a
        scale_val[name+"max"]=b
        targ[name]=(targ[name]-a)/(b-a)

ols = LinearRegression(fit_intercept=False)
ols.fit(enriched_df,target_rg)
ols.score(enriched_df,target_rg)
def recon_predicted(y):
    k=0
    out_ls=['pm','stator_yoke','stator_tooth','stator_winding']
    range_ls=[100,20,100,20,110,20,125,20]
    for i,name in enumerate(out_ls):
        a,b=min(df[name]),max(df[name])
        mat_ls=[[a,1],[b,1]]
        A=np.array(mat_ls)
        mx,mn=range_ls[k],range_ls[k+1]
        B=np.array([mn,mx])
        X=np.linalg.inv(A) @ B
        #print(A,B,X)
        std,mean=X[0],X[1]
        k+=2
        y[:,i]=y[:,i]*std+mean
    return y
y=ols.predict(enriched_df)
y=recon_predicted(y)
df_tr[target_features]=recon_predicted(np.array(df_tr[target_features]))
def cal_map(y,y_hat):
    mse_arr=sum(abs(y-y_hat))/y.shape[0]
    return mse_arr
avg_map=[]
out_ls=['pm','stator_yoke','stator_tooth','stator_winding']
for i,name in enumerate(out_ls):
    map1=cal_map(y[:,i],df_tr[name])
    avg_map.append(map1)
print(avg_map,sum(avg_map)/4)
y=ols.predict(enriched_df_val)
y=recon_predicted(y)
df_val[target_features]=recon_predicted(np.array(df_val[target_features]))
def cal_map(y,y_hat):
    mse_arr=sum(abs(y-y_hat))/y.shape[0]
    return mse_arr
avg_map=[]
fig, big_axes = plt.subplots( figsize=(25, 4) , nrows=1, ncols=4, sharey=False) 
out_ls=['pm','stator_yoke','stator_tooth','stator_winding']
for i,name in enumerate(out_ls):
    map1=cal_map(y[:,i],df_val[name])
    avg_map.append(map1)
    title=name
    #ax = fig.add_subplot(1,4,i+1)
    #plt.title(title)
    
    #plt.plot(y[:,i],label='Predicted by Linear regressor')
    #plt.plot(df_val[name],label='Actual')
    big_axes[i].set_title(title)
    big_axes[i].plot(y[:,i],label='Linear regressor')
    big_axes[i].plot(df_val[name],label='Actual')
    k = int(len(df_val)> 4*3600) + 1
    ticks_loc=df_val[name].index.values[::k*3600]
    labels=[(y-ticks_loc[0])/2/3600 for y in ticks_loc]
    big_axes[i].set_xticklabels(labels)
    big_axes[i].set_xlabel("Time in hrs")
    big_axes[i].set_ylabel("Temperature")
    plt.legend()
plt.show()
print(avg_map,sum(avg_map)/4)
scale_val={}
def prepad_zero(window,out):
    #target
    df_y=enriched_df[out]
    #input
    df_x_=enriched_df.drop(target_features,axis=1).reset_index(drop=True)
    
    df_x=pd.DataFrame()
    print(df_x_.shape)
    #print(df_x_.head())
    for p_id, p_df in df_x_.groupby(['profile_id']):
        #remove profile_id column
        #p_df = p_df.drop(['profile_id'], axis=1).reset_index(drop=True)
        p_df=p_df.reset_index()
        p_df=p_df.drop(['index'],axis=1)
        prepadding = pd.DataFrame(np.zeros((window-1, len(p_df.columns))),
                                  columns=p_df.columns)
        #prepad with actual starting value of temperature and keep profile id same
        prepadding['ambient']=(p_df['ambient'][0])
        prepadding['coolant']=(p_df['coolant'][0])
        prepadding['profile_id']=p_id
        p_df=pd.concat([prepadding,p_df],ignore_index=True)
        
        # add to empty dataframe
        df_x=pd.concat([df_x,p_df],ignore_index=True)
    print("x",df_x.shape,"y",df_y.shape)    
    # scale target temperature to the (0,1) range
    scale_val={}
    for name in out:
        if name!='profile_id':
            a=min(df_y[name])
            b=max(df_y[name])
            scale_val[name+"min"]=a
            scale_val[name+"max"]=b
            df_y[name]=(df_y[name]-a)/(b-a)
    #print(df_x_train.shape,df_y.shape)
    
    return df_x,df_y
def seq_idwise(df_x_train,df_y,window,out_length):
    #make an aray of input varibale as a sequence
    profile=list(set(df_x_train['profile_id'].values))
    x=[]
    y=[]
    window=window
    for i,_id in enumerate(profile):
        p_df=df_x_train[df_x_train['profile_id']==_id]
        y_df=df_y[df_y['profile_id']==_id]
        
        # assert if these are not equal
        assert p_df.shape[0]-(window-1)==y_df.shape[0]
    
        #remove profile id column
        y_df =y_df.drop(['profile_id'], axis=1).reset_index(drop=True)
        p_df =p_df.drop(['profile_id'], axis=1).reset_index(drop=True)


        # convert all input data into a sequence of shape (p_df.shape[0],8,10)
        seq=[]
        # -7 is done becuase we need to loop over original profile_id readings
        for j in range(p_df.shape[0]-(window-1)):
            seq.append(np.array(p_df[j:window+j]).reshape(window,30))

        x.append(np.array(seq).reshape(p_df.shape[0]-(window-1),window,30))        

        # add all data with the shape (y_df.shape[0],out_length)
        y.append(np.array(y_df).reshape(y_df.shape[0],out_length))
        
        # assert if shape is not same
        assert y[i].shape[0]==x[i].shape[0], '{},{},{}'.format(i,y[i].shape[0],x[i].shape[0])
        
        #print(len(seq),y_df.shape)
    del df_x_train
    del df_y
    x_arr=np.concatenate(x)
    y_arr=np.concatenate(y)
    return x_arr,y_arr
from keras.models import Model
from keras.layers import Input, Dense, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
#from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras import backend as K
from tqdm import tqdm_notebook as tqdm
import pickle
import gc
from keras.callbacks import EarlyStopping

import keras.backend as K
import keras.layers
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
from typing import List, Tuple

import keras.optimizers as opts
import keras.regularizers as regularizers
def add_common_layers(z,dropout_rate):
    activation='relu'
    dropout_layer = keras.layers.AlphaDropout if activation == 'selu' else keras.layers.SpatialDropout1D
    batchnorm=True
    if batchnorm:
        z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Activation(activation)(z)
    z = dropout_layer(dropout_rate)(z)
    return z
def seperate_model(name,arch='res',kernel_size= 6,dilation_start_rate= 1,lr_rate=1.4e-4,n_layers=4,t_steps=32,nb_filters=121,
              reg_rate=1e-8,dropout_rate=0.29,batch_size=128,out_length=1):

    regs = {'kernel_regularizer': regularizers.l2(reg_rate),
            'bias_regularizer': regularizers.l2(reg_rate),
            'activity_regularizer': regularizers.l2(reg_rate)}
    x = keras.layers.Input(shape=(t_steps,50),batch_size=batch_size,name="Input_Measurements")
    y = x
    for i in range(n_layers):
        dilation_rate = dilation_start_rate * (2 ** i)
        if i % 2 == 0 and arch == 'res':  # every two layers
            shortcut = y

        y = keras.layers.Conv1D(nb_filters, kernel_size, padding='causal',
                          dilation_rate=dilation_rate,
                          activation=None,
                           **regs)(y)
        y = add_common_layers(y,dropout_rate)

        if i % 2 == 1 and arch == 'res':  # every two layers (anti-cyclic)
            shortcut = keras.layers.Conv1D(nb_filters, kernel_size=1,
                                     padding='causal',
                                     dilation_rate=dilation_rate,
                                     activation=None,
                                      **regs)(shortcut)
            y = keras.layers.add([shortcut, y])

    y = keras.layers.GlobalMaxPooling1D()(y)
    y = keras.layers.Dense(out_length)(y)
    opt=opts.Adam(lr=lr_rate)
    tcn = Model(inputs=x, outputs=y)
    tcn.compile(loss='mse', optimizer=opt,metrics=[keras.metrics.RootMeanSquaredError()])
    tcn.summary()
    return tcn
stator_target=['stator_yoke','stator_tooth','stator_winding','profile_id']
rotor_target=['pm','profile_id']
#names={'rotor':1, 'stator':3}
#names={'stator':3}
names={'rotor':1}
rotor_conf={'kernel_size': 2,'dilation_start_rate' :1,"lr_rate": 1e-4,"n_layers":2,"t_steps":33,"nb_filters":126,
              "reg_rate":1e-9,"dropout_rate":0.35,"batch_size":128,"out_length":1}
stator_conf={'kernel_size': 6,'dilation_start_rate' :1,"lr_rate": 1.4e-4,"n_layers":4,"t_steps":32,"nb_filters":121,
              "reg_rate":1e-8,"dropout_rate":0.29,"batch_size":128,"out_length":3}
models_tcn={}
batch_size=128
for name in names:
    if name=='rotor':
        conf=rotor_conf
        out=rotor_target
    else:
        conf=stator_conf
        out=stator_target
    mod=seperate_model(name,arch='res',**conf)
    df_x_tr,df_y_tr=prepad_zero(conf['t_steps'],out)
    del enriched_df
    x_tr,y_tr=seq_idwise(df_x_tr,df_y_tr,conf['t_steps'],names[name])
    print(x_tr.shape,y_tr.shape)
    history=mod.fit(x_tr[:953984],y_tr[:953984],batch_size=128,epochs=10,steps_per_epoch=953984/batch_size,verbose=1,shuffle='batch')
    plt.plot(history.history['loss'][1:],label=name)
    plt.legend()
    models_tcn[name]=mod
    mod.save('./{}.h5'.format(name))

def recon_rotor(y):
    k=0
    out_ls=['pm']
    range_ls=[100,20]
    for i,name in enumerate(out_ls):
        a,b=min(df[name]),max(df[name])
        mat_ls=[[a,1],[b,1]]
        A=np.array(mat_ls)
        mx,mn=range_ls[k],range_ls[k+1]
        B=np.array([mn,mx])
        X=np.linalg.inv(A) @ B
        #print(A,B,X)
        std,mean=X[0],X[1]
        k+=2
        y=y*(b-a)+a
        y=y*std+mean
    return y
def recon_stator(y):
    k=0
    out_ls=['stator_yoke','stator_tooth','stator_winding']
    range_ls=[100,20,110,20,125,20]
    for i,name in enumerate(out_ls):
        a,b=min(df[name]),max(df[name])
        mat_ls=[[a,1],[b,1]]
        A=np.array(mat_ls)
        mx,mn=range_ls[k],range_ls[k+1]
        B=np.array([mn,mx])
        X=np.linalg.inv(A) @ B
        #print(A,B,X)
        std,mean=X[0],X[1]
        k+=2
        y[:,i]=y[:,i]*(b-a)+a
        y[:,i]=y[:,i]*std+mean
    return y
def cal_map(y,y_hat):
    map_arr=sum(abs(y-y_hat))/y.shape[0]
    return map_arr
def generate_batches(df_x,df_y,batch_size,window):
    samples_per_epoch = df_y.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    p_count=0
    profile=list(set(df_x['profile_id'].values))
    #df_x=df_x.drop(['profile_id'], axis=1).reset_index(drop=True)
    #df_y=df_y.drop(['profile_id'], axis=1).reset_index(drop=True)
    k=0
    
    while(True):
        #form batches groupby on profile id as every profile is prepaded with the zeros 
        X_list=[]
        
        pd_x=df_x[df_x['profile_id']==profile[k]]
        pd_y=df_y[df_y['profile_id']==profile[k]]
        pd_x=pd_x.drop(['profile_id'], axis=1).reset_index(drop=True)
        pd_y=pd_y.drop(['profile_id'], axis=1).reset_index(drop=True)
        assert pd_y.shape[0]==pd_x.shape[0]-window+1
        #add extra window length
        pd_x_=pd_x[batch_size*p_count:batch_size*(p_count+1)+window-1]
        #assert pd_y.shape[0]==pd_x_.shape[0]-window+1
        if pd_x_.shape[0]==batch_size+window-1:
            for j in range(batch_size):
                X_list.append(np.array(pd_x_[j:j+window]).astype("float32"))
        #convert the list to array
            """try:
            X_batch=np.stack(X_list)
            #.reshape(128,window,50)
            except:
            X_batch=np.stack(X_list)#.reshape(-1,window,50)
            k+=1
            start_flag=True"""
            
            X_batch=np.stack(X_list)
            y_batch = np.array(pd_y[batch_size*p_count:batch_size*(p_count+1)]).astype('float32')
            p_count+=1
            assert X_batch.shape[0]==y_batch.shape[0]
            yeild_flag=True
        else:
            k+=1
            p_count=0
            yeild_flag=False
        
        #print("profile:",k," main counter:",counter," p_counter ",p_count," ",X_batch.shape,y_batch.shape)
        if yeild_flag==True:
            yield X_batch,y_batch
        #increase counter with every batch  
        counter += 1
        #restart counter to yeild data in the next epoch as well
        #print(counter)

        if counter >= number_of_batches-1:
            counter = 0
            k=0
            p_count=0
            #print('New epoch')
names={'rotor':1, 'stator':3}
rotor_conf={'kernel_size': 2,'dilation_start_rate' :1,"lr_rate": 1e-4,"n_layers":2,"t_steps":33,"nb_filters":126,
              "reg_rate":1e-8,"dropout_rate":0.35,"batch_size":128,"out_length":names['rotor']}
stator_conf={'kernel_size': 6,'dilation_start_rate' :1,"lr_rate": 1.4e-4,"n_layers":4,"t_steps":32,"nb_filters":121,
              "reg_rate":1e-9,"dropout_rate":0.29,"batch_size":128,"out_length":names['stator']}
models_tcn={}
batch_size=128
names=['rotor']
for name in names:
    if name=='rotor':
        conf=rotor_conf
        out=rotor_target
    else:
        conf=stator_conf
        out=stator_target
    mod=seperate_model(name,arch='res',**conf)
    df_x_tr,df_y_tr=prepad_zero(conf['t_steps'],out)
    
    gen=generate_batches(df_x_tr,df_y_tr,batch_size,conf['t_steps'])
    history=mod.fit_generator(generator=gen,
                steps_per_epoch=df_y_tr.shape[0]/batch_size,
                epochs=10,
                verbose=1,
                shuffle=False)
    plt.plot(history.history['loss'][0:],label=name)
    plt.legend()
    models_tcn[name]=mod
    mod.save('./{}.h5'.format(name))
names={'rotor':1, 'stator':3}
rotor_conf={'kernel_size': 2,'dilation_start_rate' :1,"lr_rate": 1e-4,"n_layers":2,"t_steps":33,"nb_filters":126,
              "reg_rate":1e-8,"dropout_rate":0.35,"batch_size":128,"out_length":names['rotor']}
stator_conf={'kernel_size': 6,'dilation_start_rate' :1,"lr_rate": 1.4e-4,"n_layers":4,"t_steps":32,"nb_filters":121,
              "reg_rate":1e-9,"dropout_rate":0.29,"batch_size":128,"out_length":names['stator']}
models_tcn={}
batch_size=128
names=['stator']
for name in names:
    if name=='rotor':
        conf=rotor_conf
        out=rotor_target
    else:
        conf=stator_conf
        out=stator_target
    mod=seperate_model(name,arch='res',**conf)
    df_x_tr,df_y_tr=prepad_zero(conf['t_steps'],out)
    
    gen=generate_batches(df_x_tr,df_y_tr,batch_size,conf['t_steps'])
    history=mod.fit_generator(generator=gen,
                steps_per_epoch=df_y_tr.shape[0]/batch_size,
                epochs=10,
                verbose=1,
                shuffle=False)
    plt.plot(history.history['loss'][0:],label=name)
    plt.legend()
    models_tcn[name]=mod
    mod.save('./{}.h5'.format(name))
X_list=[]
window=33
tr_targ=enriched_df[target_features]
for name in tr_targ.columns:
    a=min(tr_targ[name])
    b=max(tr_targ[name])
    tr_targ[name]=(tr_targ[name]-a)/(b-a)
for name in target_features:
    del enriched_df[name]
del enriched_df['profile_id']
print(tr_targ.shape,enriched_df.shape)
for j in range(100000):
    X_list.append(np.array(enriched_df[j:j+window]).astype("float32").reshape(window,50))

X_batch=np.asarray(X_list)#.reshape(128,8,10)
y_batch = np.array(tr_targ[window-1:100000*(1)+window-1]).astype('float32')
assert X_batch.shape[0]==y_batch.shape[0], "length of input and target not match. X_batch {} y_batch {}".format(X_batch.shape[0],y_batch.shape[0])
assert X_batch.shape[1]==window," seq not made correctly"
y_pred=models_tcn['rotor'].predict(X_batch)
y_batch=recon_rotor(y_batch[:,0])
y_pred=recon_rotor(y_pred)
avg_map_pm=cal_map(y_pred.squeeze(),y_batch)
plt.plot(y_pred,label='Predicted by TCN')
plt.plot(y_batch,label='Actual')
plt.legend()
plt.ylabel('Temperature(C)')
plt.xlabel("Time (sec)")
plt.title("pm")
plt.show()
print("train rotor error ",avg_map_pm,"K")
X_list=[]
window=33

for j in range(40000):
    X_list.append(np.array(enriched_df_val[j:j+window]).astype("float32").reshape(window,50))

X_batch=np.asarray(X_list)#.reshape(128,8,10)
y_batch = np.array(targ[window-1:40000*(1)+window-1]).astype('float32')
assert X_batch.shape[0]==y_batch.shape[0], "length of input and target not match. X_batch {} y_batch {}".format(X_batch.shape[0],y_batch.shape[0])
assert X_batch.shape[1]==window," seq not made correctly"
y_pred=models_tcn['rotor'].predict(X_batch)
y_batch=recon_rotor(y_batch[:,0])
y_pred=recon_rotor(y_pred)
avg_map_pm=cal_map(y_pred.squeeze(),y_batch)
plt.plot(y_pred,label='Predicted by TCN')
plt.plot(y_batch,label='Actual')
plt.legend()
plt.ylabel('Temperature(C)')
plt.xlabel("Time (sec)")
plt.title("pm")
plt.show()
avg_map_pm
X_list=[]
window=32
trgt=['stator_yoke','stator_tooth','stator_winding']
tr_targ=enriched_df[target_features]
for name in tr_targ.columns:
    a=min(tr_targ[name])
    b=max(tr_targ[name])
    tr_targ[name]=(tr_targ[name]-a)/(b-a)

for name in target_features:
    del enriched_df[name]
del enriched_df['profile_id']
print(tr_targ.shape,enriched_df.shape)
for j in range(100000):
    X_list.append(np.array(enriched_df[j:j+window]).astype("float32").reshape(window,50))

X_batch=np.asarray(X_list)#.reshape(128,8,10)

y_batch = np.array(tr_targ[window-1:100000*(1)+window-1]).astype('float32')

assert X_batch.shape[0]==y_batch.shape[0], "length of input and target not match. X_batch {} y_batch {}".format(X_batch.shape[0],y_batch.shape[0])
assert X_batch.shape[1]==window," seq not made correctly"
y_pred=models_tcn['stator'].predict(X_batch)
y_batch=recon_stator(y_batch[:,1:])
y_pred=recon_stator(y_pred)
avg_map=cal_map(y_batch,y_pred)
for i,name in enumerate(trgt):
    plt.plot(y_pred[:,i],label='Predicted by TCN')
    plt.plot(y_batch[:,i],label='Actual')
    plt.title(name)
    plt.legend()
    plt.show()
print("train error ",avg_map," K", "Average stator error ",sum(avg_map)/3,"K") 
X_list=[]
window=32
trgt=['stator_yoke','stator_tooth','stator_winding']
"""df_x=df_x_tr[df_x_tr['profile_id']==65]
df_y=df_y_tr[df_y_tr['profile_id']==65]
df_x=df_x.drop('profile_id',axis=1).reset_index(drop=True)
df_y=df_y.drop('profile_id',axis=1).reset_index(drop=True)
"""
for j in range(40000):
    X_list.append(np.array(enriched_df_val[j:j+window]).astype("float32").reshape(window,50))

X_batch=np.asarray(X_list)#.reshape(128,8,10)
#y_batch = np.array(df_y[window-1:30000*(1)+window-1]).astype('float32')
y_batch = np.array(targ[window-1:40000*(1)+window-1]).astype('float32')
assert X_batch.shape[0]==y_batch.shape[0], "length of input and target not match. X_batch {} y_batch {}".format(X_batch.shape[0],y_batch.shape[0])
assert X_batch.shape[1]==window," seq not made correctly"
y_pred=models_tcn['stator'].predict(X_batch)
y_batch=recon_stator(y_batch[:,1:])
y_pred=recon_stator(y_pred)
avg_map=cal_map(y_batch,y_pred)
for i,name in enumerate(trgt):
    plt.plot(y_pred[:,i],label='Predicted by TCN')
    plt.plot(y_batch[:,i],label='Actual')
    plt.title(name)
    plt.legend()
    plt.show()
print(avg_map)

X_list=[]
window=32
trgt=['stator_yoke','stator_tooth','stator_winding']

for j in range(40000):
    X_list.append(np.array(enriched_df_val[j:j+window]).astype("float32").reshape(window,50))

X_batch=np.asarray(X_list)#.reshape(128,8,10)
y_batch = np.array(targ[window-1:40000*(1)+window-1]).astype('float32')
assert X_batch.shape[0]==y_batch.shape[0], "length of input and target not match. X_batch {} y_batch {}".format(X_batch.shape[0],y_batch.shape[0])
assert X_batch.shape[1]==window," seq not made correctly"
y_pred=models_tcn['stator'].predict(X_batch)

for i,name in enumerate(trgt):
    plt.plot(y_pred[:,i],label='Predicted by TCN')
    plt.plot(y_batch[:,i],label='Actual')
    plt.title(name)
    plt.legend()
    plt.show()

