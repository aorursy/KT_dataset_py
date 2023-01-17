
import os 
import gc
import sys
sys.path.append('../input/plasticc-extra')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import pandas as pd 
import time
import pdb
from tqdm import tqdm
from keras.models import Model
from keras import optimizers
from keras import layers
from keras.layers import Activation,Flatten, Dense, Dropout,Conv1D, GlobalMaxPooling1D,LeakyReLU,BatchNormalization,Input,ReLU
from keras.layers.merge import concatenate
import keras
from keras.activations import sigmoid, softmax
from generators import load_process,prepare_process
from generators import calculate_inputs
import tensorflow as tf
import pickle
from keras import regularizers
from keras.utils import to_categorical
import concurrent.futures
from multiprocessing import Process, Queue, current_process, freeze_support
#make wider graphs

plt.figure(figsize=(12,5))

# the following two lines are for changing imported functions, and not needing to restart kernel to use their updated version
%load_ext autoreload
%autoreload 2
%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

DATA_DIR = '../input/PLAsTiCC-2018/'
my_DATA_DIR='../input/plasticc-extra/'
color_map = {0:'g', 1:'r', 2:'c', 3:'m', 4:'y', 5:'k'}
galactic_targets = np.array([6,16,53,65,92])
extragalactic_targets = np.array([15,42,52,62,64,67,88,90,95])
kw = [1,1,1,1,1,2,1,1,1,2,1,1,1,1]
real_targets = np.append(galactic_targets,extragalactic_targets)
Targets = np.append(real_targets,99)

galactic=np.array([1,0,1,0,0,1,0,0,1,0,0,0,1,0])            
ex_galactic = 1-galactic
num_samples=128
#################################################################
#  Running Params
################################################################
test_train_switch=None      # Values : 'Train', 'Test', None (if none, use on training set to calculate CV)
Submit_name=None   # if not None will submit
meta_type='Nyanp'           # meta features type: None, 'Mamas', 'Nyanp' 
folds=range(4)
max_workers=3               # for multiprocessing
##########################################
# Test Parameters
weight_ext='best'        # traing weight named: model16fr{0}{1}_{2}.weights (if training set weights_ext=post_weights_ext)
                         # {0} - number of extraf features 4 if not added features, 19 - added nyanps, 20 - added mamas
                         # {1} - pre_weights_ext
                         # {2} -  fold number
submit_name=None     # Submit file name. if None - don't submit
##########################################
# Training parameters
pre_weight_ext=None      # if not None - use pre trasined weights
post_weight_ext='a'      # must not be None - used for naming training result
training_trails = 4
aug1={'shift':32, 'skew':0.05,'sample':0.85,'noise':0.3} #Augmentation values for corse training
aug2={'shift':128, 'skew':0.05,'sample':0.85,'noise':0.3} #Augmentation values for fine training
                                                            # shift: control cyclic shift
                                                            # skew: every channel is multiplied by (1+rand.norm(skew))
                                                            # sample: determin how which precentage of the data set would be used
                                                            #         the other samples will be deleted
                                                            # noise: added noise as proportion of flux_err
# learning rates for corse training
lr_corse=np.array([1e-3,1e-3,1e-3,3e-4,3e-4,3e-4,1e-4,1e-4,1e-4,3e-4,3e-4,3e-4,1e-4,1e-4,1e-4,3e-5,3e-5,3e-5,
             1e-3,3e-4,1e-4,3e-5,3e-5,3e-5,1e-5,1e-5,1e-5,3e-6,3e-6,3e-6,1e-5,1e-5,1e-5,3e-6,
                                     3e-6,3e-6,1e-5,1e-5,1e-5,3e-6,3e-6,3e-6,3e-6,3e-6,3e-6]) 
# learning rates for fine training
lr_fine=np.array([3e-6,3e-6,3e-6,3e-5,3e-6,3e-6,3e-5,3e-6,3e-4,3e-5,3e-6,3e-6,3e-5,3e-6,3e-6,3e-5,3e-6,3e-6])


df_timeseries=pd.read_csv(DATA_DIR+'training_set.csv')
df_training_meta=[]
df_validation_meta=[]
for i in folds:
    df_training_meta.append(pd.read_csv('{}training_meta{}.csv'.format(my_DATA_DIR,i)).sort_values('object_id').reset_index(drop=True))
    df_validation_meta.append(pd.read_csv('{}validation_meta{}.csv'.format(my_DATA_DIR,i)).sort_values('object_id').reset_index(drop=True))

training_meta=pd.read_csv(DATA_DIR+'training_set_metadata.csv')
tar_dict=dict(zip(real_targets,range(len(real_targets))))
ty=to_categorical(training_meta.target.map(tar_dict))
wtable=ty.sum(axis=0)/ty.shape[0]
wtable
wtable=wtable/kw
wtable=wtable/wtable.sum()
wtable

def mywloss(y_true,y_pred):
    
    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.log(yc),axis=0)/wtable))
    return loss
def score(y_pred,y_true):
    y=np.clip(y_pred,1e-3,1)
    for i in range(y.shape[0]):
        y[i,...]=y[i,...]/y[i,...].sum()
    return -(np.mean(np.mean(y_true*np.log(y),axis=0)/wtable))
# This layer if forcing galactic/extra galactic object to output 0

from keras import backend as K
from keras.engine.topology import Layer

class MySwitch(Layer):

    def __init__(self ,split,**kwargs):
        self.split=split
        super(MySwitch, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert self.split<=input_shape[1][-1]
        self.nsplit=input_shape[1][-1]
        # Create a trainable weight variable for this layer.
        super(MySwitch, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        x0=tf.equal(x[0],tf.zeros_like(x[0]))
        x1=tf.logical_not(x0)
        xt0=tf.tile(x0,[1,self.split])
        xt1=tf.tile(x1,[1,self.nsplit-self.split])
        xf=tf.concat([xt0,xt1],axis=-1)
        return tf.where(xf,x[1],-1e4*tf.ones_like(x[1]))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_b[0],shape_b[1])

def build_model16():
    input_timeseries = Input(shape=(num_samples, 6,),name='input_timeseries')
    input_timeseries0 = Input(shape=(num_samples, 6,),name='input_timeseries0')
    input_timeseriese = Input(shape=(num_samples, 6,),name='input_timeseriese')
    input_meta = Input(shape=(6,),name='input_meta')
    input_gal = Input(shape=(1,),name='input_gal')
    _series=concatenate([input_timeseries,input_timeseries0,input_timeseriese])
    x = Conv1D(256,8,padding='same',name='Conv1')(_series)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(256,5,padding='same',name='Conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(256,3,padding='same',name='Conv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = GlobalMaxPooling1D()(x)
    x1 = Dense(16,activation='relu',name='dense0')(input_meta)
    x1 = Dense(32,activation='relu',name='dense1')(x1)
    xc = concatenate([x,x1],name='concat')
    x = Dense(256,activation='relu',name='features')(xc)
    x = Dense(real_targets.shape[0],name='bout')(x)
    x = MySwitch(galactic_targets.shape[0])([input_gal,x])
    out = Activation('softmax',name='out')(x)
    model=Model([input_timeseries,input_timeseries0,input_timeseriese,input_meta,input_gal],out)
    return model
build_model16().summary()
# build MLP for extra data and concat to CNN output 
def build_top(ntop,convout,n_first=64):
    input_meta = Input(shape=(ntop,),name='input_meta')
    input_gal = Input(shape=(1,),name='input_gal')
    x = BatchNormalization()(convout)
    x = LeakyReLU(alpha=0.1)(x)
    x = GlobalMaxPooling1D()(x)
    x1 = Dense(n_first,activation='relu',name='dense0')(input_meta)
    x1 = Dropout(0.2)(x1)  #was 0.2
    x1 = BatchNormalization()(x1)
    x1 = Dense(64,activation='relu',name='dense2')(x1)
    xc = concatenate([x,x1],name='concat')
    x = Dense(256,activation='relu',name='features')(xc)
    x = Dense(real_targets.shape[0],name='bout')(x)
    x = MySwitch(galactic_targets.shape[0])([input_gal,x])
    out = Activation('softmax',name='out')(x)
    return [input_meta,input_gal],out

# Change the Original CNN extra data MLP
def model_retop(model,ntop,last_conv_layer_name='Conv5',n_first=64):
    convout = model.get_layer(name=last_conv_layer_name).output
    inputs=[]
    for inp in model.inputs:
        if 'timeseries' in inp.name:
            inputs.append(inp)
    dinputs,out=build_top(ntop,convout,n_first=n_first)
    inputs=inputs+dinputs
    topmodel=Model(inputs,out)
    return topmodel
model_retop(build_model16(),8).summary()
# output from verious layers in the model
def model_tap(model,last_conv_layer_name='Conv5'):
    convout = model.get_layer(name=last_conv_layer_name).output
    x = BatchNormalization()(convout)
    x = LeakyReLU(alpha=0.1)(x)
    out2 = GlobalMaxPooling1D()(x)
    out=model.get_layer(name='out').output
    out1=model.get_layer(name='out').input
    return Model(model.inputs,outputs=[out,out1,out2])
model_tap(build_model16()).summary()    
opt=optimizers.Adam()

opt.lr=1e-3
meta_cols=['hostgal_photoz','dmjd','std_flux','distmod']
if (meta_type=='Nyanp'):
    nyanp_train=pd.read_csv('{}nyanp_train.csv'.format(my_DATA_DIR)).set_index('object_id')
    nyanp_train=nyanp_train.rename(columns=int)
    nyanp_train.head()
    df_add=nyanp_train
    meta_cols.extend(range(15))
elif (meta_type=='Mamas'):
    mamas_train=pd.read_csv('{}mamas_train.csv'.format(my_DATA_DIR)).set_index('object_id')
    mamas_train=mamas_train.rename(columns=int)
    mamas_train.head()
    df_add=mamas_train
    meta_cols.extend(range(16))
meta_size=len(meta_cols)
col_weights=np.ones(meta_size)
    
if meta_type:
    for i in folds:
            df_validation_meta[i]=df_validation_meta[i].merge(df_add,left_on='object_id',right_index=True,how='left').sort_values('object_id').reset_index(drop=True).copy()
            df_training_meta[i]=df_training_meta[i].merge(df_add,left_on='object_id',right_index=True,how='left').sort_values('object_id').reset_index(drop=True).copy()
# Training
if test_train_switch=='Train':
    opt=optimizers.Adam()
    opt.lr=1e-3


    myhist={'val_loss':[],
            'val_categorical_accuracy':[],
            'loss':[],
            'categorical_accuracy':[],
            'lr':[],
            'score':[]}

    # Loop over all Folds
    fold=tqdm_notebook(folds,desc='fold:')
    for f in fold:
        smin1=100
        # Prepare validation data
        validate_timeseries,validate_timeseries0,validate_void,validate_meta,validate_switch,validate_y=\
            calculate_inputs(df_timeseries,df_validation_meta[f],meta_cols,col_weights,real_targets,length=num_samples,aug=None,return_y=True)
        tq=tqdm_notebook(range(training_trails),desc='full:')
        # Corse loop
        for p in tq:
            weights_arr=[]
            val_loss_arr=[]

            lr=lr_corse
            smin=100
            executor=concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            futurs=[]
            modelb = build_model16()
            model=model_retop(modelb,meta_size)
            if pre_weight_ext:
                model.load_weights('{}model16fr{}{}_{}.weights'.format(my_DATA_DIR,meta_size,pre_weight_ext,f))
            model.compile(opt,loss=mywloss,metrics=['categorical_accuracy'])
            aug_vals=aug1
            for k in range(lr.shape[0]):
                futurs.append(executor.submit(calculate_inputs,df_timeseries,df_training_meta[f],\
                                              meta_cols,col_weights,real_targets,length=num_samples,aug=aug_vals,return_y=True))
            tq1=tqdm_notebook(zip(lr,concurrent.futures.as_completed(futurs)),leave=False,total=lr.shape[0])
            # Epoch loop - use different learning rates
            for opt.lr,future in tq1:
                sr,sr0,srv,train_meta,train_switch,train_y=future.result(timeout=200)
                _=gc.collect()
                history=model.fit([sr,sr0,srv,train_meta,train_switch],train_y,batch_size=64,epochs=1,
                                validation_data=([validate_timeseries,validate_timeseries0,validate_void,
                                                  validate_meta,validate_switch],validate_y),
                                                    verbose=0 )
                y=model.predict([validate_timeseries,validate_timeseries0,validate_void,
                                                  validate_meta,validate_switch],batch_size=64)

                s=score(y,validate_y)
                myhist['val_loss'].extend(history.history['val_loss'])
                myhist['val_categorical_accuracy'].extend(history.history['val_categorical_accuracy'])
                myhist['loss'].extend(history.history['loss'])
                myhist['categorical_accuracy'].extend(history.history['categorical_accuracy'])
                myhist['lr'].append(opt.lr)
                myhist['score'].append(s)
                val_loss_arr.append(s)
                weights_arr.append(model.get_weights())
                if s<smin:
                    smin=s
                    weights_min=model.get_weights()
                tq1.set_postfix(lr=opt.lr,current_score=s,loss=history.history['loss'][0],val_loss=history.history['val_loss'][0])
                tq.set_postfix(temp_min=smin,total_min=min(smin,smin1))
            executor.shutdown(wait=True)

            lr=lr_fine
            aug_vals=aug2
            m=np.argsort(val_loss_arr)
            # Use the average of 5 best weights for next step
            new_weights=[]
            for ii in range(len(weights_arr[0])):
                new_weights.append(np.stack([weights_arr[m[kk]][ii] for kk in range(5) ],axis=0).mean(axis=0))

            executor=concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            futurs=[]

            modelb = build_model16()
            model=model_retop(modelb,meta_size)        
            model.set_weights(new_weights)
            model.compile(opt,loss=mywloss,metrics=['categorical_accuracy'])
            weights_arr=[]
            val_loss_arr=[]
            # fine training loop
            for k in range(lr.shape[0]):
                futurs.append(executor.submit(calculate_inputs,df_timeseries,df_training_meta[f],\
                                              meta_cols,col_weights,real_targets,length=num_samples,aug=aug_vals,return_y=True))
            tq2=tqdm_notebook(zip(lr,concurrent.futures.as_completed(futurs)),leave=False,total=lr.shape[0])
            for opt.lr,future in tq2:

                sr,sr0,srv,train_meta,train_switch,train_y=future.result(timeout=200)
                _=gc.collect()
                history=model.fit([sr,sr0,srv,train_meta,train_switch],train_y,batch_size=256,epochs=1,
                                validation_data=([validate_timeseries,validate_timeseries0,validate_void,
                                                  validate_meta,validate_switch],validate_y),
                                                    verbose=0 )
                y=model.predict([validate_timeseries,validate_timeseries0,validate_void,
                                                  validate_meta,validate_switch],batch_size=64)
                s=score(y,validate_y)

                myhist['val_loss'].extend(history.history['val_loss'])
                myhist['val_categorical_accuracy'].extend(history.history['val_categorical_accuracy'])
                myhist['loss'].extend(history.history['loss'])
                myhist['categorical_accuracy'].extend(history.history['categorical_accuracy'])
                myhist['lr'].append(opt.lr)
                myhist['score'].append(s)
                val_loss_arr.append(s)
                weights_arr.append(model.get_weights())
                if s<smin:
                    smin=s
                    weights_min=model.get_weights()
                tq2.set_postfix(lr=opt.lr,current_score=s,loss=history.history['loss'][0],val_loss=history.history['val_loss'][0])
                tq.set_postfix(temp_min=smin,total_min=min(smin,smin1))
            executor.shutdown(wait=True)
            if smin<smin1:
                smin1=smin
                weights_min1=weights_min.copy()

            m=np.argsort(val_loss_arr)
            new_weights=[]
            for ii in range(len(weights_arr[0])):
                new_weights.append(np.stack([weights_arr[m[kk]][ii] for kk in range(5) ],axis=0).mean(axis=0))
            model.set_weights(new_weights)
            y=model.predict([validate_timeseries,validate_timeseries0,validate_void,
                                                  validate_meta,validate_switch],batch_size=64)
            s=score(y,validate_y)
            if s<smin1:
                smin1=s
                weights_min1=new_weights.copy()
        model.set_weights(weights_min1)
        model.save_weights('model16fr{}{}_{}.weights'.format(meta_size,post_weight_ext,f))
        y=model.predict([validate_timeseries,validate_timeseries0,validate_void,
                                                  validate_meta,validate_switch],batch_size=64)

        score(y,validate_y)

    plt.plot(myhist['loss'])
    plt.plot(myhist['val_loss'])
    plt.plot(myhist['score'])
#Calculate CV
ys=[]
yv=[]

modelb = build_model16()
model=model_retop(modelb,meta_size)

for f in folds:
    validate_timeseries,validate_timeseries0,validate_void,validate_meta,validate_switch,validate_y=\
        calculate_inputs(df_timeseries,df_validation_meta[f],meta_cols,col_weights,real_targets,length=num_samples,aug=None,return_y=True)

    model.load_weights('{}model16fr{}{}_{}.weights'.format(my_DATA_DIR,meta_size,weight_ext,f))
    
    y=model.predict([validate_timeseries,validate_timeseries0,validate_void,
                                              validate_meta,validate_switch],batch_size=64)

    score(y,validate_y)
    ys.append(y.copy())
    yv.append(validate_y.copy())
score(np.concatenate(ys),np.concatenate(yv))
# if Traning, let's see the CV before training
if (test_train_switch=='Train') and (pre_weight_ext):
    ys=[]
    yv=[]

    modelb = build_model16()
    model=model_retop(modelb,meta_size)

    for f in folds:
        validate_timeseries,validate_timeseries0,validate_void,validate_meta,validate_switch,validate_y=\
            calculate_inputs(df_timeseries,df_validation_meta[f],meta_cols,col_weights,real_targets,length=num_samples,aug=None,return_y=True)

        model.load_weights('model16fr{}{}_{}.weights'.format(meta_size,pre_weight_ext,f))

        y=model.predict([validate_timeseries,validate_timeseries0,validate_void,
                                                  validate_meta,validate_switch],batch_size=64)

        score(y,validate_y)
        ys.append(y.copy())
        yv.append(validate_y.copy())
    score(np.concatenate(ys),np.concatenate(yv))
# Run on test
meta_cols=['hostgal_photoz','dmjd','std_flux','distmod']
if test_train_switch=='Test':
    df_test_meta=pd.read_csv(DATA_DIR+'test_set_metadata.csv')
    if (meta_type=='Nyanp'):
        nyanp_test=pd.read_csv('{}nyanp_test.csv'.format(my_DATA_DIR)).set_index('object_id')
        nyanp_test=nyanp_test.rename(columns=int)
        nyanp_test.head()
        added_cols=range(15)
        meta_cols.extend(added_cols)
    elif (meta_type=='Mamas'):
        mamas_test=pd.read_csv('{}mamas_test.csv'.format(my_DATA_DIR)).set_index('object_id')
        mamas_test=mamas_test.rename(columns=int)
        mamas_test.head()
        added_cols=range(16)
        meta_cols.extend(added_cols)

    meta_size=len(meta_cols)
    col_weights=np.ones(meta_size)

    models=[]
    res_dfs=[]
    res_dfps=[]
    queue_length=max_workers
    chunksize=500000
    for f in folds:
        modelb = build_model16()
        modelp=model_retop(modelb,meta_size)
        model=model_tap(modelp)
        modelp.load_weights('{}model16fr{}{}_{}.weights'.format(my_DATA_DIR,meta_size,weight_ext,f))
        models.append(model)
        res_dfs.append([])
        res_dfps.append([])
    timeseries_file=DATA_DIR+'test_set.csv'
    df_test_meta=pd.read_csv(DATA_DIR+'test_set_metadata.csv')
    df_test_extra = pd.read_csv(my_DATA_DIR+'test_extra.csv')
    df_test_meta=df_test_meta.merge(df_test_extra,on='object_id',how='left')
    if (meta_type=='Mamas'):
        df_test_meta=df_test_meta.merge(mamas_test[[i for i in added_cols]],left_on='object_id',right_index=True,how='left').sort_values('object_id')
    elif (meta_type=='Nyanp'):
        df_test_meta=df_test_meta.merge(nyanp_test[[i for i in added_cols]],left_on='object_id',right_index=True,how='left').sort_values('object_id')
    in_queue =  Queue(queue_length)
    out_queue =  Queue(queue_length)
    print('start')
    Process(target=load_process, args=(df_test_meta,timeseries_file,in_queue,queue_length,chunksize)).start()
    for q in range(queue_length):
        Process(target=prepare_process,args=(meta_cols,col_weights,real_targets,in_queue,out_queue,num_samples)).start()
    print('running')
    tq=tqdm_notebook(desc='objects:',total=df_test_meta.shape[0])
    for st in range(queue_length):
        for object_id,timeseries,timeseries0,void,meta,sw in iter(out_queue.get, 'STOPED'):
            for f in folds:
                [y,yp,yf]=models[f].predict([timeseries,timeseries0,void,meta,sw],batch_size=64)
                res_dfs[f].append(pd.DataFrame(index=object_id, data=y, columns=['class_%d' % k for k in real_targets]))
                res_dfps[f].append(pd.DataFrame(index=object_id, data=yp, columns=['class_%d' % k for k in real_targets]))
            tq.update(meta.shape[0])
    df_res=[]
    df_resp=[]
    for f in range(4):
        df_res.append(pd.concat(res_dfs[f]).sort_index())
        df_resp.append(pd.concat(res_dfps[f]).sort_index())


    for f in range(4):
        df_res[f].to_csv('pred16r{}{}_{}.csv'.format(meta_size,weight_ext,f))
    
if submit_name:
    for f in folds:
        df_res[f]['class_99']=(1-df_res[f].max(axis=1))
    for f in folds:
        m=df_res[f]['class_99'].mean()
        df_res[f]['class_99']=df_res[f]['class_99']/m/8
        df_res[f]=df_res[f].div(df_res[f].sum(axis=1), axis=0)
    df_rest=df_res[0]
    for f in folds[1:]:
        df_rest=df_rest+df_res[f]
    df_rest.head()
    df_rest=df_rest.clip(1e-3,1)
    df_rest=df_rest.div(df_rest.sum(axis=1), axis=0)
    df_rest.sample(20)
    df_rest.to_csv('{}.csv'.format(External_Dir,submit_name),float_format='%.4f')
def calc_extra(df_timeseries):
    gp=df_timeseries.groupby('object_id')
    dfe=(gp['mjd'].max()-gp['mjd'].min()).rename('dmjd').reset_index()
    dfe['dmjd']=dfe['dmjd']/1000
    dfe['std_flux']=gp.flux.std().reset_index().flux/1000
    return dfe
    
if False:     # We don't want to run the code below
    foldn=4
    df_group=[]
    df_group_meta=[]
    df_validate_meta=[]
    df_train_meta=[]
    df_train_l=[]
    for i in range(foldn):
        df_group.append([])
        df_group_meta.append([])
        df_validate_meta.append(None)
        df_train_meta.append(None)
        df_train_l.append([])
    df_training=pd.read_csv(DATA_DIR+'training_set.csv')
    df_training_meta=pd.read_csv(DATA_DIR+'training_set_metadata.csv')
    extra = calc_extra(df_training)
    df_training_metas=df_training_meta.merge(extra,on='object_id',how='left').sample(frac=1).copy()

    for target in real_targets:
        df = df_training_metas[df_training_metas.target==target].copy()
        for i in range(foldn):
            df_group[i].append(df.iloc[int(i*df.shape[0]/foldn):int(((i+1)*df.shape[0])/foldn)])

    for i in range(foldn):
        df_group_meta[i]=pd.concat(df_group[i])


    for i in range(foldn):
        for j in range(foldn):
            if (i==j):
                df_validate_meta[i]=df_group_meta[j].copy()
            else:
                df_train_l[i].append(df_group_meta[j].copy())

        df_train_meta[i]=pd.concat(df_train_l[i])
        df_train_meta[i].shape
        df_validate_meta[i].shape
        df_train_meta[i].to_csv('{}training_meta{}.csv'.format(DATA_DIR,i),index=False)
        df_validate_meta[i].to_csv('{}validation_meta{}.csv'.format(DATA_DIR,i),index=False)


if False:     # We don't want to run the code below
    chunksize=5000000
    df_test_meta=pd.read_csv(DATA_DIR+'test_set_metadata.csv')
    extras=[]
    df_test_extras=[]
    for chunk in tqdm_notebook(pd.read_csv(DATA_DIR+'test_set.csv', chunksize=chunksize)):
        first_id=chunk.head(1)['object_id'].values[0]
        last_id=chunk.tail(1)['object_id'].values[0]
        select=chunk['object_id'].isin([first_id,last_id])
        extras.append(chunk[select].copy())
        mid_chunk=chunk[~select].sort_values(['object_id','mjd']).copy()
        if mid_chunk.shape[0]>0:
            df_test_extras.append(calc_extra(mid_chunk))
    mid_chunk=pd.concat(extras)
    df_test_extras.append(calc_extra(mid_chunk))
    df_test_extra=pd.concat(df_test_extras).sort_values('object_id')
    df_test_extra.sample(10)
    df_test_extra.shape
    df_test_extra.to_csv(DATA_DIR+'test_extra.csv',index=False)