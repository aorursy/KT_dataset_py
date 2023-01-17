import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

from seaborn import countplot,lineplot, barplot

from scipy import stats

import math
import os

print(os.listdir("../input"))
from numpy import mean

from numpy import std

from numpy import dstack

from pandas import read_csv

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Dropout

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers.merge import concatenate

from keras.models import Sequential

from keras.models import Model

from keras.layers import AveragePooling1D

from keras.layers import GlobalAveragePooling1D

from keras.models import load_model



import warnings

warnings.filterwarnings('ignore')

import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupKFold





from sklearn.metrics import roc_auc_score

from sklearn.datasets import make_classification

from keras.utils import np_utils

from keras.callbacks import Callback, EarlyStopping

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/career-con-2019/X_train.csv')

target = pd.read_csv('../input/career-con-2019/y_train.csv')

test = pd.read_csv('../input/career-con-2019/X_test.csv')

sub = pd.read_csv('../input/career-con-2019/sample_submission.csv')
target.head()
train['series_id'].nunique(), test['series_id'].nunique() #3810 series's of train, 3816 series's of test
target['group_id'].nunique() #73 group_id 
target['surface'].value_counts().reset_index().rename(columns={'index': 'target'})
sns.set(style="darkgrid")

countplot(y= 'surface', data= target, order= target['surface'].value_counts().index)

plt.show()
plt.figure(figsize=(23,5)) 

countplot(x="group_id", data=target, order = target['group_id'].value_counts().index)

plt.show()
# # https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1

# def quaternion_to_euler(x, y, z, w):

#     import math

#     t0 = +2.0 * (w * x + y * z)

#     t1 = +1.0 - 2.0 * (x * x + y * y)

#     X = math.atan2(t0, t1)



#     t2 = +2.0 * (w * y - z * x)

#     t2 = +1.0 if t2 > +1.0 else t2

#     t2 = -1.0 if t2 < -1.0 else t2

#     Y = math.asin(t2)



#     t3 = +2.0 * (w * z + x * y)

#     t4 = +1.0 - 2.0 * (y * y + z * z)

#     Z = math.atan2(t3, t4)



#     return X, Y, Z
# def fe_step0 (data):

    

#     actual = data.copy()

#     # https://www.mathworks.com/help/aeroblks/quaternionnorm.html

#     # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html

#     # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html

        

#     actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)

#     actual['mod_quat'] = (actual['norm_quat'])**0.5

#     actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']

#     actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']

#     actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']

#     actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    

#     return actual
# def fe_step1 (actual):

#     """Quaternions to Euler Angles"""

    

#     x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()

#     nx, ny, nz = [], [], []

#     for i in range(len(x)):

#         xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

#         nx.append(xx)

#         ny.append(yy)

#         nz.append(zz)

    

#     actual['euler_x'] = nx

#     actual['euler_y'] = ny

#     actual['euler_z'] = nz

#     return actual
# train_df = fe_step0(train)

# test_df = fe_step0(test)

# train_df = fe_step1(train_df)

# test_df = fe_step1(test_df)
# from numpy.fft import rfft, rfftfreq, irfft



# # from @theoviel at https://www.kaggle.com/theoviel/fast-fourier-transform-denoising

# def filter_signal(signal, threshold=1e3):

#     fourier = rfft(signal)

#     frequencies = rfftfreq(signal.size, d=20e-3/signal.size)

#     fourier[frequencies > threshold] = 0

#     return irfft(fourier)
# def denoised_df(X_train):

#     # denoise train and test angular_velocity and linear_acceleration data

#     X_train_denoised = X_train.copy()



#     # train

#     for col in X_train_denoised.columns:

#         if col[0:3] == 'ang' or col[0:3] == 'lin':

#             # Apply filter_signal function to the data in each series

#             denoised_data = X_train_denoised.groupby(['series_id'])[col].apply(lambda x: filter_signal(x))



#             # Assign the denoised data back to X_train

#             list_denoised_data = []

#             for arr in denoised_data:

#                 for val in arr:

#                     list_denoised_data.append(val)



#             X_train_denoised[col] = list_denoised_data

#     return X_train_denoised
train_df = train.copy()

test_df = test.copy()
# plt.figure(figsize=(24, 8))

# plt.title('linear_acceleration_X')

# plt.plot(train.angular_velocity_Z[128:256], label="original");

# plt.plot(train_df.angular_velocity_Z[128:256], label="denoised");

# plt.legend()

# plt.show()
def feat_diff(data):

    for col in data.columns[3:]:

        data[col+'_diff'] = data.groupby(['series_id'])[col].diff().fillna(0)

    return data



def feat_make(data):

    data = feat_diff(data)

    

    X0  = data.loc[:, 'orientation_X':'orientation_W'].values.reshape(-1, 128, 4)

    X1  = data.loc[:, 'angular_velocity_X': 'angular_velocity_Z'].values.reshape(-1, 128, 3)

    X2  = data.loc[:, 'linear_acceleration_X': 'linear_acceleration_Z'].values.reshape(-1, 128, 3)     

#     X3  = data.loc[:, 'norm_quat':'mod_quat'].values.reshape(-1, 128, 2)

#     X4  = data.loc[:, 'norm_X':'norm_W'].values.reshape(-1, 128, 4)

#     X5  = data.loc[:, 'euler_x': 'euler_z'].values.reshape(-1, 128, 3) 

#     X6  = data.loc[:, 'totl_anglr_vel':'acc_vs_vel'].values.reshape(-1, 128, 4)

    

    X7  = data.loc[:, 'orientation_X_diff':'orientation_W_diff'].values.reshape(-1, 128, 4)

    X8  = data.loc[:, 'angular_velocity_X_diff': 'angular_velocity_Z_diff'].values.reshape(-1, 128, 3)

    X9  = data.loc[:, 'linear_acceleration_X_diff': 'linear_acceleration_Z_diff'].values.reshape(-1, 128, 3)

#     X10 = data.loc[:, 'norm_quat_diff':'mod_quat_diff'].values.reshape(-1, 128, 2)

#     X11 = data.loc[:, 'norm_X_diff':'norm_W_diff'].values.reshape(-1, 128, 4)

#     X12 = data.loc[:, 'euler_x_diff': 'euler_z_diff'].values.reshape(-1, 128, 3)

#     X13 = data.loc[:, 'totl_anglr_vel_diff':'acc_vs_vel_diff'].values.reshape(-1, 128, 4)

    

    return  X0, X1, X2, X7, X8, X9
def mfft(x):

    return [ x/math.sqrt(128.0) for x in np.absolute(np.fft.fft(x)) ][1:65]



def feat_fft(X_trn):

    ars=[]

    for ar in X_trn:

        ar= ar.T.tolist()

        ms= [] 

        for line in ar:

            m= mfft(line)

            ms.append(m)

        ms= np.array(ms).T.tolist()

        ars.append(ms)

    ars= np.array(ars)

    return ars
# import math

# def prepare_data(t):

#     def f(d):

#         d=d.sort_values(by=['measurement_number'])

#         return pd.DataFrame({

#          'lx':[ d['linear_acceleration_X'].values ],

#          'ly':[ d['linear_acceleration_Y'].values ],

#          'lz':[ d['linear_acceleration_Z'].values ],

#          'ax':[ d['angular_velocity_X'].values ],

#          'ay':[ d['angular_velocity_Y'].values ],

#          'az':[ d['angular_velocity_Z'].values ],

#         })



#     t= t.groupby('series_id').apply(f)



#     def mfft(x):

#         return [ x/math.sqrt(128.0) for x in np.absolute(np.fft.fft(x)) ][1:65]



#     t['lx_f']=[ mfft(x) for x in t['lx'].values ]

#     t['ly_f']=[ mfft(x) for x in t['ly'].values ]

#     t['lz_f']=[ mfft(x) for x in t['lz'].values ]

#     t['ax_f']=[ mfft(x) for x in t['ax'].values ]

#     t['ay_f']=[ mfft(x) for x in t['ay'].values ]

#     t['az_f']=[ mfft(x) for x in t['az'].values ]

#     return t



# t=prepare_data(train_df)



# t=pd.merge(t,target[['series_id','surface','group_id']],on='series_id')

# t=t.rename(columns={"surface": "y"})





# def aggf(d, feature):

#     va= np.array(d[feature].tolist())

#     mean= sum(va)/va.shape[0]

#     var= sum([ (va[i,:]-mean)**2 for i in range(va.shape[0]) ])/va.shape[0]

#     dev= [ math.sqrt(x) for x in var ]

#     return pd.DataFrame({

#         'mean': [ mean ],

#         'dev' : [ dev ],

#     })



# display={

# 'hard_tiles_large_space':'r-.',

# 'concrete':'g-.',

# 'tiled':'b-.',



# 'fine_concrete':'r-',

# 'wood':'g-',

# 'carpet':'b-',

# 'soft_pvc':'y-',



# 'hard_tiles':'r--',

# 'soft_tiles':'g--',

# }



# import matplotlib.pyplot as plt

# plt.figure(figsize=(14, 8*7))

# #plt.margins(x=0.0, y=0.0)

# #plt.tight_layout()

# # plt.figure()



# features=['lx_f','ly_f','lz_f','ax_f','ay_f','az_f']

# count=0



# for feature in features:

#     stat= t.groupby('y').apply(aggf,feature)

#     stat.index= stat.index.droplevel(-1)

#     b=[*range(len(stat.at['carpet','mean']))]



#     count+=1

#     plt.subplot(len(features)+1,1,count)

#     for i,(k,v) in enumerate(display.items()):

#         plt.plot(b, stat.at[k,'mean'], v, label=k)

#         # plt.errorbar(b, stat.at[k,'mean'], yerr=stat.at[k,'dev'], fmt=v)

   

#     leg = plt.legend(loc='best', ncol=3, mode="expand", shadow=True, fancybox=True)

#     plt.title("sensor: " + feature)

#     plt.xlabel("frequency component")

#     plt.ylabel("amplitude")



# count+=1

# plt.subplot(len(features)+1,1,count)

# k='concrete'

# v=display[k]

# feature='lz_f'

# stat= t.groupby('y').apply(aggf,feature)

# stat.index= stat.index.droplevel(-1)

# b=[*range(len(stat.at['carpet','mean']))]



# plt.errorbar(b, stat.at[k,'mean'], yerr=stat.at[k,'dev'], fmt=v)

# plt.title("sample for error bars (lz_f, surface concrete)")

# plt.xlabel("frequency component")

# plt.ylabel("amplitude")



# plt.show()
# def feat_reshape(t):  

#     m0= []

#     for rows in t:

#         m= []

#         for cols in rows: 

#             m.append(cols)

#         m= np.array(m).T.tolist()



#         m0.append(m)



#     m0 = np.array(m0)

#     return m0
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



le.fit(target['surface'])

y_train= le.transform(target['surface'])

le.classes_
le.transform(['carpet', 'concrete', 'fine_concrete', 'hard_tiles',

       'hard_tiles_large_space', 'soft_pvc', 'soft_tiles', 'tiled',

       'wood'])
le.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8])
X_trn_q, X_trn_a, X_trn_l, X_trn_q_d, X_trn_a_d, X_trn_l_d = feat_make(train_df)



X_tst_q, X_tst_a, X_tst_l, X_tst_q_d, X_tst_a_d, X_tst_l_d = feat_make(test_df)
X_trn_q_f = feat_fft(X_trn_q)

X_trn_a_f = feat_fft(X_trn_a)

X_trn_l_f = feat_fft(X_trn_l)

X_trn_q_d_f = feat_fft(X_trn_q_d)

X_trn_a_d_f = feat_fft(X_trn_a_d)

X_trn_l_d_f = feat_fft(X_trn_l_d)



X_tst_q_f = feat_fft(X_tst_q)

X_tst_a_f = feat_fft(X_tst_a)

X_tst_l_f = feat_fft(X_tst_l)

X_tst_q_d_f = feat_fft(X_tst_q_d)

X_tst_a_d_f = feat_fft(X_tst_a_d)

X_tst_l_d_f = feat_fft(X_tst_l_d)



def run_set(X_trn):

    for i in run_id:

        X_run = target[:][target['run_id']==i]

        idx= X_run.index

        X= X_trn[idx].tolist()

        X=[item for sublist in X for item in sublist]

        end=int(len(X)/2048)*2048     

        X=X[:end]

        X=np.array(X).reshape(-1, 2048, X_trn.shape[-1])

    return X



def test_run_set(X_trn):

    X_=[]

    for i in test_run_id:

        X_run = test_target[:][test_target['run_id']==i]

        idx= X_run.index

        X= X_trn[idx].tolist()

        X=[item for sublist in X for item in sublist]

        end=int(len(X)/2048)*2048     

        X=X[:end]

        X=np.array(X).reshape(-1, 2048, X_trn.shape[-1])

    return X
# from sklearn.neighbors import KernelDensity

# from sklearn.decomposition import PCA

# from sklearn.model_selection import GridSearchCV



# def X_gen_kde(X_trn, surface_type, numbers):

#     st= target[:][(target['surface'] == surface_type)]

#     X=[]

#     i=0

#     for idx in st.index:

#         X_t= X_trn[idx].tolist()

#         X.append(X_t)

#         i+=1

# #     print(i)

#     X= np.array(X).reshape(-1,(X_trn.shape[1]*X_trn.shape[2]))

#     #reshape 



#     # project the multi-dimensional data to a lower dimension

# #     pca = PCA(n_components=21, whiten=False)

# #     data = pca.fit_transform(X)



#     # use grid search cross-validation to optimize the bandwidth

#     params = {'bandwidth': np.logspace(-1, 1, 20)}

#     grid = GridSearchCV(KernelDensity(), params, cv=5)

# #     grid.fit(data)

#     grid.fit(X)



#     print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))



#     # use the best estimator to compute the kernel density estimate

#     kde = grid.best_estimator_



#     # sample numbers new points from the data

#     new_data = kde.sample(numbers, random_state=0)

# #     new_data = pca.inverse_transform(new_data)



#     # turn data back to multi-dimensional

#     new_data = new_data.reshape((-1, X_trn.shape[1], X_trn.shape[2]))

#     return new_data
# def X_gen(X_trn, surface_type, numbers):

#     st= target[:][(target['surface'] == surface_type)]



#     X_sum= np.zeros((X_trn.shape[1], X_trn.shape[2]))

#     i=0

#     for idx in st.index:

#         X_sum += X_trn[idx]

#         i+=1

# #     print(i)

#     X_mean = X_sum/ i

#     #mean

    

#     sq_sum= np.zeros_like(X_mean)

#     j=0

#     for idx in st.index:    

#         dev= np.abs(X_trn[idx] - X_mean)

#         sq_sum += dev**2

#         j+=1

# #     print(j)

#     X_std = np.sqrt(sq_sum/ j)

#     #std

    

#     X_trn_gen = np.random.normal(X_mean, X_std, (numbers, X_trn.shape[1], X_trn.shape[2]))

#     #gen

#     return X_trn_gen
# ingred= {'carpet':300, 'concrete':0, 'fine_concrete':200, 'hard_tiles':500,

#          'hard_tiles_large_space':200, 'soft_pvc':0, 'soft_tiles':200, 'tiled':0,

#          'wood':0}



# def X_aug(X_trn):    

#     i=0

#     for surface_type, numbers in ingred.items():

#         X_aug= X_gen(X_trn, surface_type, numbers)

# #         X_aug= X_gen_kde(X_trn, surface_type, numbers)

#         if i==0:

#             X_aug_total= X_aug

#         else:

#             X_aug_total= np.append(X_aug_total, X_aug, axis=0)

#         i+=1

        

#     X_aug_total= np.append(X_trn, X_aug_total, axis=0)

#     return X_aug_total



# def y_aug(y_trn):

#     i=0

#     for surface_type, numbers in ingred.items():

#         y_aug= np.full( (numbers,), le.transform([surface_type]) )

#         if i==0:

#             y_aug_total= y_aug

#         else:

#             y_aug_total= np.append(y_aug_total, y_aug, axis=0)

#         i+=1    

#     y_aug_total= np.append(y_trn, y_aug_total, axis=0)

#     return y_aug_total
X_trn_ = [X_trn_q, X_trn_a, X_trn_l, X_trn_q_d, X_trn_a_d, X_trn_l_d,

        X_trn_a_f, X_trn_l_f, X_trn_a_d_f, X_trn_l_d_f]



# X_train = [X_aug(item) for item in X_trn_]

# X_train = X_trn_
X_train = [run_set(x) for x in X_trn_]
# X_trn_2= [X_trn_a_f, X_trn_l_f]



# X_train_2=[X_aug(item) for item in X_trn_2]
# y_train = y_aug(y_train)
X_test= [X_tst_q, X_tst_a, X_tst_l, X_tst_q_d, X_tst_a_d, X_tst_l_d,

         X_tst_a_f, X_tst_l_f, X_tst_a_d_f, X_tst_l_d_f]



# X_test_2= [X_tst_a_f, X_tst_l_f]
X_test = [test_run_set(x) for x in X_test]
class Head:

    def build(n_steps, n_features):

        inputA = Input(shape=(n_steps, n_features))

        x = Conv1D(filters=100, kernel_size=10, activation='relu')(inputA)

        x = Conv1D(filters=100, kernel_size=10, activation='relu')(x)

        x = MaxPooling1D(pool_size=3)                             (x)

        x = Conv1D(filters=160, kernel_size=10, activation='relu')(x)

        x = Conv1D(filters=160, kernel_size=10, activation='relu')(x)

        x = AveragePooling1D(pool_size=3)                         (x)

        x = Dropout(0.5)                                          (x)

        x = Flatten()                                             (x)

        x = Model(inputs= inputA, outputs= x)

        return x

    

    def build_f(n_steps, n_features):

        inputA = Input(shape=(n_steps, n_features))

        x = Conv1D(filters=100, kernel_size=5, activation='relu')(inputA)

        x = Conv1D(filters=100, kernel_size=5, activation='relu')(x)

        x = MaxPooling1D(pool_size=3)                            (x)

        x = Conv1D(filters=160, kernel_size=5, activation='relu')(x)

        x = Conv1D(filters=160, kernel_size=5, activation='relu')(x)

        x = AveragePooling1D(pool_size=3)                        (x)

        x = Dropout(0.5)                                         (x)

        x = Flatten()                                            (x)

        x = Model(inputs= inputA, outputs= x)

        return x
def sub_model(X_trn, y_trn):     

# CNN heads

    #orignal

    flatA1 = Head.build(2048, 4)   

    flatA2 = Head.build(2048, 3)

    flatA3 = Head.build(2048, 3)

#     flatA4 = Head.build(128, 3)

    #diff

    flatB1 = Head.build(2048, 4)   

    flatB2 = Head.build(2048, 3)

    flatB3 = Head.build(2048, 3)

#     flatB4 = Head.build(128, 3)

    #orignal fft

#     fftA1 = Head.build_f(64, 4)

    fftA2 = Head.build_f(2048, 3)

    fftA3 = Head.build_f(2048, 3)

#     fftA4 = Head.build_f(64, 3)

    #diff fft

#     fftB1 = Head.build_f(64, 4)

    fftB2 = Head.build_f(2048, 3)

    fftB3 = Head.build_f(2048, 3)

#     fftB4 = Head.build_f(64, 3)

    

# merge CNN heads

    

    x= concatenate([

        flatA1.output, flatA2.output, flatA3.output, #flatA4.output,

        flatB1.output, flatB2.output, flatB3.output, #flatB4.output,

#         fftA1.output,

        fftA2.output, fftA3.output, 

#         fftA4.output,

#         fftB1.output, 

        fftB2.output, fftB3.output, 

#         fftB4.output

    ])

       

# interpretation

     

    x = Dense(500, activation='relu') (x)

    x = Dense(100, activation='relu') (x)

    x = Dense(n_surfaces, activation='softmax')(x)

    

    model = Model(inputs=[

                          flatA1.input, flatA2.input, flatA3.input, #flatA4.input,

                          flatB1.input, flatB2.input, flatB3.input, #flatB4.input, 

#                           fftA1.input, 

                          fftA2.input, fftA3.input, #fftA4.input,

#                           fftB1.input, 

                          fftB2.input, fftB3.input, #fftB4.input

                         ],

                  outputs= x)

#--------

# save a plot of the model

    plot_model(model, show_shapes=True, to_file='multichannel.png')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    model.fit(x= X_trn, y= y_trn, epochs= epochs, batch_size= batch_size, verbose= verbose)

    

    return model
# load models from file

def load_all_models(n_models):

    all_models = []

    for i in range(n_models):

        # define filename for this ensemble

        filename = 'model_' + str(i) + '.h5'

        # load model from file

        model = load_model(filename)

        # add to list of members

        all_models.append(model)

        print('>loaded %s' % filename)

    return all_models
# define stacked model from multiple member input models

def define_stacked_model(members):

    # update all layers in all models to not be trainable

    for i in range(len(members)):

        model = members[i]

        for layer in model.layers:

            # make not trainable

            layer.trainable = False

            # rename to avoid 'unique layer name' issue

            layer.name = 'ensemble_' + str(i) + '_' + layer.name

    # define multi-headed input

    ensemble_visible = [model.input for model in members] #nested list

    ensemble_visible = [item for sublist in ensemble_visible for item in sublist] ## unnest the nested list

    # concatenate merge output from each model

    ensemble_outputs = [model.output for model in members]

    merge = concatenate(ensemble_outputs)

    hidden = Dense(45, activation='relu')(merge)

    output = Dense(9, activation='softmax')(hidden)

    model = Model(inputs=ensemble_visible, outputs=output)

    # plot graph of ensemble

    plot_model(model, show_shapes=True, to_file='model_graph.png')

    # compile

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



#     print(cm)



    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    plt.grid(False)

    return ax
verbose, epochs, batch_size = 2, 15, 32

n_surfaces = 9

n_models= 2





folds = StratifiedShuffleSplit(n_splits= 2, test_size= 0.5, random_state=0)

# folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=0)

# folds = GroupKFold(n_splits=5)



np.set_printoptions(precision=2)

class_names= le.classes_
# def run_cv(X_trn, y_trn):

    

X_trn= X_train ##

y_trn= y_train ##



for fold_, (trn, val) in enumerate(folds.split(np.zeros(len(y_trn)), y_trn)):

    print("fold {}".format(fold_))

    

    y_oh = to_categorical(y_trn)

    xtrn= [x[trn] for x in X_trn]

    

    for i in range(n_models): #devide to n models for the specific surface

        

#         gp_idx = np.where(y_trn==i)[0].tolist()

#         co_idx= [idx for idx in gp_idx if idx in trn]

#         xtrn= [x[co_idx] for x in X_trn]

#         model = sub_model(xtrn, y_oh[co_idx])



        model = sub_model(xtrn, y_oh[trn])

        filename = 'model_' + str(i) + '.h5'

        model.save(filename)

        print('>Saved %s' % filename)



    # load all models

    members = load_all_models(n_models)

    print('Loaded %d models' % len(members))



    # define ensemble model

    stacked_model = define_stacked_model(members)



    # fit stacked model on test dataset

    X_t = [x[trn] for x in X_trn] 

    X_t = [X_t for _ in range(n_models)] # duplicate X_v for n level_0-models

    X_t = [item for sublist in X_t for item in sublist] #un-nest the list

    

    X_v = [x[val] for x in X_trn]

    X_v = [X_v for _ in range(n_models)] # duplicate X_v for n level_0-models

    X_v = [item for sublist in X_v for item in sublist] #un-nest the list

    

    stacked_model.fit(X_t, y_oh[trn], validation_data= (X_v, y_oh[val]), epochs=epochs, verbose=verbose)

    

    # summarize history for accuracy

    plt.plot(stacked_model.history.history['acc'])

    plt.plot(stacked_model.history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(stacked_model.history.history['loss'])

    plt.plot(stacked_model.history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

       

    # make predictions and evaluate 

    y_pred_ = stacked_model.predict(X_v, verbose=verbose)

    y_pred = y_pred_.argmax(axis=-1)

    acc = accuracy_score(y_trn[val], y_pred)

    print('Stacked Test Accuracy: %.3f' % acc)





    # Plot non-normalized confusion matrix

    plot_confusion_matrix(y_trn[val], y_pred, classes= class_names, 

                          title='Confusion matrix, without normalization')



    # Plot normalized confusion matrix

    plot_confusion_matrix(y_trn[val], y_pred, classes= class_names, 

                                normalize=True, title='Normalized confusion matrix')

     
#predict

n_folds = 2

for f in range(n_folds):

    print('fold',f) 

    y_oh = to_categorical(y_train)

    for i in range(n_models): #devide to n models for the specific surface

        model = sub_model(X_train, y_oh)

        filename = 'model_' + str(i) + '.h5'

        model.save(filename)

        print('>Saved %s' % filename)



    # load all models

    members = load_all_models(n_models)

    print('Loaded %d models' % len(members))



    # define ensemble model

    stacked_model = define_stacked_model(members)

    

    # fit stacked model on test dataset

    X_t = X_train

    X_t = [X_t for _ in range(n_models)] # duplicate X_v for n level_0-models

    X_t = [item for sublist in X_t for item in sublist] #un-nest the list

    stacked_model.fit(X_t, y_oh, epochs=5, verbose=verbose)

  

    X_tst = X_test 

    X_tst = [X_tst for _ in range(n_models)] # duplicate X_tst for n level_0-models

    X_tst = [item for sublist in X_tst for item in sublist] #un-nest the list

    y_test_ = stacked_model.predict(X_tst, verbose=verbose)

    

    # list all data in history

    #print(model.history.history.keys())

    # summarize history for accuracy

    plt.plot(model.history.history['acc'])

#     plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

#     plt.legend(['test'], loc='upper left')

#     plt.show()

    # summarize history for loss

    plt.plot(model.history.history['loss'])

#     plt.title('model loss')

#     plt.ylabel('loss')

#     plt.xlabel('epoch')



#     plt.legend(['test'], loc='upper left')

#     plt.show()

        

    if f==0:

        y_test = np.zeros_like(y_test_)

    y_test += y_test_

    

y_test = y_test/ n_folds
#output

y_test = y_test.argmax(axis=-1) #one-hot prob to classes code
y_run=[]

for i in run_id:

    X_run = target[:][target['run_id']==i]

    y = le.transform(X_run['surface'])[0]

    y_run.append(y)
for run_id, i in enumerate (y_test):

    if test_target['run_id']==run_id:

        test_target['surface']= le.inverse_transform(i)


subm = pd.DataFrame()

subm['series_id'] = test_target['series_id']

subm['surface']= test_target['surface']

subm = subm.sort_values('series_id')



subm.to_csv("submission.csv", index= False) 
subm