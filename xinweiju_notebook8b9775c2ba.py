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





import os

import cv2

import pydicom

import pandas as pd

import numpy as np 

import tensorflow as tf 

import matplotlib.pyplot as plt 

import random

from tqdm.notebook import tqdm 

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_absolute_error

from tensorflow_addons.optimizers import RectifiedAdam

from tensorflow.keras import Model

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M

from tensorflow.keras.optimizers import Nadam

import seaborn as sns

from PIL import Image



def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(42)





# In[4]:





config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)







train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv') 





# In[6]:





def get_tab(df):

    vector = [(df.Age.values[0] - 30) / 30] 

    

    if df.Sex.values[0] == 'male':

       vector.append(0)

    else:

       vector.append(1)

    

    if df.SmokingStatus.values[0] == 'Never smoked':

        vector.extend([0,0])

    elif df.SmokingStatus.values[0] == 'Ex-smoker':

        vector.extend([1,1])

    elif df.SmokingStatus.values[0] == 'Currently smokes':

        vector.extend([0,1])

    else:

        vector.extend([1,0])

    return np.array(vector) 





# In[7]:





A = {} 

TAB = {} 

P = [] 

for i, p in tqdm(enumerate(train.Patient.unique())):

    sub = train.loc[train.Patient == p, :] 

    fvc = sub.FVC.values

    weeks = sub.Weeks.values

    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc)[0]

    

    A[p] = a

    TAB[p] = get_tab(sub)

    P.append(p)





# In[8]:





def get_img(path):

    d = pydicom.dcmread(path)

    return cv2.resize(d.pixel_array / 2**11, (512, 512))





# In[9]:





from tensorflow.keras.utils import Sequence



class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(self, keys, a, tab, batch_size=32):

        self.keys = [k for k in keys if k not in self.BAD_ID]

        self.a = a

        self.tab = tab

        self.batch_size = batch_size

        

        self.train_data = {}

        for p in train.Patient.values:

            self.train_data[p] = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

    

    def __len__(self):

        return 1000

    

    def __getitem__(self, idx):

        x = []

        a, tab = [], [] 

        keys = np.random.choice(self.keys, size = self.batch_size)

        for k in keys:

            try:

                i = np.random.choice(self.train_data[k], size=1)[0]

                img = get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}')

                x.append(img)

                a.append(self.a[k])

                tab.append(self.tab[k])

            except:

                print(k, i)

       

        x,a,tab = np.array(x), np.array(a), np.array(tab)

        x = np.expand_dims(x, axis=-1)

        return [x, tab] , a





# In[10]:





from tensorflow.keras.layers import (

    Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 

    LeakyReLU, Concatenate 

)

import efficientnet.tfkeras as efn



def get_efficientnet(model, shape):

    models_dict = {

        'b0': efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),

        'b1': efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),

        'b2': efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),

        'b3': efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),

        'b4': efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),

        'b5': efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),

        'b6': efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),

        'b7': efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)

    }

    return models_dict[model]



def build_model(shape=(512, 512, 1), model_class=None):

    inp = Input(shape=shape)

    base = get_efficientnet(model_class, shape)

    x = base(inp)

    x = GlobalAveragePooling2D()(x)

    inp2 = Input(shape=(4,))

    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)

    x = Concatenate()([x, x2]) 

    x = Dropout(Dropout_model)(x)

    x = Dense(1)(x)

    model = Model([inp, inp2] , x)

    

    weights = [w for w in os.listdir('../input/osic-model-weights') if model_class in w][0]

    model.load_weights('../input/osic-model-weights/' + weights)

    return model



model_classes = ['b7'] #['b0','b1','b2','b3',b4','b5','b6','b7']

models = [build_model(shape=(512, 512, 1), model_class=m) for m in model_classes]

print('Number of models: ' + str(len(models)))





# In[11]:





tr_p, vl_p = train_test_split(P, shuffle=True, train_size = 0.8) 





# In[12]:





def score(fvc_true, fvc_pred, sigma):

    sigma_clip = np.maximum(sigma, 70) # changed from 70, trie 66.7 too

    delta = np.abs(fvc_true - fvc_pred)

    delta = np.minimum(delta, 1000)

    sq2 = np.sqrt(2)

    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip* sq2)

    return np.mean(metric)





# In[13]:





subs = []

for model in models:

    metric = []

    for q in tqdm(range(1, 10)):

        m = []

        for p in vl_p:

            x = [] 

            tab = [] 



            if p in ['ID00011637202177653955184', 'ID00052637202186188008618']:

                continue



            ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/')

            for i in ldir:

                if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                    x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/train/{p}/{i}')) 

                    tab.append(get_tab(train.loc[train.Patient == p, :])) 

            if len(x) < 1:

                continue

            tab = np.array(tab) 



            x = np.expand_dims(x, axis=-1) 

            _a = model.predict([x, tab]) 

            a = np.quantile(_a, q / 10)



            percent_true = train.Percent.values[train.Patient == p]

            fvc_true = train.FVC.values[train.Patient == p]

            weeks_true = train.Weeks.values[train.Patient == p]



            fvc = a * (weeks_true - weeks_true[0]) + fvc_true[0]

            percent = percent_true[0] - a * abs(weeks_true - weeks_true[0])

            m.append(score(fvc_true, fvc, percent))

        print(np.mean(m))

        metric.append(np.mean(m))



    q = (np.argmin(metric) + 1)/ 10



    sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv') 

    test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv') 

    A_test, B_test, P_test,W, FVC= {}, {}, {},{},{} 

    STD, WEEK = {}, {} 

    for p in test.Patient.unique():

        x = [] 

        tab = [] 

        ldir = os.listdir(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/')

        for i in ldir:

            if int(i[:-4]) / len(ldir) < 0.8 and int(i[:-4]) / len(ldir) > 0.15:

                x.append(get_img(f'../input/osic-pulmonary-fibrosis-progression/test/{p}/{i}')) 

                tab.append(get_tab(test.loc[test.Patient == p, :])) 

        if len(x) <= 1:

            continue

        tab = np.array(tab) 



        x = np.expand_dims(x, axis=-1) 

        _a = model.predict([x, tab]) 

        a = np.quantile(_a, q)

        A_test[p] = a

        B_test[p] = test.FVC.values[test.Patient == p] - a*test.Weeks.values[test.Patient == p]

        P_test[p] = test.Percent.values[test.Patient == p] 

        WEEK[p] = test.Weeks.values[test.Patient == p]



    for k in sub.Patient_Week.values:

        p, w = k.split('_')

        w = int(w) 



        fvc = A_test[p] * w + B_test[p]

        sub.loc[sub.Patient_Week == k, 'FVC'] = fvc

        sub.loc[sub.Patient_Week == k, 'Confidence'] = (

            P_test[p] - A_test[p] * abs(WEEK[p] - w) 

    ) 



    _sub = sub[["Patient_Week","FVC","Confidence"]].copy()

    subs.append(_sub)





# In[14]:





N = len(subs)

sub = subs[0].copy() # ref

sub["FVC"] = 0

sub["Confidence"] = 0

for i in range(N):

    sub["FVC"] += subs[0]["FVC"] * (1/N)

    sub["Confidence"] += subs[0]["Confidence"] * (1/N)





# In[15]:





sub.head()





# In[16]:





sub[["Patient_Week","FVC","Confidence"]].to_csv("submission_img.csv", index=False)





# In[17]:





img_sub = sub[["Patient_Week","FVC","Confidence"]].copy()









ROOT = "../input/osic-pulmonary-fibrosis-progression"

BATCH_SIZE=128



tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

chunk = pd.read_csv(f"{ROOT}/test.csv")



print("add infos")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")





# In[19]:





tr['WHERE'] = 'train'

chunk['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = tr.append([chunk, sub])





# In[20]:





print(tr.shape, chunk.shape, sub.shape, data.shape)

print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(), 

      data.Patient.nunique())





# In[21]:





data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')





# In[22]:





base = data.loc[data.Weeks == data.min_week]

base = base[['Patient','FVC']].copy()

base.columns = ['Patient','min_FVC']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)





# In[23]:





data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base





# In[24]:





COLS = ['Sex','SmokingStatus'] #,'Age'

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)





# In[25]:





#

data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','percent','week','BASE']





# In[26]:





tr = data.loc[data.WHERE=='train']

chunk = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']

del data





# In[27]:





tr.shape, chunk.shape, sub.shape









C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")



def score(y_true, y_pred):

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return K.mean(metric)



def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)



def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss



def make_model(nh):

    z = L.Input((nh,), name="Patient")

    x = L.Dense(100, activation="relu", name="d1")(z)

    x = L.Dense(100, activation="relu", name="d2")(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)

    p2 = L.Dense(3, activation="relu", name="p2")(x)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                     name="preds")([p1, p2])

    

    model = M.Model(z, preds, name="CNN")

    model.compile(loss=mloss(0.64), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])

    return model





# In[29]:





y = tr['FVC'].values

z = tr[FE].values

ze = sub[FE].values

nh = z.shape[1]

pe = np.zeros((ze.shape[0], 3))

pred = np.zeros((z.shape[0], 3))





# In[30]:





net = make_model(nh)

print(net.summary())

print(net.count_params())





# In[31]:





NFOLD = 5 # originally 5

kf = KFold(n_splits=NFOLD)





# In[32]:





get_ipython().run_cell_magic('time', '', 'cnt = 0\nEPOCHS = 800\nfor tr_idx, val_idx in kf.split(z):\n    cnt += 1\n    print(f"FOLD {cnt}")\n    net = make_model(nh)\n    net.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, \n            validation_data=(z[val_idx], y[val_idx]), verbose=0) #\n    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))\n    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))\n    print("predict val...")\n    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)\n    print("predict test...")\n    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD')





# In[33]:





sigma_opt = mean_absolute_error(y, pred[:, 1])

unc = pred[:,2] - pred[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)





# In[34]:





idxs = np.random.randint(0, y.shape[0], 100)

plt.plot(y[idxs], label="ground truth")

plt.plot(pred[idxs, 0], label="q25")

plt.plot(pred[idxs, 1], label="q50")

plt.plot(pred[idxs, 2], label="q75")

plt.legend(loc="best")

plt.show()





# In[35]:





print(unc.min(), unc.mean(), unc.max(), (unc>=0).mean())





# In[36]:





plt.hist(unc)

plt.title("uncertainty in prediction")

plt.show()





# In[37]:





sub.head()





# In[38]:





# PREDICTION

sub['FVC1'] = 1.*pe[:, 1]

sub['Confidence1'] = pe[:, 2] - pe[:, 0]

subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

subm.loc[~subm.FVC1.isnull()].head(10)





# In[39]:





subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

if sigma_mean<70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']





# In[40]:





subm.head()





# In[41]:





subm.describe().T





# In[42]:





otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1





# In[43]:





subm[["Patient_Week","FVC","Confidence"]].to_csv("submission_regression.csv", index=False)





# In[44]:





reg_sub = subm[["Patient_Week","FVC","Confidence"]].copy()







# In[45]:





df1 = img_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)

df2 = reg_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)





# In[46]:





df = df1[['Patient_Week']].copy()

df['FVC'] = FVC_weight*df1['FVC'] + (1-FVC_weight)*df2['FVC']                     ####

df['Confidence'] = Confidence_weight*df1['Confidence'] + (1-Confidence_weight)*df2['Confidence']

df.head()





# In[47]:





df.to_csv('submission.csv', index=False)