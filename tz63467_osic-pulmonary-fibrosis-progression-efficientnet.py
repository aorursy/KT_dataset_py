import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import tensorflow as tf
import pydicom

plt.style.use("dark_background")

main_dir = "../input/osic-pulmonary-fibrosis-progression"
train_imgs = tf.io.gfile.glob(main_dir + "/train/*/*")
test_files = tf.io.gfile.glob(main_dir+"/test/*/*")
df_train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
df_train
df_train.isna().sum().any()
df_train.info()
#checking to see which columns change with time.
(df_train.groupby('Patient').nunique() != 1).sum() == 0 
temp = pydicom.dcmread(train_imgs[1])
type(temp)
temp
print('\n'.join(str(temp).split('\n')[:15]))
# temp.keys()
list(temp.values())[:5]
temp.dir()[:5]
#displaying the image
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(temp.pixel_array,cmap = 'gray')
df_train.nunique()
df_train['SmokingStatus'].unique()
ages = df_train.groupby('Patient').Age.head(1)
print('Maximum Patient Age: {} \n Minimum Patient Age: {}'.format(ages.max(),ages.min()))
ax = ages.plot(kind='hist', bins=50, edgecolor='red', color='y', figsize=(15, 5), xticks=range(49, 89))
ages.plot(kind='kde', ax=ax, xlim=(47, 90), color='w', secondary_y=True);
f,ax = plt.subplots(figsize=(15,5) ,ncols=2)

df_train.groupby('Patient')['SmokingStatus'].head(1).value_counts().plot(kind='pie', ax=ax[0], autopct= lambda x: str(int(x))+"%",
title = 'Smoking Status pie Chart', colors = ['orange', 'blue', 'green'])

df_train.groupby('Patient')['Sex'].head(1).value_counts().plot(kind='pie', ax=ax[1],autopct= lambda x: str(int(x))+"%",
title = 'Sex pie Chart', colors = ['red', 'green'])
df_train.groupby(['SmokingStatus', 'Sex'])['Patient'].nunique().unstack().plot(
    kind='bar', stacked=True, figsize=(10, 6), yticks=range(0, 130, 10),
    rot=0, title='Gender Across Smoking Status')
df_train.groupby(['SmokingStatus','Sex'])[['Weeks', 'FVC', 'Percent', 'Age']].agg({
    'Weeks': 'count',
    'FVC' : ['min', 'max', 'mean', 'std'],
    'Age': ['min', 'max', 'mean', 'std'],
    'Percent' : ['min', 'max', 'mean', 'std']}).rename({'Weeks': "cumulative Records"}, axis=1)
from scipy.signal import savgol_filter

def display_FVC_progress(data, title, smooth=True, drop=1, median=True):
    agg = ['count', 'min', 'max', 'median']
    if not median:
        agg.remove('median')
    
    temp = data.groupby('Weeks')[['FVC']].agg(agg)
    temp = temp[temp['FVC']['count'] > drop].drop(('FVC', 'count'), axis=1)
    
    if smooth:
        temp['FVC', 'max'] = savgol_filter(temp['FVC', 'max'], 9, 3)
        temp['FVC', 'min'] = savgol_filter(temp['FVC', 'min'], 9, 3)

    ax = temp.plot(
        figsize=(15, 5), 
        title=f'Variation & progress of FVC over the Weeks ({title})', 
        legend=True, xticks=range(-10, 150, 5)
    );

    ax.fill_between(temp.index, temp['FVC', 'max'], temp['FVC', 'min'], color='green');

    
display_FVC_progress(df_train, 'All Categories')
display_FVC_progress(df_train[df_train['Sex']== 'Male'], 'Only Males' , smooth = True, median=False)
display_FVC_progress(df_train[df_train['Sex']== 'Female'], 'Only Females' , smooth = True, median=False)
display_FVC_progress(df_train[df_train['SmokingStatus']== 'Currently smokes'], 'Current Smokers' , smooth = True, median=False)
display_FVC_progress(df_train[df_train['SmokingStatus']== 'Ex-smoker'], 'Ex-Smokers' , smooth = True, median=False)
display_FVC_progress(df_train[df_train['SmokingStatus']== 'Never smoked'], 'Never Smoked' , smooth = True, median=False)
import os
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
import cv2
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)

!pip install efficientnet
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def get_tab(df):
    vector = [(df['Age'].values[0]-30)/30]
            
    if df['Sex'].values[0] == 'Male':
        vector.append(0)
    else:
        vector.append(1)
    
    if df['SmokingStatus'].values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df['SmokingStatus'].values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df['SmokingStatus'].values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0])
    return np.array(vector)
A = {}
TAB = {}
P = []

for i, p in tqdm(enumerate(df_train.Patient.unique())):
    sub = df_train.loc[df_train.Patient == p, :]
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    a,b = np.linalg.lstsq(c, fvc)[0]
    
    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)
def get_img(path):
    d = pydicom.dcmread(path)
    return cv2.resize(d.pixel_array/ 2**11, (512,512))
from tensorflow.keras.utils import Sequence


class IGenerator(Sequence):
    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']
    def __init__(self, keys, a, tab, batch_size=32):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.tab = tab
        self.batch_size = batch_size
        self.train_data = {}
        for p in df_train.Patient.values:
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
                print(k,i)
                
        x,a,tab = np.array(x), np.array(a), np.array(tab)
        x = np.expand_dims(x, axis = -1)
        return [x, tab], a
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D,
                                    Add, Conv2D, AveragePooling2D, LeakyReLU, Concatenate)
import efficientnet.tfkeras as efn


def get_efficientnet(model, shape):
    models_dict = {
        'b0' : efn.EfficientNetB0(input_shape=shape,weights=None,include_top=False),
        'b1' :efn.EfficientNetB1(input_shape=shape,weights=None,include_top=False),
        'b2' :efn.EfficientNetB2(input_shape=shape,weights=None,include_top=False),
        'b3' :efn.EfficientNetB3(input_shape=shape,weights=None,include_top=False),
        'b4' :efn.EfficientNetB4(input_shape=shape,weights=None,include_top=False),
        'b5' :efn.EfficientNetB5(input_shape=shape,weights=None,include_top=False),
        'b6' :efn.EfficientNetB6(input_shape=shape,weights=None,include_top=False),
        'b7' :efn.EfficientNetB7(input_shape=shape,weights=None,include_top=False)        
    }
    
    return models_dict[model]


def build_model(shape=(512, 512, 1), model_class = None):
    inp = Input(shape = shape)
    base = get_efficientnet(model_class, shape)
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    inp2 = Input(shape=(4,))
    x2 = tf.keras.layers.GaussianNoise(0.2)(inp2)
    x = Concatenate()([x, x2])
    x = Dropout(0.35)(x)
    x = Dense(1)(x)
    model = Model([inp, inp2], x)
    
    weights = [w for w  in os.listdir('../input/osic-model-weights') if model_class in w][0]
    model.load_weights('../input/osic-model-weights/' + weights)
    return model

model_classes = ['b5']
models = [build_model(shape=(512, 512, 1), model_class = m) for m in model_classes]
print('Number of Models: '+str(len(models)))
from sklearn.model_selection import train_test_split
tr_p, vl_p = train_test_split(P, shuffle=True, train_size = 0.8)
sns.distplot(list(A.values()));
def score(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(sigma, 70)
    delta = np.abs(fvc_true - fvc_pred)
    delta = np.minimum(delta, 1000)
    sq2 = np.sqrt(2)
    metric = (delta / sigma_clip)*sq2 + np.log(sigma_clip*sq2)
    return np.mean(metric)
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
                    tab.append(get_tab(df_train.loc[df_train.Patient == p, :])) 
            if len(x) < 1:
                continue
            tab = np.array(tab) 

            x = np.expand_dims(x, axis=-1) 
            _a = model.predict([x, tab]) 
            a = np.quantile(_a, q / 10)

            percent_true = df_train.Percent.values[df_train.Patient == p]
            fvc_true = df_train.FVC.values[df_train.Patient == p]
            weeks_true = df_train.Weeks.values[df_train.Patient == p]

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


N = len(subs)
sub = subs[0].copy() # ref
sub["FVC"] = 0
sub["Confidence"] = 0
for i in range(N):
    sub["FVC"] += subs[0]["FVC"] * (1/N)
    sub["Confidence"] += subs[0]["Confidence"] * (1/N)
sub[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)
sub[["Patient_Week","FVC","Confidence"]]
