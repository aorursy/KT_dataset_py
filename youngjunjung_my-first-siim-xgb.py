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
train_df = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test_df = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train_df.head()
test_df.head()
# Check patient_id column is valid feature

train_df['p_id']=train_df['patient_id'].str[3:].astype(int)
train_df['int_sex'] = 0
female_ind=train_df[train_df['sex'] == 'female'].index
train_df.loc[female_ind,'int_sex'] = 1
train_df['int_benign_malignant'] = 0
malignant_ind = train_df[train_df['benign_malignant']=='malignant'].index
train_df.loc[malignant_ind,'int_benign_malignant'] = 1
train_df.head()
train_df[['int_benign_malignant','target']].corr()
train_df.drop(['int_benign_malignant','benign_malignant'],axis=1,inplace=True)
train_df.head(5)
print(train_df['patient_id'].str[:3].unique())
train_df.drop('patient_id',axis=1,inplace=True)
print(train_df[['p_id','target']].corr())
train_df.drop('p_id',axis=1,inplace=True)
train_df.head()
train_df[train_df['sex'].isnull()]['age_approx'].unique()  # NULL sex row == NULL age row
train_df[train_df['sex'].isnull()]['target'].unique()  # All benign row
print(np.round(train_df['target'].value_counts()[1] / train_df.shape[0] * 100,2),'%  malignant')
train_df.groupby('sex')['target'].mean()
import seaborn as sns
import matplotlib.pyplot as plt
plt.hist(train_df['sex'].tolist())
plt.show()
sns.kdeplot(train_df['age_approx'])
train_df.head(5)
train_df['anatom_site_general_challenge'].value_counts()
train_df.groupby('anatom_site_general_challenge')['target'].mean()
target_rate = .1
idx_0 = train_df[train_df.target==0].index
idx_1 = train_df[train_df.target==1].index

sampling_rate = (((1-target_rate)*len(train_df.loc[idx_1]))/(len(train_df.loc[idx_0])*target_rate))
under_sample_len = int(sampling_rate*len(train_df.loc[idx_0]))
print(sampling_rate)
print(under_sample_len)
from sklearn.utils import shuffle
undersampled_idx = shuffle(idx_0,random_state=801, n_samples=under_sample_len)
len(undersampled_idx)
all_idx = list(undersampled_idx)+list(idx_1)
undersampled_train_df = train_df.loc[all_idx].reset_index()
undersampled_train_df
undersampled_train_df['anatom_site_general_challenge'].fillna('NULL',inplace=True)
undersampled_train_df.drop(undersampled_train_df[undersampled_train_df['sex'].isnull()].index,axis=0,inplace=True)
undersampled_train_df.isnull().sum()
undersampled_train_df.drop(undersampled_train_df[undersampled_train_df['age_approx'].isnull()].index,axis=0,inplace=True)
undersampled_train_df.isnull().sum().sum()
undersampled_train_df.drop(['sex','index'],axis=1,inplace=True)
undersampled_train_df.head()
undersampled_train_df.drop('diagnosis',axis=1,inplace=True)
oh_us_train_df=pd.get_dummies(undersampled_train_df,columns=['anatom_site_general_challenge'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_age=scaler.fit_transform(oh_us_train_df['age_approx'].values.reshape(-1,1))
oh_us_train_df['age'] = scaled_age
oh_us_train_df.drop('age_approx',axis=1,inplace=True)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(oh_us_train_df.drop('target',axis=1),oh_us_train_df['target'],
                                                   test_size=0.05,stratify = oh_us_train_df['target'])
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
x_train
y_train
y_test.shape
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
xgb = XGBClassifier(n_estimators=400,learning_rate=.1,max_depth=3,gpu_id=0)
params = {'max_depth':[2,3,5,10],'min_child_weight':[1,3,7],'colsample_bytree':[0.5,0.75],'n_estimators':[100,200,400]}
gridcv = GridSearchCV(xgb,param_grid=params)
gridcv.fit(x_train.drop('image_name',axis=1),y_train,eval_metric='auc',eval_set=[(x_train.drop('image_name',axis=1),y_train),
                                                                                (x_test.drop('image_name',axis=1),y_test)])
print('best param',gridcv.best_params_)
preds=gridcv.predict(x_test.drop('image_name',axis=1))
print(roc_auc_score(preds,y_test.tolist()))
plt.figure(figsize=(24,24))
sns.heatmap(oh_us_train_df.corr(),annot=True)
for x,y in zip(xgb.feature_importances_,x_train.columns[1:]):
    print('# {} feature_importance : {:.4f}'.format(y,x))
import torch
import torch.nn as nn
import torch.nn.init as init
tr_img_path = '../input/siim-isic-melanoma-classification/jpeg/train'
te_img_path = '../input/siim-isic-melanoma-classification/jpeg/test'
import matplotlib.image as img

img.imread(tr_img_path+'/ISIC_0015719.jpg').shape
import torchvision.models as models
mnasnet = models.mnasnet1_0()

    
