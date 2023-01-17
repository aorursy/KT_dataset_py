! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
! conda install -c rdkit rdkit -y
import sys 
sys.path.append('/usr/local/lib/python3.7/site-packages/')
import warnings
warnings.filterwarnings('ignore')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
chem_01 = pd.read_csv("/kaggle/input/carcinogen-data/carcinogen_all_data.csv")
chem_02 = pd.read_csv("/kaggle/input/carcinogen-data/carcinogen_data_1.csv")
chem_03 = pd.read_csv("/kaggle/input/carcinogen-data/carcinogen_binary.csv")
chem_02.info()
chem_03.info()
chem_02_copy = chem_02.STRUCTURE_SMILES
chem_new = pd.concat([chem_03, chem_02_copy], axis = 1)
chem_new.shape
chem_new.info()
chem_new.drop([263],axis = 0, inplace = True)
from rdkit import Chem
mol_lst=[]
for i in chem_new.STRUCTURE_SMILES:
    mol=Chem.MolFromSmiles(i)
    mol_lst.append(mol)
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors



desc_lst=[i[0] for i in Descriptors._descList]
descriptor=MoleculeDescriptors.MolecularDescriptorCalculator(desc_lst)
descrs = []
for i in range(len(mol_lst)):
    descrs.append(descriptor.CalcDescriptors(mol_lst[i]))




molDes=pd.DataFrame(descrs,columns=desc_lst)
molDes.head(20)
cor_mat = molDes.corr()
sns.heatmap(cor_mat);
corr_feature = []
for i in range(len(cor_mat.columns)):
    for j in range(i):
        if abs(cor_mat.iloc[i,j] > 0.85): #abs is absolute
            corr_feature.append(cor_mat.columns[i])
corr_feature = list(set(corr_feature))
print(corr_feature)
len(corr_feature)
molDes.drop(columns = corr_feature, axis =1, inplace = True)
print(molDes.shape)
molDes.head()
print(chem_new['Carcinogenic Potency Expressed as P or NP'].value_counts())
print(chem_new['Carcinogenic Potency Expressed as P or NP'].unique())
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
chem_new['Carcinogenic Potency Expressed as P or NP'] = label_encoder.fit_transform(chem_new['Carcinogenic Potency Expressed as P or NP'])
print(chem_new['Carcinogenic Potency Expressed as P or NP'].unique())
print(chem_new['Carcinogenic Potency Expressed as P or NP'].value_counts())
chem_new
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
x = molDes
y = chem_new['Carcinogenic Potency Expressed as P or NP'] # target value

## feature selection with Extra Tree Regressor
model = ExtraTreeRegressor()
model.fit(x,y)

feat_importance = pd.Series(model.feature_importances_, index=x.columns)
feat_importance = feat_importance.sort_values(ascending = False)
imp_feat = feat_importance.head(20)
print(type(imp_feat))
imp_feat
molDes_feat = pd.DataFrame(imp_feat)
print(molDes_feat.index)
print(molDes_feat.shape)
mol_feature = molDes[['EState_VSA7', 'qed', 'PEOE_VSA1', 'MinAbsEStateIndex', 'EState_VSA6',
       'MinEStateIndex', 'EState_VSA2', 'BCUT2D_MRHI', 'EState_VSA1',
       'BalabanJ', 'PEOE_VSA2', 'EState_VSA10', 'VSA_EState9', 'SlogP_VSA1',
       'NumAromaticRings', 'MinPartialCharge', 'SMR_VSA5', 'BCUT2D_CHGLO',
       'PEOE_VSA3', 'SlogP_VSA2']]
mol_feature.shape
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
log_reg=LogisticRegression()
X=mol_feature
Y=chem_new['Carcinogenic Potency Expressed as P or NP']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
log_reg=log_reg.fit(X_train,Y_train)
print("Training score: {} \nValidation score: {} ".format(round(log_reg.score(X_train, Y_train),3),
                                                          round(log_reg.score(X_test,Y_test),3)))
from sklearn.model_selection import cross_val_score, GridSearchCV
GridSearchCV.get_params(log_reg)
log_para = {'fit_intercept':[True],'max_iter': [100,150,200,250,300],
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'penalty': ['l2','l1','elasticnet','none']}
log=GridSearchCV(LogisticRegression(),param_grid=log_para).fit(X_train, Y_train).best_estimator_
log
log_reg_1 = log.fit(X_train,Y_train)
print("Training score: {} \nValidation score: {} ".format(round(log_reg_1.score(X_train, Y_train),3),round(log_reg_1.score(X_test,Y_test),3)))
Y_pred = log_reg_1.predict(X_test)
print('Mean squared error (MSE): %.2f'% mean_squared_error(Y_test, Y_pred))
ax = sns.regplot(Y_test, Y_pred, scatter_kws={'alpha':0.5})
ax.set_xlabel('experimental CAT: RandomFor')
ax.set_ylabel('predicted CAT: RandomFor')
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
plt.show()
Y_pred
print(classification_report(Y_test,Y_pred))
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
RFECV.get_params(RandomForestRegressor)
rf_params={'n_estimators':[90,100,200],'max_depth':[80,100,200],
           'max_features':['auto', 'sqrt', 'log2'],'oob_score':[True],'min_samples_split':[2,8,32],
            'criterion':['mse', 'mae']}
rf=GridSearchCV(RandomForestRegressor(),param_grid=rf_params).fit(X_train, Y_train).best_estimator_
rf
rf.fit(X_train,Y_train)
print("Training score: {} \nValidation score: {} ".format(round(rf.score(X_train, Y_train),3),round(rf.score(X_test,Y_test),3)))
