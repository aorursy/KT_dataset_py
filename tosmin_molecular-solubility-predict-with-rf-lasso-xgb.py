! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh

! chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh

! bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local

! conda install -c rdkit rdkit -y

import sys 

sys.path.append('/usr/local/lib/python3.7/site-packages/')

import warnings

warnings.filterwarnings('ignore')
! wget https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt
! wget https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv
import pandas as pd

import numpy as np

data=pd.read_csv('delaney.csv')

data.head()
data_slice=data[['measured log(solubility:mol/L)','ESOL predicted log(solubility:mol/L)']]
data_slice2=data_slice[['measured log(solubility:mol/L)']]
from rdkit import Chem

mol_lst=[]

for i in data.SMILES:

    mol=Chem.MolFromSmiles(i)

    mol_lst.append(mol)
from rdkit.Chem import Descriptors


from rdkit.ML.Descriptors import MoleculeDescriptors
desc_lst=[i[0] for i in Descriptors._descList]

descriptor=MoleculeDescriptors.MolecularDescriptorCalculator(desc_lst)
descrs = []

for i in range(len(mol_lst)):

    descrs.append(descriptor.CalcDescriptors(mol_lst[i]))
len(descrs)
df=pd.DataFrame(descrs,columns=desc_lst)

df.head()
import numpy as np

from scipy.stats import mannwhitneyu
Stat,p=mannwhitneyu(data_slice[['measured log(solubility:mol/L)']], df[['MolLogP']])

print("Statistics=%.3f, p=%.3f"%(Stat,p))

alpha=0.05

if p>alpha:

    print("reject to fail the null hypothesis(same distributation)")

else:

    print("reject the null hypothesis(different hypothesis)")
data_slice2.rename(columns = {'measured log(solubility:mol/L)':'ground_sol'}, inplace = True) 
#it shows the two samples does not belongs to same population

#or

#the probability is 50% that a randomly drawn member of the first population will not exceed a member of the second population.

#form this we can infer that MolLogP 
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error
data_slice2.shape
df.shape
X=df

Y=data_slice2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train.shape, Y_train.shape
# loop to build a tree, make predictions and get the roc-auc

# for each feature of the train set

# Regression is used for feature selection and model selection
mse_values = []

for feature in X_train.columns:

    clf = DecisionTreeRegressor()

    clf.fit(X_train[feature].fillna(0).to_frame(), Y_train)

    Y_scored = clf.predict(X_test[feature].fillna(0).to_frame())

    mse_values.append(mean_squared_error(Y_test, Y_scored))
mse_values = pd.Series(mse_values)

mse_values.index = X_train.columns

mse_values.sort_values(ascending=False)
mse_values.sort_values(ascending=True).head(20)
df_temp=pd.DataFrame(mse_values.sort_values(ascending=True).head(20))
df_temp.index
df_feature=df[['MolLogP', 'ExactMolWt', 'HeavyAtomMolWt', 'PEOE_VSA6', 'Kappa1',

       'MolWt', 'SMR_VSA7', 'SMR_VSA10', 'HallKierAlpha', 'Chi0v', 'MolMR',

       'LabuteASA', 'fr_benzene', 'NumAromaticCarbocycles', 'Chi0',

       'FpDensityMorgan1', 'NumValenceElectrons', 'EState_VSA9', 'SlogP_VSA8',

       'HeavyAtomCount']]
df_feature.shape
X=df_feature

Y=data_slice2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train.shape, Y_train.shape
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100)

model.fit(X_train, Y_train)

r2 = model.score(X_test, Y_test)

r2
Y_pred = model.predict(X_test)
import seaborn as sns

import matplotlib.pyplot as plt

Y_pred.max()
ax = sns.regplot(Y_test, Y_pred, scatter_kws={'alpha':0.5})

ax.set_xlabel('experimental solubuility values: RandomFor')

ax.set_ylabel('predicted solubuility values: RandomFor')

ax.set_xlim(-9, 2)

ax.set_ylim(-9, 2)

plt.show()
from sklearn import linear_model

clf = linear_model.Lasso(alpha=0.1)

clf.fit(X_train, Y_train)

r2 = clf.score(X_test, Y_test)

r2
Y_predlaso = clf.predict(X_test)
ax = sns.regplot(Y_test, Y_predlaso, scatter_kws={'alpha':0.5})

ax.set_xlabel('experimental solubuility values: LASSO')

ax.set_ylabel('predicted solubuility values: LASSO')

ax.set_xlim(-9, 2)

ax.set_ylim(-9, 2)

plt.show()
import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=50)
xgb_model.fit(X_train, Y_train)
Y_predxgb = xgb_model.predict(X_test)
mse=mean_squared_error(Y_test, Y_predxgb)



print(np.sqrt(mse))
ax = sns.regplot(Y_test, Y_predxgb, scatter_kws={'alpha':0.5})

ax.set_xlabel('experimental solubuility values: xgb')

ax.set_ylabel('predicted solubuility values: xgb')

ax.set_xlim(-9, 2)

ax.set_ylim(-9, 2)

plt.show()