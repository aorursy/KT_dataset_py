# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#predefine some of the data type, for memory efficiency 

type_dict={'ncodpers':np.int32, 'ind_ahor_fin_ult1':np.uint8, 'ind_aval_fin_ult1':np.uint8, 

       'ind_cco_fin_ult1':np.uint8,'ind_cder_fin_ult1':np.uint8,

            'ind_cno_fin_ult1':np.uint8,'ind_ctju_fin_ult1':np.uint8,'ind_ctma_fin_ult1':np.uint8,

            'ind_ctop_fin_ult1':np.uint8,'ind_ctpp_fin_ult1':np.uint8,'ind_deco_fin_ult1':np.uint8,

            'ind_deme_fin_ult1':np.uint8,'ind_dela_fin_ult1':np.uint8,'ind_ecue_fin_ult1':np.uint8,

            'ind_fond_fin_ult1':np.uint8,'ind_hip_fin_ult1':np.uint8,'ind_plan_fin_ult1':np.uint8,

            'ind_pres_fin_ult1':np.uint8,'ind_reca_fin_ult1':np.uint8,'ind_tjcr_fin_ult1':np.uint8,

            'ind_valo_fin_ult1':np.uint8,'ind_viv_fin_ult1':np.uint8,

            'ind_recibo_ult1':np.uint8 }



# only loading the top 5,000,000 for demonstration purpose

train=pd.read_csv('../input/train_ver2.csv', nrows=5000000, dtype=type_dict, )

test=pd.read_csv('../input/test_ver2.csv')
train['age']=pd.to_numeric(train.age, errors='coerce')

test['renta']=pd.to_numeric(test.renta, errors='coerce')
le = LabelEncoder()

train_mask = ~train['sexo'].isnull()

train.loc[train_mask, 'sexo'] = le.fit_transform(train['sexo'][train_mask])
# Plot age distribution

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

plt.subplots_adjust(wspace=0.5, hspace=0.5)



sns.distplot(train.age[(train.age>=0) & (train.age<=130)], kde=False, ax=axes[0], axlabel='train age')

sns.distplot(test.age[test.age<=130], kde=False, ax=axes[1], axlabel='test age')
# Plot distribution among different province 

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

plt.subplots_adjust(wspace=0.5, hspace=0.5)



sns.distplot(train.cod_prov[train.cod_prov>=0], kde=False, ax=axes[0], axlabel='train cod_prov')

sns.distplot(test.cod_prov[test.cod_prov>=0], kde=False, ax=axes[1], axlabel='test cod_prov')
up_renta=400000

low_renta=0



train_select1=train.loc[(train.renta<=up_renta) & (train.renta>=low_renta) 

                        & (train.cod_prov>0) & (train.cod_prov<=20) ]



train_select2=train.loc[(train.renta<=up_renta) & (train.renta>=low_renta) 

                        & (train.cod_prov>20) & (train.cod_prov<=40) ]



train_select3=train.loc[(train.renta<=up_renta) & (train.renta>=low_renta) 

                        & (train.cod_prov>40) ]



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,10))

plt.subplots_adjust(hspace=0.5)



boxplot1=sns.boxplot(x='cod_prov', y='renta', data=train_select1, ax=axes[0])

boxplot2=sns.boxplot(x='cod_prov', y='renta', data=train_select2, ax=axes[1])

boxplot3=sns.boxplot(x='cod_prov', y='renta', data=train_select3, ax=axes[2])



boxplot1.set(xlabel='cod_prov 1-20')

boxplot2.set(xlabel='cod_prov 21-39')

boxplot3.set(xlabel='cod_prov 41-52')
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8,14))

plt.subplots_adjust(wspace=1.3, hspace=0.6)

fig_row=0

for col_id in range(24, 48):

    ax_id=col_id-24

    fig_label=train.columns[col_id]

    feat=train.columns[col_id]

    fig_col=(col_id+1)%5

    box_plot=sns.boxplot(y='age', data=train[(train[feat]==1) & 

                                             (train['age']>0) & (train['age']<100)], ax=axes[fig_row][fig_col])

    box_plot.set(xlabel=fig_label)

    if fig_col==4: fig_row+=1
fig, axes = plt.subplots(nrows=8, ncols=3, figsize=(9,18))

plt.subplots_adjust(wspace=0.4, hspace=0.6)

fig_row=0

for col_id in range(24, 48):

    ax_id=col_id-24

    fig_label= train.columns[col_id]

    feat=train.columns[col_id]

    fig_col=col_id%3

    sns.distplot(train.cod_prov[(train[feat]==1) & (train['cod_prov']>=0)], kde=False, 

                 axlabel=fig_label, ax=axes[fig_row][fig_col])

    if fig_col==2: fig_row+=1
fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(8,14))

plt.subplots_adjust(wspace=0.8, hspace=0.5)

# fig.tight_layout()

fig_row=0

for col_id in range(24, 48):

    ax_id=col_id-24

    fig_label= train.columns[col_id]

    feat=train.columns[col_id]

    fig_col=(col_id)%4

    countplot=sns.countplot(x='sexo', data=train[(train[feat]==1) & (train['sexo']>=0)],ax=axes[fig_row][fig_col])

    countplot.set(xlabel=fig_label)

    if fig_col==3: fig_row+=1