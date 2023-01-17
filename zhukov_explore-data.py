import pandas as pd

import numpy as np



df_train = pd.read_csv('../input/train_ver2.csv', nrows=100) #

df_test = pd.read_csv('../input/test_ver2.csv', nrows=100)
df_train.head()
df_test.head()
df_train.columns
parm_names = ['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo',

       'age', 'fecha_alta', 'ind_nuevo', 'antiguedad', 'indrel',

       'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext',

       'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',

       'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']

product_names = ['fecha_dato', 'ncodpers','fecha_dato', 'ncodpers','ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1', 

                 'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',

                 'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1',

                 'ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1',

                 'ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
df_train = pd.read_csv('../input/train_ver2.csv', usecols=product_names) #
df_test = pd.read_csv('../input/test_ver2.csv', usecols=['fecha_dato', 'ncodpers'])
print("Number rows in train data: ", df_train.shape)

print("Number rows in test data: ", df_test.shape)
df_train.ix[df_train.ncodpers == 1050613, ]
print("Number customers in train data: ", df_train.ncodpers.unique().shape[0])

print("Number customers in test data: ", df_test.ncodpers.unique().shape[0])
train_cast = set(df_train.ncodpers.unique())

test_cast = set(df_test.ncodpers.unique())

test_in_train_cat = train_cast.intersection(test_cast)

only_train_cast = train_cast.difference(test_cast)

print("From {} customers in test {} are in train ".format(len(test_cast), len(test_cast.intersection(train_cast))))

print("From {} customers in train {} are in test ".format(len(train_cast), len(train_cast.intersection(test_cast))))

print("{} customers are only in train".format(len(only_train_cast)))
df_train.groupby('fecha_dato')['ncodpers'].agg([('all customers', 'count')]).join(

df_train[df_train.ncodpers.isin(test_cast)].groupby('fecha_dato')['ncodpers'].agg([('test customers in train', 'count')])).join(

df_train[df_train.ncodpers.isin(only_train_cast)].groupby('fecha_dato')['ncodpers'].agg([('only train customers', 'count')])).join(

    df_test.groupby('fecha_dato')['ncodpers'].agg([('test', 'count')]), how = 'outer').fillna(0)
df_train.isnull().sum()