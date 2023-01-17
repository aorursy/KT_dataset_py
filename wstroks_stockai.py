import pandas as pd

import numpy as np
base = pd.read_json('/kaggle/input/megahack-stockai/data_stock.json')

base_saida = pd.read_json('/kaggle/input/megahack-stockai/label_stock.json')
base
base['sale'].sum()
base['stock'].sum()
base_saida
previsores= base.iloc[:,0:9].values

classe_1 = base_saida.iloc[:,0].values

classe_2 = base_saida.iloc[:,1].values

classe_3 = base_saida.iloc[:,2].values
import calendar

import datetime  

import matplotlib.pyplot as pl

from datetime import datetime as dt
analisando_vendas_camarao=base.loc[base.name=='Camarão 500g',['name','date','sale']]

analisando_vendas_camarao
pl.title('Quantidade de vendas do Camarão')

pl.xlabel('Data')

pl.ylabel('Quantidade de vendas')

pl.ylim(0, 20)

pl.plot(analisando_vendas_camarao['date'],analisando_vendas_camarao['sale'])

pl.show()



analisando_estoque_camarao=base.loc[base.name=='Camarão 500g',['name','date','stock']]
pl.title('Quantidade de itens em estoque de Camarão')

pl.xlabel('Data')

pl.ylabel('Quantidade de itens em estoque')

pl.ylim(0, 50)

pl.plot(analisando_estoque_camarao['date'],analisando_estoque_camarao['stock'])

pl.show()
analisando_vendas_camarao2=base

analisando_vendas_camarao2['date']=base['date'].dt.month_name()
analisando_vendas_camarao2=base.loc[base.name=='Camarão 500g',['name','date','sale']]
analisando_vendas_camarao2.date
y_pos=np.arange(len(analisando_vendas_camarao2))
analisando_vendas_camarao_j=analisando_vendas_camarao2.loc[analisando_vendas_camarao2.date=='January',['date','sale']]

analisando_vendas_camarao_f=analisando_vendas_camarao2.loc[analisando_vendas_camarao2.date=='February',['date','sale']]

analisando_vendas_camarao_m=analisando_vendas_camarao2.loc[analisando_vendas_camarao2.date=='March',['date','sale']]

analisando_vendas_camarao_f['sale'].sum()
sales_c=[]

sales_c.append(analisando_vendas_camarao_j['sale'].sum())

sales_c.append(analisando_vendas_camarao_f['sale'].sum())

sales_c.append(analisando_vendas_camarao_m['sale'].sum())

mes=['Janeiro','Fevereiro', 'Março']

y_pos=np.arange(len(mes))
pl.bar(y_pos,  sales_c,align='center', alpha=0.5)

pl.xticks(y_pos, mes)

pl.ylabel('Quantidade de vendas')

pl.title('Quantidade de Vendas por mês do Camarão')

pl.show()
classe_1
classe_3
classe_2
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_previsores=LabelEncoder()
previsores[:,0]=label_previsores.fit_transform(previsores[:,0])

previsores[:,4]=label_previsores.fit_transform(previsores[:,4])

previsores[:,5]=label_previsores.fit_transform(previsores[:,5])

previsores[:,7]=label_previsores.fit_transform(previsores[:,7])
previsores
previsores.shape
from sklearn.preprocessing import StandardScaler

escalona = StandardScaler()
previsores=escalona.fit_transform(previsores)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score
def random_forest_1(previsores,classe_1):

    previsores_treinamento1 , previsores_teste1,classe_treinamento1, classe_teste1=train_test_split(previsores,classe_1,test_size=0.25, random_state=0)

    classificador_random_florest1=RandomForestClassifier(n_estimators=40,criterion='entropy',random_state=0)

    classificador_random_florest1.fit(previsores_treinamento1,classe_treinamento1)

    previsao_1=classificador_random_florest1.predict(previsores_teste1)

    precisao_label1=accuracy_score(classe_teste1,previsao_1)

    matriz=confusion_matrix(classe_teste1,previsao_1)

    return precisao_label1, matriz
precisao,matriz=random_forest_1(previsores,classe_1)
precisao
matriz
precisao2,matriz2=random_forest_1(previsores,classe_2)

precisao3,matriz3=random_forest_1(previsores,classe_3)
matriz2,matriz3
array=[]
array.append({'precisão':precisao, 'label': "new_stock", "model":"random_florest"})

array.append({'precisão':precisao2, 'label': "stock_date_expiration", "model":"random_florest"})

array.append({'precisão':precisao3, 'label': "sell_loss", "model":"random_florest"})
array
from sklearn.tree import DecisionTreeClassifier
def tree_decision_1(previsores, classe_1):

    previsores_treinamento1 , previsores_teste1,classe_treinamento1, classe_teste1=train_test_split(previsores,classe_1,test_size=0.25, random_state=0)

    classificador1=DecisionTreeClassifier(criterion='entropy',random_state=0)

    classificador1.fit(previsores_treinamento1,classe_treinamento1)

    previsoes1=classificador1.predict(previsores_teste1)

    precisao1=accuracy_score(classe_teste1,previsoes1)

    matriz1=confusion_matrix(classe_teste1,previsoes1)

    return precisao1, matriz1
precisao1,matriz1=tree_decision_1(previsores,classe_1)

precisao2,matriz2=tree_decision_1(previsores,classe_2)

precisao3,matriz3=tree_decision_1(previsores,classe_3)
array.append({'precisão':precisao1, 'label': "new_stock", "model":"tree_decision"})

array.append({'precisão':precisao2, 'label': "stock_date_expiration", "model":"tree_decision"})

array.append({'precisão':precisao3, 'label': "sell_loss", "model":"tree_decision"})
array
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation
onehotencoder= OneHotEncoder(categories='auto', drop=None, sparse=True, dtype='float64', handle_unknown='error')

previsores = onehotencoder.fit_transform(previsores).toarray()
def neural_network(previsores, classe_1):

    previsores_treinamento1, previsores_teste1,classe_treinamento1, classe_teste1=train_test_split(previsores,classe_1,test_size=0.25, random_state=0)

    classificador=Sequential()

    classificador.add(Dense(units=55,activation='relu',input_dim=420))

    classificador.add(Dense(units=55,activation='relu'))

    classificador.add(Dense(units=1,activation='sigmoid'))#camada binária retorna sigmode

    classificador.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    classificador.fit(previsores_treinamento1,classe_treinamento1,batch_size=10,epochs=100)

    previsoes1=classificador.predict(previsores_teste1)

    previsoes1=(previsoes1>0.5)

    precisao1=accuracy_score(classe_teste1,previsoes1)

    matriz1=confusion_matrix(classe_teste1,previsoes1)

    return precisao1, matriz1;
precisao1,matriz1=neural_network(previsores,classe_1)

precisao2,matriz2=neural_network(previsores,classe_2)

precisao3,matriz3=neural_network(previsores,classe_3)
array.append({'precisão':precisao1, 'label': "new_stock", "model":"neural_network"})

array.append({'precisão':precisao2, 'label': "stock_date_expiration", "model":"neural_network"})

array.append({'precisão':precisao3, 'label': "sell_loss", "model":"neural_network"})
array