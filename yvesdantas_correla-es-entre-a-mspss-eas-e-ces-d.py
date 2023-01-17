import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.cluster import KMeans

import seaborn as sns 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go



init_notebook_mode(connected=True)

%matplotlib inline



import warnings 

warnings.filterwarnings('ignore')
data = pd.read_excel('../input/Planilha do K_Means e do Voroni.xlsx')



data.head()
new_data = pd.read_excel('../input/Planilha do K_Means e do Voroni.xlsx')



new_data = new_data[['IDADE','ESTADO CIVIL EM MESES','RENDA ECONÔMICA PESSOAL',

            'PERÍODO GESTACIONAL BRUTO','FILHOS ANTERIORES','CONSULTAS PRÉ-NATAL',

            'PESSOAS NA CASA','MSPSS_TOTAL','FAMÍLIA','AMIGOS','OUTROS_SIGNIFICATIVOS','EAS TOTAL','EAS_MATERIAL',

             'EAS_EMOCIONAL','EAS_INFORMAÇÃO','EAS_AFETIVO','EAS_INTERACAO_SOCIAL_POSITIVA','CES_D_TOTAL',

             'CES_D_HUMOR','CES_D_SINTOMAS_SOMÁTICOS','CES_D_INTERAÇÕES_SOCIAIS','CES_D_SINTOMAS_MOTORES']]



for column in new_data.columns[1:]:

   new_data[column] = pd.to_numeric(new_data[column], errors='coerce')



new_data = new_data.dropna()



new_data.head()
plt.figure(figsize=(12, 8))



crr = new_data[['MSPSS_TOTAL','FAMÍLIA','AMIGOS','OUTROS_SIGNIFICATIVOS','EAS TOTAL','CES_D_TOTAL',

             'CES_D_HUMOR','CES_D_SINTOMAS_SOMÁTICOS','CES_D_INTERAÇÕES_SOCIAIS','CES_D_SINTOMAS_MOTORES']].corr()



sns.heatmap(crr.iloc[0:4,4:], annot=True, linewidth=.5)
#Curva de correlação MSPSS_TOTAL vs APOIO_EMOCIONAL



sns.lmplot(x='MSPSS_TOTAL', y='EAS_EMOCIONAL', data=new_data, aspect=2)
#Curva de correlação FAMÍLIA vs APOIO_INFORMAÇÃO



sns.lmplot(x='FAMÍLIA', y='EAS_INFORMAÇÃO', data=new_data, aspect=2)
plt.figure(figsize=(12, 8))



crr = new_data[['MSPSS_TOTAL','FAMÍLIA','AMIGOS','OUTROS_SIGNIFICATIVOS','EAS TOTAL','CES_D_TOTAL',

             'CES_D_HUMOR','CES_D_SINTOMAS_SOMÁTICOS','CES_D_INTERAÇÕES_SOCIAIS','CES_D_SINTOMAS_MOTORES']].corr()



sns.heatmap(crr.iloc[0:4,4:], annot=True, linewidth=.5)
#curva de correlação FAMÍLIA E TOTAL_EAS



sns.lmplot(x='FAMÍLIA', y='EAS TOTAL', data=new_data, aspect=2)
#normalizando os dados



min_max_scaler = preprocessing.MinMaxScaler()



features = min_max_scaler.fit_transform(np.array(new_data))



pd.DataFrame(features).head()
#check the optimal k value



Sum_of_squared_distances = []

K = range(1,15)

for k in K:

    km = KMeans(n_clusters=k)

    km = km.fit(features)

    Sum_of_squared_distances.append(km.inertia_)

    



plt.plot(K, Sum_of_squared_distances, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS TOTAL', y='MSPSS_TOTAL', data=new_data, aspect=2, hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_MATERIAL', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_EMOCIONAL', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_AFETIVO', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_INFORMAÇÃO', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_INTERACAO_SOCIAL_POSITIVA', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_TOTAL', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_HUMOR', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_SINTOMAS_SOMÁTICOS', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_INTERAÇÕES_SOCIAIS', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_SINTOMAS_MOTORES', y='MSPSS_TOTAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS TOTAL', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_MATERIAL', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_EMOCIONAL', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_AFETIVO', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_INFORMAÇÃO', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_INTERACAO_SOCIAL_POSITIVA', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_TOTAL', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_HUMOR', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_SINTOMAS_SOMÁTICOS', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_INTERAÇÕES_SOCIAIS', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_SINTOMAS_MOTORES', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS TOTAL', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_MATERIAL', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_EMOCIONAL', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_INFORMAÇÃO', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_AFETIVO', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_INTERACAO_SOCIAL_POSITIVA', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_TOTAL', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_HUMOR', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_SINTOMAS_SOMÁTICOS', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_INTERAÇÕES_SOCIAIS', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_SINTOMAS_MOTORES', y='AMIGOS', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS TOTAL', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_MATERIAL', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_EMOCIONAL', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_INFORMAÇÃO', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_AFETIVO', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='EAS_INTERACAO_SOCIAL_POSITIVA', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_TOTAL', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_HUMOR', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_SINTOMAS_SOMÁTICOS', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_INTERAÇÕES_SOCIAIS', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='CES_D_SINTOMAS_MOTORES', y='FAMÍLIA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='ESTADO CIVIL EM MESES', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='FAMÍLIA', y='ESTADO CIVIL EM MESES', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='AMIGOS', y='ESTADO CIVIL EM MESES', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='OUTROS_SIGNIFICATIVOS', y='ESTADO CIVIL EM MESES', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='RENDA ECONÔMICA PESSOAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='FAMÍLIA', y='RENDA ECONÔMICA PESSOAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='AMIGOS', y='RENDA ECONÔMICA PESSOAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='OUTROS_SIGNIFICATIVOS', y='RENDA ECONÔMICA PESSOAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='PERÍODO GESTACIONAL BRUTO', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='FAMÍLIA', y='PERÍODO GESTACIONAL BRUTO', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='AMIGOS', y='PERÍODO GESTACIONAL BRUTO', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='OUTROS_SIGNIFICATIVOS', y='PERÍODO GESTACIONAL BRUTO', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='CONSULTAS PRÉ-NATAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='FAMÍLIA', y='CONSULTAS PRÉ-NATAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='AMIGOS', y='CONSULTAS PRÉ-NATAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='OUTROS_SIGNIFICATIVOS', y='CONSULTAS PRÉ-NATAL', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='PESSOAS NA CASA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='FAMÍLIA', y='PESSOAS NA CASA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='AMIGOS', y='PESSOAS NA CASA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='OUTROS_SIGNIFICATIVOS', y='PESSOAS NA CASA', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='IDADE', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='OUTROS_SIGNIFICATIVOS', y='IDADE', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='AMIGOS', y='IDADE', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
kmeans = KMeans(n_clusters=3, random_state=0)



kmeans.fit(features)



new_data['Cluster'] = kmeans.labels_



def converter(change):

    if change == 1:

        return 'Grupo 2'

    elif change == 2:

        return 'Grupo 1'

    else:

        return 'Grupo 3'



new_data['Cluster'] = new_data['Cluster'].apply(converter)



sns.lmplot(x='FAMÍLIA', y='IDADE', data=new_data, aspect=2,  hue='Cluster', fit_reg=False)
new_data['Filhos Anteriores'] = data['TEVE FILHOS ANTERIORES A GRAVIDEZ?']



def converter(change):

    if change == 1:

        return 'Primípara'

    else:

        return 'Multípera'



new_data['Filhos Anteriores'] = new_data['Filhos Anteriores'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='FAMÍLIA', data=new_data, aspect=2,  hue='Filhos Anteriores', fit_reg=False)
new_data['Filhos Anteriores'] = data['TEVE FILHOS ANTERIORES A GRAVIDEZ?']



def converter(change):

    if change == 1:

        return 'Primípara'

    else:

        return 'Multípera'



new_data['Filhos Anteriores'] = new_data['Filhos Anteriores'].apply(converter)



sns.lmplot(x='AMIGOS', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Filhos Anteriores', fit_reg=False)
new_data['Período Gestacional'] = data['PERÍODO GESTACIONAL CATEGORIZADO']



def converter(change):

    if change == 1:

        return 'Segundo Trimestre'

    elif change == 2:

        return 'Terceiro Trimestre'

    else:

        return 'Primeiro Trimestre'



new_data['Período Gestacional'] = new_data['Período Gestacional'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='FAMÍLIA', data=new_data, aspect=2,  hue='Período Gestacional', fit_reg=False)
new_data['Período Gestacional'] = data['PERÍODO GESTACIONAL CATEGORIZADO']



def converter(change):

    if change == 1:

        return 'Segundo Trimestre'

    elif change == 2:

        return 'Terceiro Trimestre'

    else:

        return 'Primeiro Trimestre'



new_data['Período Gestacional'] = new_data['Período Gestacional'].apply(converter)



sns.lmplot(x='AMIGOS', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Período Gestacional', fit_reg=False)
new_data['Tipo de Gestação'] = data['TIPO DE GESTAÇÃO']



def converter(change):

    if change == 1:

        return 'Adulta Jovem'

    elif change == 2:

        return 'Adolescente'

    else:

        return 'Tardia'



new_data['Tipo de Gestação'] = new_data['Tipo de Gestação'].apply(converter)



sns.lmplot(x='MSPSS_TOTAL', y='FAMÍLIA', data=new_data, aspect=2,  hue='Tipo de Gestação', fit_reg=False)
new_data['Tipo de Gestação'] = data['TIPO DE GESTAÇÃO']



def converter(change):

    if change == 1:

        return 'Adulta Jovem'

    elif change == 2:

        return 'Adolescente'

    else:

        return 'Tardia'



new_data['Tipo de Gestação'] = new_data['Tipo de Gestação'].apply(converter)



sns.lmplot(x='AMIGOS', y='OUTROS_SIGNIFICATIVOS', data=new_data, aspect=2,  hue='Tipo de Gestação', fit_reg=False)