import numpy as np

import pandas as pd
country_data_full = pd.read_csv('/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv')

country_data_full.head()
# We're going to use a copy of the original data as input to the autoencoder

data_input = country_data_full.drop('country',axis=1)
# normalizes data so the weights and biases of the autoencoder aren't off the charts

for col in data_input.columns.to_list():

    col_mean = data_input[col].mean()

    data_input[col] = data_input[col]/col_mean
data_input.head()
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input,Dense

import matplotlib.pyplot as plt

plt.style.use('seaborn')
# Building a simple autoencoder to cluster countries



# Encoder

encoder = Sequential()

encoder.add(Input(shape=(9,)))

encoder.add(Dense(30,activation='relu'))

encoder.add(Dense(20,activation='relu'))

encoder.add(Dense(15,activation='relu'))

encoder.add(Dense(10,activation='relu'))

encoder.add(Dense(5,activation='relu'))



# Decoder

decoder = Sequential()

decoder.add(Dense(10,activation='relu'))

decoder.add(Dense(15,activation='relu'))

decoder.add(Dense(20,activation='relu'))

decoder.add(Dense(30,activation='relu'))

decoder.add(Dense(9,activation='relu'))
autoencoder = Sequential([encoder,decoder])
autoencoder.compile(loss='mse',

                   optimizer='Adam',

                   metrics = ['accuracy'])

autoencoder_res = autoencoder.fit(data_input,data_input,

               epochs = 1500,

               verbose = False)
metrics_fig = plt.figure(figsize=(12,5))



loss_ax = metrics_fig.add_subplot(1,2,1)

loss_ax.plot(autoencoder_res.history['loss'],label='loss')

loss_ax.set_xlabel('epoch')

loss_ax.set_ylabel('loss')

loss_ax.set_title('loss vs epoch')

loss_ax.legend()



acc_ax = metrics_fig.add_subplot(1,2,2)

acc_ax.plot(autoencoder_res.history['accuracy'],label='accuracy',color='green')

acc_ax.set_xlabel('epoch')

acc_ax.set_ylabel('accuracy')

acc_ax.set_title('accuracy vs epoch')

acc_ax.legend()



plt.show()
all_layers = autoencoder.trainable_variables

print('number of total tensors = ', len(all_layers))

# Tensorflow puts intermediate tensors to link layers, so since we want layer 5 (clustering layer), we check index 9
print('length of the ninth tensor = ',len(all_layers[9].numpy()))

# This means all_layers[9].numpy() holds our clustering dimension parameters
d1 = []

d2 = []

d3 = []

d4 = []

d5 = []

dim_dict = {'d1':d1,

           'd2':d2,

           'd3':d3,

           'd4':d4,

           'd5':d5}

for i in range(len(data_input)):

    prediction = pd.Series(encoder.predict(data_input.iloc[i].values.reshape(-1,9))[0])

    for j in range(len(prediction)):

        dim_dict['d'+str(j+1)].append(prediction[j])

        
clustered_dims = pd.DataFrame(dim_dict)
clustered_dims.head() 

# Keep in mind the index of the row corresponds directly to the index of country in the original table
quant_columns = ['child_mort','exports','health','imports','income','inflation','life_expec',

                'total_fer','gdpp']

for col in quant_columns:

    percentile_scores = np.percentile(country_data_full[col].values,[25,50,75])

    raw_vals = country_data_full[col].values

    quartile = []

    for val in raw_vals:

        if val<percentile_scores[0]:

            quartile.append(1)

        elif val<percentile_scores[1]:

            quartile.append(2)

        elif val<percentile_scores[2]:

            quartile.append(3)

        else:

            quartile.append(4)

    quartile_ser = pd.Series(quartile)

    country_data_full[col+'_quartile'] = quartile_ser
quartile_cols = [col+'_quartile' for col in quant_columns]
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from mpl_toolkits import mplot3d
# Grouping dimensions together by "country"

dims_zipped = []

for (dim1,dim2,dim3,dim4,dim5) in zip(d1,d2,d3,d4,d5):

    dims_zipped.append((dim1,dim2,dim3,dim4,dim5))
# Performing PCA and taking the first three components

pca = PCA(n_components=3)

pca_res = pca.fit_transform(dims_zipped)
# Storing the first three components in a DataFrame

pca_df = pd.DataFrame(columns=['pca1','pca2','pca3'])

pca_df['pca1'] = pca_res[:,0]

pca_df['pca2'] = pca_res[:,1]

pca_df['pca3'] = pca_res[:,2]
pca_fig1 = plt.figure(figsize=(7,18))



ax1 = pca_fig1.add_subplot(4,1,1)

a1 = ax1.scatter(pca_df['pca1'],pca_df['pca2'],

            c=country_data_full['life_expec'],

            cmap='viridis',

            alpha=0.6)

ax1.set_xlabel('first component')

ax1.set_ylabel('second component')

ax1.set_title('life expectancy')

pca_fig1.colorbar(a1)



ax2 = pca_fig1.add_subplot(4,1,2)

a2 = ax2.scatter(pca_df['pca1'],pca_df['pca2'],

            c=country_data_full['health'],

            cmap='viridis',

            alpha=0.6)

ax2.set_xlabel('first component')

ax2.set_ylabel('second component')

ax2.set_title('health score')

plt.text(1,6,'two distinct tiers')

plt.arrow(0.75,5.75,-1.5,-3.75,

         width=0.1,

         color='black')

pca_fig1.colorbar(a2)



ax3 = pca_fig1.add_subplot(4,1,3)

a3 = ax3.scatter(pca_df['pca1'],pca_df['pca2'],

             c=country_data_full['child_mort'],

             cmap='viridis',

             alpha=0.6)

ax3.set_xlabel('first component')

ax3.set_ylabel('second component')

ax3.set_title('child mortality rate')

pca_fig1.colorbar(a3)



ax4 = pca_fig1.add_subplot(4,1,4)

a4 = ax4.scatter(pca_df['pca1'],pca_df['pca2'],

            c=country_data_full['total_fer'],

            cmap='viridis',

            alpha=0.6)

ax4.set_xlabel('first component')

ax4.set_ylabel('second component')

ax4.set_title('fertility rate')

pca_fig1.colorbar(a4)



plt.tight_layout()

plt.show()
pca_fig2 = plt.figure(figsize=(7,21))



ax11 = pca_fig2.add_subplot(5,1,1)

a11 = ax11.scatter(pca_df['pca1'],pca_df['pca2'],

            c=country_data_full['gdpp'],

            cmap='plasma', # note that this colormap is different for distinguishable stratification in the crescent.

            alpha=0.6)

ax11.set_xlabel('first component')

ax11.set_ylabel('second component')

ax11.set_title('gdp')

pca_fig2.colorbar(a11)



ax21 = pca_fig2.add_subplot(5,1,2)

a21 = ax21.scatter(pca_df['pca1'],pca_df['pca2'],

            c=country_data_full['imports'],

            cmap='viridis',

            alpha=0.6)

ax21.set_xlabel('first component')

ax21.set_ylabel('second component')

ax21.set_title('imports')

pca_fig2.colorbar(a21)



ax31 = pca_fig2.add_subplot(5,1,3)

a31 = ax31.scatter(pca_df['pca1'],pca_df['pca2'],

             c=country_data_full['exports'],

             cmap='viridis',

             alpha=0.6)

ax31.set_xlabel('first component')

ax31.set_ylabel('second component')

ax31.set_title('exports')

pca_fig2.colorbar(a31)



ax41 = pca_fig2.add_subplot(5,1,4)

a41 = ax41.scatter(pca_df['pca1'],pca_df['pca2'],

            c=country_data_full['income'],

            cmap='viridis',

            alpha=0.6)

ax41.set_xlabel('first component')

ax41.set_ylabel('second component')

ax41.set_title('income')

pca_fig2.colorbar(a41)



ax51 = pca_fig2.add_subplot(5,1,5)

a51 = ax51.scatter(pca_df['pca1'],pca_df['pca2'],

            c=country_data_full['inflation'],

            cmap='viridis',

            alpha=0.6)

ax51.set_xlabel('first component')

ax51.set_ylabel('second component')

ax51.set_title('inflation')

pca_fig2.colorbar(a51)



plt.tight_layout()

plt.show()
from PIL import Image

import glob
# Now we perform tSNE to see if we can get an alternate visualization. 

# Note: I played around with the perplexity parameter a bit in order to get some decent groupings!

tsne2 = TSNE(n_components=2,perplexity=40) 

tsne3 = TSNE(n_components=3,perplexity=25)

embedded2 = tsne2.fit_transform(dims_zipped)

embedded3 = tsne3.fit_transform(dims_zipped)
tsne_2dfigs = plt.figure(figsize=(12,23))



ax2d1 = tsne_2dfigs.add_subplot(5,2,1)

tsne21 = ax2d1.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['life_expec'],

            cmap='plasma')

ax2d1.set_xlabel('embedded dimension 1')

ax2d1.set_ylabel('embedded dimension 2')

ax2d1.set_title('life expectancy')

plt.colorbar(tsne21)



ax2d2 = tsne_2dfigs.add_subplot(5,2,2)

tsne22 = ax2d2.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['gdpp'],

            cmap='plasma')

ax2d2.set_xlabel('embedded dimension 1')

ax2d2.set_ylabel('embedded dimension 2')

ax2d2.set_title('gdp')

plt.colorbar(tsne22)



ax2d3 = tsne_2dfigs.add_subplot(5,2,3)

tsne23 = ax2d3.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['health'],

            cmap='plasma')

ax2d3.set_xlabel('embedded dimension 1')

ax2d3.set_ylabel('embedded dimension 2')

ax2d3.set_title('health score')

plt.colorbar(tsne23)



ax2d4 = tsne_2dfigs.add_subplot(5,2,4)

tsne24 = ax2d4.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['imports'],

            cmap='plasma')

ax2d4.set_xlabel('embedded dimension 1')

ax2d4.set_ylabel('embedded dimension 2')

ax2d4.set_title('imports')

plt.colorbar(tsne24)



ax2d5 = tsne_2dfigs.add_subplot(5,2,5)

tsne25 = ax2d5.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['child_mort'],

            cmap='plasma')

ax2d5.set_xlabel('embedded dimension 1')

ax2d5.set_ylabel('embedded dimension 2')

ax2d5.set_title('child mortality rate')

plt.colorbar(tsne25)



ax2d6 = tsne_2dfigs.add_subplot(5,2,6)

tsne26 = ax2d6.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['exports'],

            cmap='plasma')

ax2d6.set_xlabel('embedded dimension 1')

ax2d6.set_ylabel('embedded dimension 2')

ax2d6.set_title('exports')

plt.colorbar(tsne26)



ax2d7 = tsne_2dfigs.add_subplot(5,2,7)

tsne27 = ax2d7.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['total_fer'],

            cmap='plasma')

ax2d7.set_xlabel('embedded dimension 1')

ax2d7.set_ylabel('embedded dimension 2')

ax2d7.set_title('fertility rate')

plt.colorbar(tsne27)



ax2d8 = tsne_2dfigs.add_subplot(5,2,8)

tsne28 = ax2d8.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['income'],

            cmap='plasma')

ax2d8.set_xlabel('embedded dimension 1')

ax2d8.set_ylabel('embedded dimension 2')

ax2d8.set_title('income')

plt.colorbar(tsne28)



ax2d10 = tsne_2dfigs.add_subplot(5,2,10)

tsne210 = ax2d10.scatter(embedded2[:,0],embedded2[:,1],

            c = country_data_full['inflation'],

            cmap='plasma')

ax2d10.set_xlabel('embedded dimension 1')

ax2d10.set_ylabel('embedded dimension 2')

ax2d10.set_title('inflation')

plt.colorbar(tsne210)





plt.tight_layout()

plt.show()
tsne3d_df = pd.DataFrame(columns=['embedded dimension 1','embedded dimension 2','embedded dimension 3'])

for i in range(len(tsne3d_df.columns)):

    tsne3d_df[tsne3d_df.columns.to_list()[i]] = embedded3[:,i]
import plotly.express as px

df = tsne3d_df

tsne3d_fig = px.scatter_3d(df,

                          x = 'embedded dimension 1', y = 'embedded dimension 2', z = 'embedded dimension 3',

                          color = country_data_full['income'])

tsne3d_fig.show()
tsne3d_quart = px.scatter_3d(df,

                          x = 'embedded dimension 1', y = 'embedded dimension 2', z = 'embedded dimension 3',

                          color = country_data_full['income_quartile'])

tsne3d_quart.show()