import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

%matplotlib inline

import matplotlib.pyplot as plt

from glob import glob

from PIL import Image

import seaborn as sns

base_skin_dir = os.path.join('..', 'input/skin-cancer-mnist-ham10000')

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}



lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'Melanoma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}
tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)

tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 

tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes

tile_df.sample(3)
tile_df['image'] = tile_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
n_samples = 7

fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))

for n_axs, (type_name, type_rows) in zip(m_axs, 

                                         tile_df.sort_values(['cell_type']).groupby('cell_type')):

    n_axs[0].set_title(type_name)

    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):

        c_ax.imshow(c_row['image'])

        c_ax.axis('off')

fig.savefig('category_samples.png', dpi=300)
tile_df.isnull().sum()
tile_df['age'].fillna((tile_df['age'].mean()), inplace=True)
tile_df.isnull().sum()
fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))

tile_df['cell_type'].value_counts().plot(kind='bar', ax=ax1)
tile_df['dx_type'].value_counts().plot(kind='bar')
tile_df['localization'].value_counts().plot(kind='bar')
tile_df['age'].hist(bins=40)
tile_df['sex'].value_counts().plot(kind='bar')
sns.boxplot(x='dx_type', y='age', data=tile_df)
plt.figure(figsize=(16,6))

sns.boxplot(x='cell_type', y='age', data=tile_df)