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
data_train_file = '/kaggle/input/digit-recognizer/train.csv'
data_test_file = '/kaggle/input/digit-recognizer/test.csv'

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)
print(df_train.head(5))
df_train.shape
l = df_train['label']
d = df_train.drop('label',axis=1)
print("Labels\n",l.head(5),"\n")
print("784 Dimensions\n",d.head(5))
import matplotlib.pyplot as plt
# display a number
plt.figure(figsize=(2,2))
idx=0
grid_data = d.iloc[idx].to_numpy().reshape(28,28) #Convert a flatten list into array matrix another word reshaping  
plt.imshow(grid_data, cmap ="gray")
plt.show()
print(l[idx])
labels =l.head(15000)
data =d.head(15000)
print( "The shape of sample data = ",data.shape)
#Data-preprocessing 
from sklearn.preprocessing import StandardScaler
standard_data = StandardScaler().fit_transform(data)
standard_data.shape
# PCA using Scikit-Learn
s= standard_data
from sklearn import decomposition
pca = decomposition.PCA()
pca.n_components =2 # number of component =2 
pca_data = pca.fit_transform(s)
print("shape of PCA" ,pca_data.shape)
print(pca_data.T.shape)
print(labels.shape)
print(labels)
# attaching the Label with data point
pca_data = np.vstack((pca_data.T,labels)).T
print(pca_data.shape)
pca_df =pd.DataFrame(data=pca_data,columns=("1st_PCA","2nd_PCA","label"))
sn.FacetGrid(pca_df,hue="label",height=6).map(plt.scatter,'1st_PCA','2nd_PCA').add_legend()
plt.show()
pca.n_components = 784
pca_data = pca.fit_transform(s)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)



plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()
print(pca.explained_variance_.shape)
np.sum(pca.explained_variance_)
