#imports
import numpy as np 
import pandas as pd 
import seaborn as sns
from pandas.plotting import scatter_matrix
from IPython.display import display_html
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Loading csvs
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')

bool_columns = ['blue','dual_sim','four_g','three_g','touch_screen','wifi'] 
numeric_columns =  [item for item in df_train.columns if item not in bool_columns] 
# Basic info of dataset
display(df_train.info())
# Basic statistics
pd.options.display.float_format = "{:.2f}".format
display(df_train.describe())
colors = ['#58ACFA','#81F7BE','#F5A9D0','#F3E2A9','#F7D358','#F7FE2E','#BDBDBD','#F78181','#04B4AE','#BF00FF','#FE9A2E']
colors += colors
count = 0
for i in df_train.columns:
    print ('Column ' + i )
    count +=1
    plt.figure(figsize=(4,2))
    sns.distplot(df_train[i].values, kde = False, axlabel = i, color = colors[count])
    plt.show()
# Outliers single column
scaler = StandardScaler(copy=True)
df_data = pd.DataFrame(scaler.fit_transform(df_train[numeric_columns]))
df_data.columns = numeric_columns
plt.figure(figsize=(25,10))
sns.boxplot( data= df_data, palette='rainbow', orient ='h')
plt.show()

# Outliers multiple columns
enc = EllipticEnvelope(support_fraction = 0.6)
enc.fit(df_train)
keys_list = enc.predict(df_train)
outliers_keys = []
for i in range(keys_list.shape[0]):
    if keys_list[i] == -1:
        outliers_keys.append(i)
print(outliers_keys)
# Correlation map
scaler = StandardScaler()
df_corr = pd.DataFrame(scaler.fit_transform(df_train), columns = df_train.columns)
corr = df_corr.corr()
plt.figure(figsize=(25,11))
sns.heatmap(corr, cmap = 'inferno')
plt.show()
# Principal Components Analysis
target_columm = ['price_range'] 
columns_pca =  [item for item in df_train.columns if item not in target_columm] 
scaler = StandardScaler()
pca = PCA(n_components =2)
df_pca = pd.DataFrame(pca.fit_transform(scaler.fit_transform(df_train[columns_pca])), columns = ["PCA_0","PCA_1"])
df_pca = pd.concat((df_pca,df_train['price_range']), axis =1)
print(pca.components_)
# PCA array_0 and array_1 
data_df = pd.DataFrame(pca.components_, columns = columns_pca)
plt.figure(figsize=(25,15))
plt.xticks(rotation='vertical')
sns.barplot( palette='Paired', data=data_df[0:1], orient ='v')
plt.show()
plt.figure(figsize=(25,15))
plt.xticks(rotation='vertical')
sns.barplot( palette='Paired', data=data_df[1:2], orient ='v')
plt.show()
clf = LinearDiscriminantAnalysis(n_components = 2)
scaler = StandardScaler()
clf.fit(scaler.fit_transform(df_train[columns_pca]),df_train['price_range'])
clf_pred = clf.transform(scaler.fit_transform(df_train[columns_pca]))
df_lda = pd.DataFrame(clf_pred, columns = ["LDA_0","LDA_1"])
df_lda = pd.concat((df_lda,df_train['price_range']), axis =1)
# Attempt to separate the data
colors = ['bo','yo','co','mo']
count=0
plt.figure(figsize=(26,10))
for price_range_value in np.unique(df_lda['price_range']):
    df_train_partition = df_lda[df_lda['price_range'] == price_range_value]
    plt.plot(df_train_partition['LDA_0'],df_train_partition['LDA_1'],colors[count])
    count+=1
plt.ylabel("LDA_1")
plt.xlabel("LDA_0")
plt.show()