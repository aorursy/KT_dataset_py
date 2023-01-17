#imports
import numpy as np 
import pandas as pd 
import seaborn as sns
from pandas.plotting import scatter_matrix
from IPython.display import display_html
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
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
# LDA
clf = LinearDiscriminantAnalysis(n_components = 2)
scaler = StandardScaler()
clf.fit(scaler.fit_transform(df_train[columns_pca]),df_train['price_range'])
clf_pred = clf.transform(scaler.fit_transform(df_train[columns_pca]))
df_lda = pd.DataFrame(clf_pred, columns = ["LDA_0","LDA_1"])
df_lda = pd.concat((df_lda,df_train['price_range']), axis =1)
print(clf.coef_ )
df_numeric = df_train[numeric_columns].drop(columns = 'price_range')
# Linear Regression
reg_lin = LinearRegression(n_jobs = -1)
reg_rid = Ridge(alpha=5)
X_train, X_test, y_train, y_test = train_test_split(df_numeric, df_train['price_range'], test_size=0.7)
reg_lin.fit(X_train, y_train)
y_pred = reg_lin.predict(X_test)
rms = np.sqrt(mean_squared_error(y_pred,y_test))
print('rms = ', rms)
reg_rid = Ridge(alpha=5)
X_train, X_test, y_train, y_test = train_test_split(df_numeric, df_train['price_range'], test_size=0.7)
reg_rid.fit(X_train, y_train)
y_pred = reg_rid.predict(X_test)
rms = np.sqrt(mean_squared_error(y_pred,y_test))
print('rms = ', rms)
reg_log = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(df_numeric, df_train['price_range'], test_size=0.7)
reg_log.fit(X_train, y_train)
y_pred = reg_log.predict(X_test)
rms = np.sqrt(mean_squared_error(y_pred,y_test))
print('rms = ', rms)
clf = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(df_numeric, df_train['price_range'], test_size=0.7)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_pred,y_test)
print('acc = ', acc)
confusion_matrix(y_test,y_pred)
df_train['price_range'].value_counts()
# K-Means
kmeans = KMeans(n_clusters=4)
vec = kmeans.fit(df_numeric)
print(kmeans.cluster_centers_)  
print(kmeans.labels_)  