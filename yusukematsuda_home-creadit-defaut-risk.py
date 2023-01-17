# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.
%%javascript
IPython.OutputArea.auto_scroll_threshold = 9999;
pd.options.display.max_columns = None
application_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
application_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
print('Training data shape: ', application_train.shape)
application_train.head()
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
previous_application = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
POS_CASH_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
installments_payments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
cols_with_missing = [col for col in application_train.columns
                     if application_train[col].isnull().any()]
application_train[cols_with_missing].head(10)
# Drop columns in training and validation data
reduced_X_train = application_train.drop(cols_with_missing, axis=1)
reduced_X_valid = application_test.drop(cols_with_missing, axis=1)
reduced_X_train.head(10)
#オブジェクト型になっているやつらが，categorical_value保持
s = (reduced_X_train.dtypes == 'object')
object_cols = list(s[s].index)
object_cols
reduced_X_train = reduced_X_train.drop(object_cols, axis=1)
reduced_X_valid = reduced_X_valid.drop(object_cols, axis=1)
reduced_X_train['loan_vs_income'] = reduced_X_train['AMT_CREDIT']/reduced_X_train['AMT_INCOME_TOTAL']
reduced_X_valid['loan_vs_income'] = reduced_X_valid['AMT_CREDIT']/reduced_X_valid['AMT_INCOME_TOTAL']
# #Categorical Valueへの処置は後ほど考える
# print("Categorical variables:")
# drop_X_train[object_cols].head(10)
# from sklearn.preprocessing import LabelEncoder

# # Make copy to avoid changing original data 
# label_X_train = application_train.copy()
# label_X_valid = application_test.copy()

# # Label encodeingによってcategorical variablesを数値データに変換する fit_transform&transform
# # Apply label encoder to each column with categorical data
# label_encoder = LabelEncoder()
# for col in object_cols:
#     label_X_train[col] = label_encoder.fit_transform(application_train[col])
#     label_X_valid[col] = label_encoder.transform(application_test[col])
X_train_target0 = reduced_X_train[reduced_X_train['TARGET']==1]
X_train_target1 = reduced_X_train[reduced_X_train['TARGET']==0]
#債務不履行者の集団の統計情報
X_train_target1_desc = reduced_X_train[reduced_X_train['TARGET']==1].describe()
X_train_target1_desc

#債務履行者の集団の統計情報
X_train_target0_desc =  reduced_X_train[reduced_X_train['TARGET']==0].describe()
X_train_target0_desc
print(len(X_train_target0))
print(len(X_train_target1))
X_train_target0['AMT_INCOME_TOTAL']/len(X_train_target0)
# グラフ描画のためmatplotlibのpyplotをインポート
#output_notebook()
fig = plt.figure(figsize=(15, 10), dpi=50,facecolor='w')
plt.rcParams["font.size"] = 18
ax = plt.gca()
ax.set_yscale('log')  # メイン: y軸をlogスケールで描く
ax.set_xscale('log')  # メイン: y軸をlogスケールで描く

ax.set_xlabel('AMT_INCOME_TOTAL',fontsize=18)
ax.set_ylabel('count',fontsize=18)
ax.grid()
ax.hist(X_train_target0['AMT_INCOME_TOTAL'],alpha=0.5, color='blue',bins=50, density=True)

ax.hist(X_train_target1['AMT_INCOME_TOTAL'],alpha=0.5, color='red',bins=50, density=True)
plt.show()
fig = plt.figure(figsize=(10, 10), dpi=50)
plt.rcParams["font.size"] = 18
plt.grid(True)

plt.xlabel("AMT_INCOME_TOTAL")
plt.ylabel("AMT_INCOME_TOTAL")

plt.bar(['X0 mean', 'X1 mean'] ,[X_train_target0_desc['AMT_INCOME_TOTAL']['mean'],X_train_target1_desc['AMT_INCOME_TOTAL']['mean']], 
        yerr = [X_train_target0_desc['AMT_INCOME_TOTAL']['std'],X_train_target1_desc['AMT_INCOME_TOTAL']['std']], color = ['blue', 'red'], alpha = 0.6)
plt.show()
for column in X_train_target0.columns:
    print(column)
def plot_hist_mean(X_target0,X_target1,X_target0_desc,X_target1_desc):
    #カラム毎にヒストグラムと平均・分散棒グラフを描画する
    for column in X_target0.columns:
        plot_hist_mean_every_column(X_target0,X_target1,X_target0_desc,X_target1_desc,column)
def plot_hist_mean_every_column(X_target0,X_target1,X_target0_desc,X_target1_desc,column):

    #ヒストグラムの描画
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(25,5),facecolor='w')
    
    plt.rcParams["font.size"] = 18

    #fig = plt.figure(figsize=(15, 10), dpi=50)
    axL.set_yscale('log')  # メイン: y軸をlogスケールで描く
    axL.set_xlabel(column, fontsize=18)
    axL.set_ylabel('count',fontsize=18)

    axL.grid(True)
    axL.hist(X_target0[column],alpha=0.5, color='blue',bins=50, density=True)
    axL.hist(X_target1[column],alpha=0.5, color='red',bins=50, density=True)

    #平均・分散（エラーバー）の描画
    axR.grid(True)

    axR.set_ylabel("AMT_INCOME_TOTAL")

    #横軸の名前
    x0 = 'X_target0'
    x1 = 'X_target1'

    axR.bar([x0, x1] ,[X_target0_desc[column]['mean'],X_target1_desc[column]['mean']], 
            yerr = [X_target0_desc[column]['std'],X_target1_desc[column]['std']], color = ['blue', 'red'], alpha = 0.6)
    fig.show()

plot_hist_mean(X_train_target0,X_train_target1,X_train_target0_desc,X_train_target1_desc)
bureau.describe()
bureau_balance.describe()
previous_application.describe()
POS_CASH_balance.describe()
installments_payments.describe()
credit_card_balance.describe()
