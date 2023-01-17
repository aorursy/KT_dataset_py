%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("../input/mushrooms.csv", dtype="category")
columns = df.columns.values
#print(df.dtypes)
#df.head()
#for column in columns :
#    print(df[column].value_counts().sort_index())
# matplotlibで描画
for column in columns :
    plt.figure(figsize=(3,2))
    cat = df[column].value_counts().sort_index()
    plt.title(column)
    plt.bar(cat.index.codes, cat.values, tick_label=cat.index.tolist())
    plt.show()
# seabornで描画
for column in columns:
    ax = sns.factorplot(column, col="class", data=df, size=3.0, aspect=.9, kind="count")
for column in columns :
    df[df[column].isnull()]
# まず、数値に変換できる変数を先に変換しておく
df2 = df.copy()
df2['class'] = df['class'].map({'e':0, 'p':1})
df2['ring-number'] = df['ring-number'].map({'n':0, 'o':1, 't':2})
# カテゴリカル変数をすべてダミー変数に変更する
dummy_columns = ['cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']

for column in dummy_columns:
    df2[column+'_str'] = df[column].astype(str).map(lambda x:column+"_"+x)
    df2 = pd.concat([df2, pd.get_dummies(df2[column+'_str'])],axis=1)

drop_columns = dummy_columns.copy()
for column in dummy_columns:
    drop_columns.append(column+"_str")

df2_fin = df2.drop(drop_columns,axis=1)
#df2_fin.head(10)
# 訓練用データとテスト用データに分割する
train = df2_fin[:6000]
test = df2_fin[6000:-1]

y_train = train['class']
y_test = test['class']
x_train = train.drop('class', 1)
x_test = test.drop('class', 1)
# ロジスティック回帰モデルを使って学習する
lr = LogisticRegression()
lr.fit(x_train,y_train)
print(lr.coef_,lr.intercept_)

y_pred = lr.predict(x_test)
print(classification_report(y_test,y_pred))
# 有効な変数と推測した変数だけをPickupしてダミー変数化する
df3 = df.copy()
df3['class'] = df['class'].map({'e':0, 'p':1})
dummy_columns = ['cap-color','bruises','odor','gill-size','gill-color','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','ring-type','spore-print-color','population','habitat']

for column in dummy_columns:
    df3[column+'_str'] = df[column].astype(str).map(lambda x:column+"_"+x)
    df3 = pd.concat([df3, pd.get_dummies(df3[column+'_str'])], axis=1)

drop_columns = dummy_columns.copy()
drop_list = list(filter(lambda x:x not in dummy_columns, columns))
drop_list.remove('class')
drop_columns.extend(drop_list)

for column in dummy_columns:
    drop_columns.append(column+"_str")

df3_fin = df3.drop(drop_columns,axis=1)
#fdf3_fin.head(10)
# 訓練用データとテスト用データに分割する
train = df3_fin[:6000]
test = df3_fin[6000:-1]

y_train = train['class']
y_test = test['class']
x_train = train.drop('class', 1)
x_test = test.drop('class', 1)
# ロジスティック回帰モデルを使って学習する
lr = LogisticRegression()
lr.fit(x_train,y_train)
print(lr.coef_,lr.intercept_)

y_pred = lr.predict(x_test)
print(classification_report(y_test,y_pred))
