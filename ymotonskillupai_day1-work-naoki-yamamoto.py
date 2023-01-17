%matplotlib inline 
#グラフをnotebook内に描画させるための設定
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
df_data = pd.read_csv("../input/mushrooms.csv")
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
# coutn missing
pd.DataFrame(df_data.isnull().sum(), columns=["num of missing"])
df_data['tmp_class'] = df_data['class'].astype(str).map(lambda x:'class-'+x)
df_data['tmp_cap-shape'] = df_data['cap-shape'].astype(str).map(lambda x:'cap-shape-'+x)
df_data['tmp_cap-surface'] = df_data['cap-surface'].astype(str).map(lambda x:'cap-surface-'+x)
df_data['tmp_cap-color'] = df_data['cap-color'].astype(str).map(lambda x:'cap-color-'+x)
df_data['tmp_bruises'] = df_data['bruises'].astype(str).map(lambda x:'bruises-'+x)
df_data['tmp_odor'] = df_data['odor'].astype(str).map(lambda x:'odor-'+x)
df_data['tmp_gill-attachment'] = df_data['gill-attachment'].astype(str).map(lambda x:'gill-attachment-'+x)
df_data['tmp_gill-spacing'] = df_data['gill-spacing'].astype(str).map(lambda x:'gill-spacing-'+x)
df_data['tmp_gill-size'] = df_data['gill-size'].astype(str).map(lambda x:'gill-size-'+x)
df_data['tmp_gill-color'] = df_data['gill-color'].astype(str).map(lambda x:'gill-color-'+x)
df_data['tmp_stalk-shape'] = df_data['stalk-shape'].astype(str).map(lambda x:'stalk-shape-'+x)
df_data['tmp_stalk-root'] = df_data['stalk-root'].astype(str).map(lambda x:'stalk-root-'+x)
df_data['tmp_stalk-surface-above-ring'] = df_data['stalk-surface-above-ring'].astype(str).map(lambda x:'stalk-surface-above-ring-'+x)
df_data['tmp_stalk-surface-below-ring'] = df_data['stalk-surface-below-ring'].astype(str).map(lambda x:'stalk-surface-below-ring-'+x)
df_data['tmp_stalk-color-above-ring'] = df_data['stalk-color-above-ring'].astype(str).map(lambda x:'stalk-color-above-ring-'+x)
df_data['tmp_stalk-color-below-ring'] = df_data['stalk-color-below-ring'].astype(str).map(lambda x:'stalk-color-below-ring-'+x)
df_data['tmp_veil-type'] = df_data['veil-type'].astype(str).map(lambda x:'veil-type-'+x)
df_data['tmp_veil-color'] = df_data['veil-color'].astype(str).map(lambda x:'veil-color-'+x)
df_data['tmp_ring-number'] = df_data['ring-number'].astype(str).map(lambda x:'ring-number-'+x)
df_data['tmp_ring-type'] = df_data['ring-type'].astype(str).map(lambda x:'ring-type-'+x)
df_data['tmp_spore-print-color'] = df_data['spore-print-color'].astype(str).map(lambda x:'spore-print-color-'+x)
df_data['tmp_population'] = df_data['population'].astype(str).map(lambda x:'population-'+x)
df_data['tmp_habitat'] = df_data['habitat'].astype(str).map(lambda x:'habitat-'+x)
print(df_data.columns)
display(df_data.head())
display(df_data.tail())
df_tmp_data = df_data.drop(['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat'],axis=1)
print(df_tmp_data.columns)
display(df_tmp_data.head())
display(df_tmp_data.tail())
# coutn missing
pd.DataFrame(df_tmp_data.isnull().sum(), columns=["num of missing"])
df_en = pd.concat([df_tmp_data,pd.get_dummies(df_tmp_data['tmp_class'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_cap-shape'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_cap-surface'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_cap-color'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_bruises'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_odor'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_gill-attachment'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_gill-spacing'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_gill-size'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_gill-color'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_stalk-shape'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_stalk-root'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_stalk-surface-above-ring'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_stalk-surface-below-ring'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_stalk-color-above-ring'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_stalk-color-below-ring'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_veil-type'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_veil-color'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_ring-number'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_ring-type'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_spore-print-color'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_population'])],axis=1)
df_en = pd.concat([df_en,pd.get_dummies(df_tmp_data['tmp_habitat'])],axis=1)
df_fin = df_en.drop(['tmp_class', 'tmp_cap-shape', 'tmp_cap-surface', 'tmp_cap-color',
       'tmp_bruises', 'tmp_odor', 'tmp_gill-attachment', 'tmp_gill-spacing',
       'tmp_gill-size', 'tmp_gill-color', 'tmp_stalk-shape', 'tmp_stalk-root',
       'tmp_stalk-surface-above-ring', 'tmp_stalk-surface-below-ring',
       'tmp_stalk-color-above-ring', 'tmp_stalk-color-below-ring',
       'tmp_veil-type', 'tmp_veil-color', 'tmp_ring-number', 'tmp_ring-type',
       'tmp_spore-print-color', 'tmp_population', 'tmp_habitat'],axis=1)
print(df_fin.columns)
display(df_fin.head())
display(df_fin.tail())
# coutn missing
pd.DataFrame(df_fin.isnull().sum(), columns=["num of missing"])
X = df_fin.drop(['class-e','class-p'],axis=1)
y = df_fin['class-e']
print(X.columns)
display(X.head())
display(X.tail())
import itertools #組み合わせを求めるときに使う
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=43)

lr = LogisticRegression()
lr.fit(X_train,y_train)
print(lr.coef_)
len(lr.coef_[0])
from matplotlib.ticker import ScalarFormatter
ax = pd.Series(lr.coef_[0],index=X_train.columns).sort_values().plot(kind='barh',figsize=(6,20))
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci",  axis="x",scilimits=(0,0))
y_pred = lr.predict(X_test)
ACCU = accuracy_score(y_test, y_pred)
RECL = recall_score(y_test, y_pred)
PREC = precision_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)
print("Accuracy=%s"%round(ACCU,4) )
print("Recall=%s"%round(RECL, 3) )
print("Precision=%s"%round(PREC,3) )
print("F1=%s"%round(F1,3) )
