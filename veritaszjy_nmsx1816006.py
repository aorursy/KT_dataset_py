import pandas as pd
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from itertools import combinations

train = pd.read_csv('../input/train.csv' )
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
%matplotlib inline
plt.figure()
print(sns.palettes.__all__)
sns.countplot(x='Cover_Type', data=train)
plt.savefig('Cover_type_count.png')
%matplotlib inline
plt.figure(figsize=(12,10))
ax = sns.heatmap(train.corr(), fmt=".2f", cmap='Blues')
ax.set_title("Correlation")

print([ _ for _ in train.columns if not train[_].sum() ])
# convert one-hot encoding to label encoding 
# not needed for tree ensemble
def compress(df):
#   soil_out = [ _ for _ in train.columns if not train[_].sum() ]
    
    soil_list = [ "Soil_Type" + str(i) for i in range(1,41) ]
    wild_list = [ "Wilderness_Area" + str(i) for i in range(1,5)]
    
    df[soil_list] = df[soil_list].multiply([ i for i in range(1,41)], axis=1)
    df['Soil_Type'] = df[soil_list].sum(axis=1)
    
    df[wild_list] = df[wild_list].multiply([ i for i in range(1,5)], axis=1)
    df['Wild_Type'] = df[wild_list].sum(axis=1)
    
    df=df.drop(soil_list+wild_list, axis=1)
    return df

train = compress(train)
test = compress(test)
kde=sns.kdeplot(train['Cover_Type'], train['Aspect'], shade=True, cmap='Blues')
sns.set_style('dark')
plt.figure(figsize=(10,10))
plt.subplot(221)
sns.violinplot(x='Cover_Type', y='Soil_Type', data=train, inner=None)
plt.subplot(222)
sns.violinplot(x='Cover_Type', y='Wild_Type', data=train, inner=None)
plt.subplot(223)
sns.violinplot(x='Cover_Type', y='Aspect', data=train, inner=None)
plt.subplot(224)
sns.violinplot(x='Cover_Type', y='Elevation', data=train, inner=None)
g = sns.FacetGrid(train,col="Cover_Type",hue="Cover_Type")
g.map(plt.scatter,'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',alpha=.7)
g.set_axis_labels('Vert Dist to Hydrology', 'Heri Dist to Hydrology')
g.add_legend()
HDlist=train.filter(regex="Horizontal_Dis*").columns
sns.pairplot(data=train,x_vars=HDlist, y_vars=HDlist, hue='Cover_Type', palette=sns.color_palette(), diag_kind='kde')
plt.figure(figsize=(len(HDlist)*5,5))
for i in range(len(HDlist)):
    plt.subplot(1,len(HDlist),i+1)
    sns.violinplot(x='Cover_Type', y=HDlist[i], data=train, inner=None)
Shadelist=train.filter(regex="Hillshade*").columns
#   Hillshade's domain is [0, 360]
sns.pairplot(data=train,x_vars=Shadelist, y_vars=Shadelist, hue='Cover_Type', palette=sns.color_palette(), diag_kind='kde')
f = plt.figure(figsize=(len(Shadelist)*5,5))
for i in range(len(Shadelist)):
    f.add_subplot(1,len(Shadelist),i+1)
    sns.violinplot(x='Cover_Type', y=Shadelist[i], data=train, inner=None)
total=pd.concat([train[ test.columns ], test])
# total[Shadelist]=np.cos(total[Shadelist]*np.pi/360) # 360 to avoid multiplicity
# total['Aspect']=np.cos(total['Aspect']*np.pi/360)
total['ED_to_H'] = (total['Horizontal_Distance_To_Hydrology']**2 + total['Vertical_Distance_To_Hydrology']**2)**0.5

for col1, col2 in combinations(HDlist, 2):
    total[col1 + '_ADD_' + col2] = total[col1] + total[col2]
    total[col1 + '_DIF_' + col2] = total[col1] - total[col2]
    total[col1 + '_TIMES_' + col2] = total[col1] * total[col2]
    total[col1 + '_DIV_' + col2] = (total[col1] + 1e-3) / (total[col2] + 1e-3)
for col1, col2, col3 in combinations(HDlist, 3):
    total['meanHD']=(total[col1]+total[col2]+total[col3])/3
for col1, col2, col3 in combinations(Shadelist, 3):
    total['meanS']=(total[col1]+total[col2]+total[col3])/3
X_train = total.iloc[:15120,:].drop('Id', axis=1)
X_test = total.iloc[15120:,:].drop('Id', axis=1)
y = train['Cover_Type']
kf = StratifiedKFold(n_splits=5, random_state=100, shuffle=True )
et = ExtraTreesClassifier(n_estimators=200)
score = []
et.fit(X_train, y)
for tr_i, val_i in kf.split(X_train, y):
    X_tr = X_train.iloc[tr_i, :]
    X_val = X_train.iloc[val_i, :]
    y_tr = y[tr_i]
    y_val = y[val_i]
    et.fit(X_tr, y_tr)
    y_val_pred = et.predict(X_val)
    s = accuracy_score(y_val, y_val_pred)
    score.append(s)
print(score)
res = pd.DataFrame({"Id": test['Id'],"Cover_Type": et.predict(X_test)}, columns=['Id', 'Cover_Type'])
res.to_csv("result_et.csv", index=False) 