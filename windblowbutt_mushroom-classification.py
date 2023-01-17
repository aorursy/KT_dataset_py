# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

colums_names=["class", "cshape", "csurface", "ccolor", "bruises", "odor", "gattach", "gspace",
"gsize", "gcolor", "sshape", "sroot", "ssabove", "ssbelow", "scabove", "scbelow", "vtype",
"vcolor", "rnumber", "rtype", "spcolor", "popnum", "habitat"]
df=pd.read_csv('../input/mushroom.data', names=colums_names)
#throw missing data
df=df[df['sroot']!='?']
#get edible and pos boolean value
is_edible=df['class']=='e'
is_pos=df['class']=='p'
df
df.head()
df.info()
df.describe()
sns.countplot(x='class', data=df)
sns.countplot(y='cshape', data=df,hue='class')
X=df.loc[:,df.columns!='class']
y=df['class']
def barPlot(colNames):
    count=1
    figure=plt.figure(figsize=(19,9))
    temp=[0,1]
    for colName in colNames:
        plt.subplot(5,10,count)
        count+=1
        unique_list=df[colName].unique()
        e_num=[]
        p_num=[]
        width=0.4
        for item in unique_list:
            e_num.append((is_edible&(df[colName]==item)).sum())
            p_num.append((is_pos&(df[colName]==item)).sum())

        count+=1
        p1=plt.bar(unique_list,e_num,width,color='#00bfff')
        p2=plt.bar(unique_list,p_num,width,bottom=e_num,color='#ff4000')
        temp[0]=p1
        temp[1]=p2
        plt.ylabel('amount')
        plt.title('amount by '+colName)
    # plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.9,
    #                 wspace=0.35)
    plt.tight_layout()

    figure.legend((temp[0],temp[1]),['edible','posinious'],loc='upper left')
#     plt.show()

def pairPlot(fea1, fea2,is_edible,is_pos):
    plt.rcParams['figure.figsize'] = (9, 5)
    plt.style.use('ggplot')
    a_x_label=df[fea1].unique()
    a_x_ticks=np.arange(1,len(a_x_label)+1,step=1)
    a_y_label=df[fea2].unique()
    a_y_ticks=np.arange(1,len(a_y_label)+1,step=1)

    for i in range(len(a_x_label)):
        for j in range(len(a_y_label)):
            is_x_y=(df[fea1]==a_x_label[i])&(df[fea2]==a_y_label[j])
            e_num=(is_x_y&is_edible).sum()
            p_num=(is_x_y&is_pos).sum()
            offset_e = np.random.uniform(-0.3,0.3,(e_num,2))
            x_e=offset_e[:,0]+i+1
            y_e=offset_e[:,1]+j+1
            plt.scatter(x_e,y_e,s=10,color='green',alpha=0.3)
            offset_p = np.random.uniform(-0.3,0.3,(p_num,2))
            x_p=offset_p[:,0]+i+1
            y_p=offset_p[:,1]+j+1
            plt.scatter(x_p,y_p,s=10,color='red',alpha=0.3)
    plt.xticks(a_x_ticks, a_x_label)
    plt.yticks(a_y_ticks,a_y_label)
#     plt.show()

barPlot(colums_names)
pairPlot('odor','spcolor',is_edible,is_pos)
pairPlot('cshape','csurface',is_edible,is_pos)
pairPlot('cshape','scabove',is_edible,is_pos)
pairPlot('ssbelow','odor',is_edible,is_pos)
le=LabelEncoder()
scaler=StandardScaler()
for col in df.columns:
    df[col]=le.fit_transform(df[col])
X=df.iloc[:,1:23]
y=df.iloc[:,0]
X
# sns.set(style="dark", color_codes=True)
sns.set()
g = sns.pairplot(df, vars=["odor", "spcolor","habitat","rtype"],palette="husl", hue="class")

df.fillna(0)
corrmat = df.corr()
f, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(corrmat, vmax=.9, square=True);
plt.figure(figsize=(7,7))
sns.lmplot(x='cshape',y='spcolor', hue='class',data=df)
plt.figure(figsize=(7,7))
sns.regplot(x='csurface',y='odor',data=df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
from sklearn.metrics import accuracy_score
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
predict_knn=knn.predict(X_test)
acc_rate_of_knn=accuracy_score(y_test,predict_knn) 
print('The accuracy of using knn is {0}'.format(acc_rate_of_knn))

test_acc= []
train_acc=[]
k_list=np.arange(1,25)
for k in k_list:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    test_acc.append(knn.score(X_test, y_test))
    train_acc.append(knn.score(X_train, y_train))
plt.plot(k_list, test_acc, label = 'Test Accuracy')
plt.plot(k_list, train_acc, label = 'Train Accuracy')
plt.title('The K value VS Accuracy')
plt.xlabel('value of K')
plt.ylabel('Accuracy')
plt.xticks(k_list)

kmeans = KMeans(n_clusters=2)
predict_kmeans = kmeans.fit_predict(X)
acc_rate_of_kmeans=accuracy_score(y, predict_kmeans)
print('The accuracy of using k-means is {0}'.format(acc_rate_of_kmeans))

X_enc=pd.get_dummies(X)
scaler=StandardScaler()
X_std = scaler.fit_transform(X_enc)
le = LabelEncoder()
#p=1 e=0
y_enc = le.fit_transform(y.values.ravel())
#split dataset
X_train, X_test,y_train,y_test=train_test_split(X_std,y_enc,test_size=0.3,random_state=0)
clf=RandomForestClassifier(max_depth=23, random_state=0)
clf.fit(X_train, y_train)
predict_ran_forest=clf.predict(X_test)
acc_rate_of_kmeans=accuracy_score(y_test,predict_ran_forest)
print('The accuracy of using random forest is {0}'.format(acc_rate_of_kmeans))
importance=clf.feature_importances_
print(importance)
print('The most important feature that determine whether a mushroom is edible is ', colums_names[importance.argmax()+1])
from sklearn import svm

clf_svm=svm.SVC()
clf_svm.fit(X_train,y_train)
y_predict=clf_svm.predict(X_test)
aac_of_svm=accuracy_score(y_test,y_predict)
print('The accuracy of using SVM is {0}'.format(aac_of_svm))