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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
def get_data(limit=None):
    df = pd.read_csv('jds101/fashion-mnist_train.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

if __name__ == '__main__':
    X, Y = get_data(60000)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)


cv_scores=[]
for k in range(1,9):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain, Ytrain)
    scores = knn.score(Xtest,Ytest)
    cv_scores.append([k, scores.mean(),scores.std()])
df_scores = pd.DataFrame(cv_scores, columns=['k', 'ACCURACY', 'STD'])
sns.lineplot(x="k", y="ACCURACY", data=df_scores)
new_test = pd.read_csv('jds101/new_test.csv')
new_test=new_test/255
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(Xtrain, Ytrain)
pred = knn.predict(new_test)
pred1 = pd.DataFrame(np.array(pred), columns=['label'],index=new_test.index)
pred1['id']=range(1,10001)
pred1
pred1.to_csv('pred1.csv', index=False)
names = ['DecisionTreeClassifier', 'LogisticRegression','GaussianNB', 'BernoulliNB', 'SVC', 'KNN']

clf_list = [DecisionTreeClassifier(max_depth=10, ccp_alpha = 0.005),
            LogisticRegression(C=0.1),
            GaussianNB(),
            BernoulliNB(alpha=0.01),
            SVC(kernel = 'poly', degree = 1, C = 10, gamma = 0.001),
            KNeighborsClassifier(n_neighbors=4)
           ]
for name, clf in zip(names, clf_list):
        t0 = datetime.now()
        clf.fit(Xtrain, Ytrain)
        print(name, end=': ')
        print(clf.score(Xtest, Ytest), "Time:", (datetime.now() - t0))
pred_logreg = clf_list[1].predict(new_test)

pred_logreg = pd.DataFrame(np.array(pred_logreg), columns=['label'])

pred_logreg['id']=range(1,10001)

pred_logreg.to_csv('pred.csv', index=False)
pred_svc = clf_list[4].predict(new_test)

pred_svc= pd.DataFrame(np.array(pred_svc), columns=['label'])

pred_svc['id']=range(1,10001)

pred_svc.to_csv('pred.csv', index=False)
pca = PCA()
X_pca = pd.DataFrame(pca.fit_transform(X))
(X_pca.var()/X_pca.var().sum()).cumsum().head(155)
X_pca = pca.transform(X)[:,:155]
X_pca.shape
Xtrain_pca, Xtest_pca, Ytrain_pca, Ytest_pca = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
svc = SVC(kernel = 'poly', degree = 1, C = 10, gamma = 0.001)
svc.fit(Xtrain_pca, Ytrain_pca)
print(svc.score(Xtest_pca, Ytest_pca))
new_test_pca = pca.transform(new_test)[:,:155]
pred_pca = svc.predict(new_test_pca)
pred1_pca = pd.DataFrame(np.array(pred_pca), columns=['label'])
pred1_pca['id']=range(1,10001)
pred1_pca.to_csv('pred1_pca.csv', index=False)
cv_scores=[]
t0 = datetime.now()
for k in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain_pca, Ytrain_pca)
    scores = knn.score(Xtest_pca,Ytest_pca)
    cv_scores.append([k, scores.mean(),scores.std()])
df_scores = pd.DataFrame(cv_scores, columns=['k', 'ACCURACY', 'STD'])
print("Time:", (datetime.now() - t0))
sns.lineplot(x="k", y="ACCURACY", data=df_scores)
df_scores
pred_pca_knn8 = knn.predict(new_test_pca)

pred_pca_knn8 = pd.DataFrame(np.array(pred_pca_knn8), columns=['label'])

pred_pca_knn8['id']=range(1,10001)

pred_pca_knn8.to_csv('pred_pca_knn8.csv', index=False)
for n in range(50,250,50):
    X_pca = pca.transform(X)[:,:n]
    Xtrain_pca, Xtest_pca, Ytrain_pca, Ytest_pca = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(Xtrain_pca, Ytrain_pca)
    print(n, "components" , end=': ')
    print(knn.score(Xtest_pca, Ytest_pca))
RF = RandomForestClassifier(criterion='entropy', max_depth=50, n_estimators = 300, random_state=0)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
RF.fit(Xtrain, Ytrain)
RF.score(Xtest,Ytest)
pred_RF = RF.predict(new_test)
pred_RF = pd.DataFrame(np.array(pred_RF), columns=['label'])

pred_RF['id']=range(1,10001)

pred_RF.to_csv('pred_RF.csv', index=False)
cat = CatBoostClassifier(iterations = 1500)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
cat.fit(Xtrain, Ytrain)
cat.score(Xtest,Ytest)
pred_cat = cat.predict(new_test)
pred_cat = pd.DataFrame(np.array(pred_cat), columns=['label'])

pred_cat['id']=range(1,10001)

pred_cat.to_csv('pred_cat.csv', index=False)
Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
cat.fit(Xtrain1, Ytrain1)
cat.score(Xtest1,Ytest1)
X_tsne = TSNE(random_state=42).fit_transform(X)
X_tsne.shape   
import matplotlib.patheffects as PathEffects
def scatter(x, colors):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(classes[i]), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
scatter(X_tsne, Y)
Xtrain_tsne, Xtest_tsne, Ytrain_tsne, Ytest_tsne = train_test_split(X_tsne, Y, test_size=0.2, random_state=42)
cat.fit(Xtrain_tsne, Ytrain_tsne)
cat.score(Xtest_tsne,Ytest_tsne)
X_pca_tsne = TSNE(random_state=42).fit_transform(X_pca)
X_pca_tsne.shape  
scatter(X_pca_tsne, Y)
X_pca2 = pca.transform(X)[:,:2]
X_pca2.shape
scatter(X_pca2, Y)
Xsample, __, Ysample, __ = train_test_split(Xtrain, Ytrain, test_size=0.9, random_state=42)
Xsample.shape
Xsample_train, Xsample_test, Ysample_train, Ysample_test = train_test_split(Xsample, Ysample, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(Xsample_train, Ysample_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(Xsample_train, Ysample_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))
train_scores = [clf.score(Xsample_train, Ysample_train) for clf in clfs]
test_scores = [clf.score(Xsample_test, Ysample_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
def get_data(limit=None):
    df = pd.read_csv('jds101/fashion-mnist_train.csv')
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0 # data is from 0..255
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

if __name__ == '__main__':
    X, Y = get_data(60000)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)


cv_scores=[]
for k in range(1,9):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain, Ytrain)
    scores = knn.score(Xtest,Ytest)
    cv_scores.append([k, scores.mean(),scores.std()])
df_scores = pd.DataFrame(cv_scores, columns=['k', 'ACCURACY', 'STD'])
sns.lineplot(x="k", y="ACCURACY", data=df_scores)
new_test = pd.read_csv('jds101/new_test.csv')
new_test=new_test/255
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(Xtrain, Ytrain)
pred = knn.predict(new_test)
pred1 = pd.DataFrame(np.array(pred), columns=['label'],index=new_test.index)
pred1['id']=range(1,10001)
pred1
pred1.to_csv('pred1.csv', index=False)
names = ['DecisionTreeClassifier', 'LogisticRegression','GaussianNB', 'BernoulliNB', 'SVC', 'KNN']

clf_list = [DecisionTreeClassifier(max_depth=10, ccp_alpha = 0.005),
            LogisticRegression(C=0.1),
            GaussianNB(),
            BernoulliNB(alpha=0.01),
            SVC(kernel = 'poly', degree = 1, C = 10, gamma = 0.001),
            KNeighborsClassifier(n_neighbors=4)
           ]
for name, clf in zip(names, clf_list):
        t0 = datetime.now()
        clf.fit(Xtrain, Ytrain)
        print(name, end=': ')
        print(clf.score(Xtest, Ytest), "Time:", (datetime.now() - t0))
pred_logreg = clf_list[1].predict(new_test)

pred_logreg = pd.DataFrame(np.array(pred_logreg), columns=['label'])

pred_logreg['id']=range(1,10001)

pred_logreg.to_csv('pred.csv', index=False)
pred_svc = clf_list[4].predict(new_test)

pred_svc= pd.DataFrame(np.array(pred_svc), columns=['label'])

pred_svc['id']=range(1,10001)

pred_svc.to_csv('pred.csv', index=False)
pca = PCA()
X_pca = pd.DataFrame(pca.fit_transform(X))
(X_pca.var()/X_pca.var().sum()).cumsum().head(155)
X_pca = pca.transform(X)[:,:155]
X_pca.shape
Xtrain_pca, Xtest_pca, Ytrain_pca, Ytest_pca = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
svc = SVC(kernel = 'poly', degree = 1, C = 10, gamma = 0.001)
svc.fit(Xtrain_pca, Ytrain_pca)
print(svc.score(Xtest_pca, Ytest_pca))
new_test_pca = pca.transform(new_test)[:,:155]
pred_pca = svc.predict(new_test_pca)
pred1_pca = pd.DataFrame(np.array(pred_pca), columns=['label'])
pred1_pca['id']=range(1,10001)
pred1_pca.to_csv('pred1_pca.csv', index=False)
cv_scores=[]
t0 = datetime.now()
for k in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtrain_pca, Ytrain_pca)
    scores = knn.score(Xtest_pca,Ytest_pca)
    cv_scores.append([k, scores.mean(),scores.std()])
df_scores = pd.DataFrame(cv_scores, columns=['k', 'ACCURACY', 'STD'])
print("Time:", (datetime.now() - t0))
sns.lineplot(x="k", y="ACCURACY", data=df_scores)
df_scores
pred_pca_knn8 = knn.predict(new_test_pca)

pred_pca_knn8 = pd.DataFrame(np.array(pred_pca_knn8), columns=['label'])

pred_pca_knn8['id']=range(1,10001)

pred_pca_knn8.to_csv('pred_pca_knn8.csv', index=False)
for n in range(50,250,50):
    X_pca = pca.transform(X)[:,:n]
    Xtrain_pca, Xtest_pca, Ytrain_pca, Ytest_pca = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(Xtrain_pca, Ytrain_pca)
    print(n, "components" , end=': ')
    print(knn.score(Xtest_pca, Ytest_pca))
RF = RandomForestClassifier(criterion='entropy', max_depth=50, n_estimators = 300, random_state=0)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
RF.fit(Xtrain, Ytrain)
RF.score(Xtest,Ytest)
pred_RF = RF.predict(new_test)
pred_RF = pd.DataFrame(np.array(pred_RF), columns=['label'])

pred_RF['id']=range(1,10001)

pred_RF.to_csv('pred_RF.csv', index=False)
cat = CatBoostClassifier(iterations = 1500)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
cat.fit(Xtrain, Ytrain)
cat.score(Xtest,Ytest)
pred_cat = cat.predict(new_test)
pred_cat = pd.DataFrame(np.array(pred_cat), columns=['label'])

pred_cat['id']=range(1,10001)

pred_cat.to_csv('pred_cat.csv', index=False)
Xtrain1, Xtest1, Ytrain1, Ytest1 = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
cat.fit(Xtrain1, Ytrain1)
cat.score(Xtest1,Ytest1)
X_tsne = TSNE(random_state=42).fit_transform(X)
X_tsne.shape   
import matplotlib.patheffects as PathEffects
def scatter(x, colors):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(classes[i]), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
scatter(X_tsne, Y)
Xtrain_tsne, Xtest_tsne, Ytrain_tsne, Ytest_tsne = train_test_split(X_tsne, Y, test_size=0.2, random_state=42)
cat.fit(Xtrain_tsne, Ytrain_tsne)
cat.score(Xtest_tsne,Ytest_tsne)
X_pca_tsne = TSNE(random_state=42).fit_transform(X_pca)
X_pca_tsne.shape  
scatter(X_pca_tsne, Y)
X_pca2 = pca.transform(X)[:,:2]
X_pca2.shape
scatter(X_pca2, Y)
Xsample, __, Ysample, __ = train_test_split(Xtrain, Ytrain, test_size=0.9, random_state=42)
Xsample.shape
Xsample_train, Xsample_test, Ysample_train, Ysample_test = train_test_split(Xsample, Ysample, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(Xsample_train, Ysample_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(Xsample_train, Ysample_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))
train_scores = [clf.score(Xsample_train, Ysample_train) for clf in clfs]
test_scores = [clf.score(Xsample_test, Ysample_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

