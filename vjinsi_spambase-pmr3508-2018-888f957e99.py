import pandas as pd
import sklearn
from sklearn.metrics import fbeta_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from matplotlib import gridspec
spam_train = pd.read_csv("../input/train_data.csv")
spam_test = pd.read_csv("../input/test_features.csv")
X_spam_train = spam_train.drop('ham',axis = 1)
Y_spam_train = spam_train.ham
X_spam_test = spam_test
X_spam_train_original = X_spam_train
Y_spam_train_original = Y_spam_train
gnb = GaussianNB()
Y_predict = gnb.fit(X_spam_train,Y_spam_train).predict(X_spam_test)
Y_predict
ham = 0
for i in Y_predict:
    if i == True:
        ham += 1
porcentagem = ham / len(Y_predict)
porcentagem
from sklearn.model_selection import cross_val_score
scores = cross_val_score(gnb,X_spam_train,Y_spam_train)
scores
y_predict_probabilities = gnb.predict_proba(X_spam_train)[:,1]
print(y_predict_probabilities)
print(len(y_predict_probabilities))
#####y_score = scores
Y_train_predict = gnb.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
print(Y_train_predict)
print(len(Y_train_predict))
#for i in range(len(Y_train_predict)):
#    print(Y_train_predict[i],Y_spam_train[i])

fbeta_score(Y_spam_train, Y_train_predict, average='macro', beta=3)
sklearn.metrics.roc_auc_score(Y_spam_train, y_predict_probabilities)
bnb = BernoulliNB()
scores_bernoulli = cross_val_score(bnb,X_spam_train,Y_spam_train)
print(scores_bernoulli)
sum(scores_bernoulli)/len(scores_bernoulli)
Y_train_predict_bernoulli = bnb.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
Y_train_predict_bernoulli_original = Y_train_predict_bernoulli
fbeta_score(Y_spam_train, Y_train_predict_bernoulli, average='macro', beta=3)
for i in spam_train:
    if i != 'ham' and i != 'Id':
        print(i, '           ', spam_train['ham'].corr(spam_train[str(i)]))
x = 4
corr = []
for i in spam_train:
    if i != 'ham' and i != 'Id':
        corr.append([i,spam_train['ham'].corr(spam_train[str(i)])])
menores_corr = []
for i in corr:
    if len(menores_corr) < x:
        menores_corr.append(i)
    else:
        for j in range(0,len(menores_corr)):
            if j == 0:
                j_min = j
            else:
                if max(abs(menores_corr[j][1]), abs(menores_corr[j_min][1])) == abs(menores_corr[j][1]):
                    j_min = j
        if abs(i[1]) < abs(menores_corr[j_min][1]):
            del(menores_corr[j_min])
            menores_corr.append(i)
print(menores_corr)        
for i in menores_corr:
    X_spam_train = X_spam_train.drop(columns = i[0])
scores_bernoulli = cross_val_score(bnb,X_spam_train,Y_spam_train)
print(scores_bernoulli)
print(sum(scores_bernoulli)/len(scores_bernoulli))
Y_train_predict_bernoulli = bnb.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
print(fbeta_score(Y_spam_train, Y_train_predict_bernoulli, average='macro', beta=3))
medias = []
fbetas = []
for x in range(1,56):
    corr = []
    X_spam_train = X_spam_train_original
    Y_train_predict_bernoulli = Y_train_predict_bernoulli_original
    Y_spam_train = Y_spam_train_original
    for i in spam_train:
        if i != 'ham' and i != 'Id':
            corr.append([i,spam_train['ham'].corr(spam_train[str(i)])])
    menores_corr = []
    for i in corr:
        if len(menores_corr) < x:
            menores_corr.append(i)
        else:
            for j in range(0,len(menores_corr)):
                if j == 0:
                    j_min = j
                else:
                    if max(abs(menores_corr[j][1]), abs(menores_corr[j_min][1])) == abs(menores_corr[j][1]):
                        j_min = j
            if abs(i[1]) < abs(menores_corr[j_min][1]):
                del(menores_corr[j_min])
                menores_corr.append(i)
    for i in menores_corr:
        X_spam_train = X_spam_train.drop(columns = i[0])
    scores_bernoulli = cross_val_score(bnb,X_spam_train,Y_spam_train)
    media = sum(scores_bernoulli)/len(scores_bernoulli)
    medias.append([x,media])
    Y_train_predict_bernoulli = bnb.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
    fbetas.append([x,fbeta_score(Y_spam_train, Y_train_predict_bernoulli, average='macro', beta=3)])
print(medias)
print()
print(fbetas)
maximo = 0
maximobeta = 0
for i in medias:
    maximo = max(maximo,i[1])
    if maximo == i[1]:
        x_maximo = i[0]
for j in fbetas:
    maximobeta = max(maximobeta,j[1])
    if maximobeta == j[1]:
        x_maximobeta = j[0]
print(x_maximo,maximo)
print(x_maximobeta,maximobeta)
x = 13
corr = []
X_spam_train = X_spam_train_original
Y_train_predict_bernoulli = Y_train_predict_bernoulli_original
Y_spam_train = Y_spam_train_original
for i in spam_train:
    if i != 'ham' and i != 'Id':
        corr.append([i,spam_train['ham'].corr(spam_train[str(i)])])
        menores_corr = []
for i in corr:
    if len(menores_corr) < x:
        menores_corr.append(i)
    else:
        for j in range(0,len(menores_corr)):
            if j == 0:
                j_min = j
            else:
                if max(abs(menores_corr[j][1]), abs(menores_corr[j_min][1])) == abs(menores_corr[j][1]):
                    j_min = j
        if abs(i[1]) < abs(menores_corr[j_min][1]):
            del(menores_corr[j_min])
            menores_corr.append(i)
for i in menores_corr:
    X_spam_train = X_spam_train.drop(columns = i[0])
    X_spam_test = X_spam_test.drop(columns = i[0])
Y_train_predict_bernoulli = bnb.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
print(fbeta_score(Y_spam_train, Y_train_predict_bernoulli, average='macro', beta=3))
print(X_spam_train.columns)
fig = plt.figure(figsize=(50,50))
spam_train_semId = spam_train.drop('Id',axis = 1)
bar = spam_train_semId.groupby('ham').mean().plot(kind='bar')
bar.legend(loc='upper center', bbox_to_anchor=(1.55, 3.80),fancybox=True, shadow=True)
spam_train.corr()
n = 10
pairs_to_drop = []
colunas = spam_train.columns
for i in range(0, spam_train.shape[1]):
    for j in range(0, i+1):
        pairs_to_drop.append([colunas[i], colunas[j]])
au_corr = spam_train.corr().abs().unstack()
au_corr = au_corr.drop(pairs_to_drop).sort_values(ascending=False)
print(au_corr[0:n])
fbeta_scores = []
spam_train = pd.read_csv("../input/train_data.csv")
corr_matrix = spam_train.corr()
corr_max = 1
X_spam_train_original = X_spam_train
while corr_max >= 0:

    corr_matrix = spam_train.corr()
    for i in range(len(corr_matrix.columns)-1):
        for j in range(i):
            if corr_matrix.iloc[i,j] > corr_max:
                if corr_matrix.columns[i] in spam_train.columns:
                    #del spam_train[corr_matrix.columns[i]]
                    spam_train = spam_train.drop(columns = [corr_matrix.columns[i]])
                if corr_matrix.columns[i] in X_spam_train.columns:
                    #del X_spam_train[corr_matrix.columns[i]]
                    X_spam_train = X_spam_train.drop(columns = [corr_matrix.columns[i]])
    if corr_max > 0.86:
        print(X_spam_train.shape)
    corr_max -= 0.05
    Y_train_predict_bernoulli = bnb.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
    fbeta_scores.append([corr_max,fbeta_score(Y_spam_train, Y_train_predict_bernoulli, average='macro', beta=3)])
print(fbeta_scores)
print(X_spam_train_original.shape)
spam_train = pd.read_csv("../input/train_data.csv")
X_spam_train = X_spam_train_original
corr_matrix = spam_train.corr()
corr_max = 0.9
for i in range(len(corr_matrix.columns)-1):
    for j in range(i):
        if corr_matrix.iloc[i,j] > corr_max:
            if corr_matrix.columns[i] in spam_train.columns:
                del spam_train[corr_matrix.columns[i]]
                del spam_test[corr_matrix.columns[i]]
            if corr_matrix.columns[i] in X_spam_train.columns:
                del X_spam_train[corr_matrix.columns[i]]
                del X_spam_test[corr_matrix.columns[i]]
Y_train_predict_bernoulli = bnb.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
fbeta_score(Y_spam_train, Y_train_predict_bernoulli, average='macro', beta=3)
spam_train.groupby('ham')[spam_train.columns.drop('Id')].mean()
X_spam_train_bin = X_spam_train.drop('Id',axis=1)
X_spam_test_bin = X_spam_test.drop('Id',axis=1)
agrupado = spam_train.groupby('ham')[spam_train.columns.drop('Id')].mean()
pd.options.mode.chained_assignment = None
for i in X_spam_train_bin:
    for j in range(len(X_spam_train_bin[i])):
        if abs(X_spam_train_bin[i][j] - agrupado[i][True]) > abs(X_spam_train_bin[i][j] - agrupado[i][False]):
            X_spam_train_bin[i][j] = 1
            #X_spam_test_bin[i][j] = 1
        else:
            X_spam_train_bin[i][j] = 0
            #X_spam_test_bin[i][j] = 0
Y_train_predict_bernoulli = bnb.fit(X_spam_train_bin,Y_spam_train).predict(X_spam_train_bin)
fbeta_score(Y_spam_train, Y_train_predict_bernoulli, average='macro', beta=3)
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5)
classifier = BernoulliNB()

thresholds = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X_spam_train, Y_spam_train):
    probas_ = classifier.fit(X_spam_train.iloc[train], Y_spam_train.iloc[train]).predict_proba(X_spam_train.iloc[test])
    fpr, tpr, thr = roc_curve(Y_spam_train.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    thresholds.append(interp(mean_fpr,fpr,thr))
    thresholds[-1][0] = 1.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Aleatório', alpha=.8)

mean_thresholds = np.mean(thresholds, axis=0)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC BernoulliNB')
plt.legend(loc="lower right")
plt.show()
pont = 0
for i in range(len(mean_fpr)):
    pont_p = 10*mean_tpr[i]/(10*mean_tpr[i]+9*(1-mean_tpr[i])+mean_fpr[i])
    if pont_p > pont:
        pont = pont_p
        maxindex = i

print('O threshold ideal de probabilidade é:',round(mean_thresholds[maxindex], 4))
print('Para este treshold a taxa de falsos positivos esperada é:',round(mean_fpr[maxindex], 4))
print('Para este treshhold a taxa de verdadeiros positivos esperada é:',round(mean_tpr[maxindex], 4))
bnb = BernoulliNB()
bnb.fit(X_spam_train_bin,Y_spam_train)
test_data = X_spam_test_bin
testPred = bnb.predict(test_data)
file = open('BernoulliNB_prediction.csv','w')
file.write("Id,ham\n")
for i, j in zip(test_data.index,testPred):
    file.write(str(i)+','+str((j))+'\n')
file.close()
import pandas as pd
import sklearn
from sklearn.metrics import fbeta_score
spam_train = pd.read_csv("../input/train_data.csv")
spam_train.shape
spam_train
spam_test = pd.read_csv("../input/test_features.csv")
spam_test.shape
X_spam_train = spam_train.drop('ham',axis = 1)
Y_spam_train = spam_train.ham
X_spam_test = spam_test
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

for i in range (1,10):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, X_spam_train, Y_spam_train, cv=10)
    score_m = sum(scores)/float(len(scores))
    if i == 1:
        score_max = score_m
        i_max = i
    else:
        score_max = max(score_max,score_m)
    if score_max == score_m:
        i_max = i
print(score_max)
print(i_max)    
knn = KNeighborsClassifier(n_neighbors = i_max)

knn.fit(X_spam_train,Y_spam_train)

Y_spam_test_pred = knn.predict(X_spam_test)
Y_spam_test_pred
j_count = 0
for j in Y_spam_test_pred:
    if j == True:
        j_count += 1
j_count / float(len(Y_spam_test_pred))
for i in range (2,10):
    knn = KNeighborsClassifier(n_neighbors = i)
    Y_train_predict = knn.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
    fbeta = fbeta_score(Y_spam_train, Y_train_predict, average='macro', beta=3)
    if i == 1:
        score_max = fbeta
        i_max = i
    else:
        score_max = max(score_max,fbeta)
    if score_max == fbeta:
        i_max = i
print(score_max)
print(i_max)   
knn = KNeighborsClassifier(n_neighbors = 2)
Y_train_predict = knn.fit(X_spam_train,Y_spam_train).predict(X_spam_train)
fbeta_score(Y_spam_train, Y_train_predict, average='macro', beta=3)