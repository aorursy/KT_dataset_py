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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import linear_model

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics.cluster import contingency_matrix

from statsmodels.stats.contingency_tables import mcnemar
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



print("Les données d'apprentissage train_data ont pour dimension",train_data.shape)

print("Les données de test test_data ont pour dimension",test_data.shape)

print("Les données d'exemple de soummision gender_submission ont pour dimension",gender_submission_data.shape)
train_data.info()
train_data.head()
test_data.head()
gender_submission_data.head()
train_data.head()
train_data.Survived.value_counts()
# Visualiation des variables catégorielles et quantitatives 



# variables quantitatives

var_num = list(train_data.columns[(train_data.dtypes=="int")|(train_data.dtypes=="float")])



print("\n On a {} variables numeriques : \n {} ".format(len(var_num) ,var_num))



# Variables categorielles

var_cat = list(train_data.columns[train_data.dtypes=="object"]) #possible de rajouter ceci si on veut exclure par exemple la var catégorielle y : .drop("y")) 

print("\n On a {} variables categorielles : \n {} ".format(len(var_cat),var_cat))
train_data.info()
# Examen d'une des variables catégorielles

print("\n Catégories d'une des variables {}".format(train_data["Sex"]. unique()))

plt.figure(figsize = (15,4))

sb.countplot(x = "Sex", data = train_data)



# Analyse statistique selon la variable catégorielle

print("\n Analyse statistique selon la variable Sex") 

print("Moyenne")

print(train_data.groupby("Sex").mean())

print("Variance")

print(train_data.groupby("Sex").var())
# Voir le nombre de personnes par classe

train_data.Sex.value_counts()
# Code Tutoriel Kaggle 

# ? : pourquoi la somme des pourcentage n'est pas égale à 100 ?



# Proportion de femmes qui ont survécu

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = round(sum(women)/len(women)*100,2)

print("Le pourcentage de femmes qui ont survécu est :", rate_women)



# Proportion d'hommes qui ont survécu

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = round(sum(men)/len(men)*100,2)

print("\nLe pourcentage d'hommes qui ont survécu est :", rate_men)
# Examen d'une des variables quantitatives

print("\n Catégories d'une des variables {}".format(train_data["Survived"]. unique()))

plt.figure(figsize = (15,4))

sb.countplot(x = "Survived", data = train_data)



# Analyse statistique selon la variable quantitative

print("\n Analyse statistique selon la variable Survived") 

print("Moyenne")

print(train_data.groupby("Survived").mean())

print("Variance")

print(train_data.groupby("Survived").var())
train_data.Survived.value_counts()
train_data.Pclass.value_counts()
# Proportion de personnes par classe sociale



Pclass_1 = train_data.loc[train_data.Pclass == 1]

rate_Pclass_1 = round((Pclass_1.shape[0]/train_data.Pclass.shape[0])*100,2)

print("Le pourcentage de personnes appartenant à la classe 1  :", rate_Pclass_1)



Pclass_2 = train_data.loc[train_data.Pclass == 2]

rate_Pclass_2 = round((Pclass_2.shape[0]/train_data.Pclass.shape[0])*100,2)

print("Le pourcentage de personnes appartenant à la classe 2  :", rate_Pclass_2)



Pclass_3 = train_data.loc[train_data.Pclass == 3]

rate_Pclass_3 = round((Pclass_3.shape[0]/train_data.Pclass.shape[0])*100,2)

print("Le pourcentage de personnes appartenant à la classe 3  :", rate_Pclass_3)
# Proportion de personnes (survivantes et non-survivantes) par classe sociale



Pclass_1 = train_data.loc[train_data.Pclass == 1]["Survived"]

rate_Pclass_1 = round((Pclass_1.shape[0]/train_data.Pclass.shape[0])*100,2)

print("Le pourcentage de personnes (survivantes et non-survivantes) qui appartienennt à classe sociale 1 est :", rate_Pclass_1)



Pclass_2 = train_data.loc[train_data.Pclass == 2]["Survived"]

rate_Pclass_2 = round((Pclass_2.shape[0]/train_data.Pclass.shape[0])*100,2)

print("Le pourcentage de personnes (survivantes et non-survivantes) et qui appartienennt à classe sociale 2 est :", rate_Pclass_2)



Pclass_3 = train_data.loc[train_data.Pclass == 3]["Survived"]

rate_Pclass_3 = round((Pclass_3.shape[0]/train_data.Pclass.shape[0])*100,2)

print("Le pourcentage de personnes (survivantes et non-survivantes) et qui appartienennt à classe sociale 3 est :", rate_Pclass_3)
# Examen d'une des variables quantitatives

print("\n Catégories d'une des variables {}".format(train_data["Pclass"]. unique()))

plt.figure(figsize = (15,4))

sb.countplot(x = "Pclass", data = train_data)



# Analyse statistique selon la variable quantitative

print("\n Analyse statistique selon la variable Pclass") 

print("Moyenne")

print(train_data.groupby("Pclass").mean())

print("Variance")

print(train_data.groupby("Pclass").var())
# Examen d'une des variables catégorielles

print("\n Catégories d'une des variables {}".format(train_data["Embarked"]. unique()))

plt.figure(figsize = (15,4))

sb.countplot(x = "Embarked", data = train_data)



# Analyse statistique selon la variable catégorielle

print("\n Analyse statistique selon la variable Embarked") 

print("Moyenne")

print(train_data.groupby("Embarked").mean())

print("Variance")

print(train_data.groupby("Embarked").var())
# Code tutoriel Kaggle: Random Forest



# from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]



X = pd.get_dummies(train_data[features])



X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1) # cf RandomForestClassifier dans sklearn



# n_estimators : nombre d'arbres dans la foret



# max_depth : profondeur maximale des arbres



# random_state : Contrôle à la fois le caractère aléatoire du bootstrap des échantillons utilisés 

# lors de la construction d'arbres (si bootstrap=True) et l'échantillonnage des entités à prendre en compte 

# lors de la recherche de la meilleure répartition sur chaque nœud (si )



model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_1_RANDOMFOREST.csv', index=False)

print("Your submission was successfully saved!")
print(output.shape)
output.head(10)
# Découpage du jeu de données en apprentissage et validation

Xa, Xv, Ya, Yv = train_test_split(X, y, shuffle=True, test_size=0.3, stratify=y)
print(X.shape)

print(y.shape)

print(Xa.shape)

print(Xv.shape)

print(Ya.shape)

print(Yv.shape)
# LDA sur données d'apprentissage et validation et calculs d'erreur de classification



# LDA

clf_lda = LinearDiscriminantAnalysis(solver='svd', store_covariance = False)

clf_lda.fit(Xa, Ya)

Y_lda = clf_lda.predict(Xa)

err_lda = sum(Y_lda != Ya)/Ya.size

print('LDA : taux d''erreur apprentissage = {}%'.format(100*err_lda))

Y_ldat = clf_lda.predict(Xv)

err_ldat = sum(Y_ldat != Yv)/Yv.size

print('LDA : taux d''erreur validation= {}%'.format(100*err_ldat))



print('\n \n')

# QDA

clf_qda = QuadraticDiscriminantAnalysis(store_covariance = False)

clf_qda.fit(Xa, Ya)

Y_qda = clf_qda.predict(Xa)

err_qda = sum(Y_qda!= Ya)/Ya.size

print('QDA : taux d''erreur apprentissage = {}%'.format(100*err_qda))

Y_qdat = clf_qda.predict(Xv)

err_qdat = sum(Y_qdat!= Yv)/Yv.size

print('QDA : taux d''erreur validation = {}%'.format(100*err_qdat))
# Recherche de la valeur optimale du paramètre de régularisaton C avec GridSearchCV



C_grid = np.logspace(-1, 1, 50)



#C_grid = [i/100 for i in range (1, 150)]



#from sklearn.model_selection import GridSearchCV



# the grid

parameters = [{"C": C_grid}]



# the classifier

clf_reg = linear_model.LogisticRegression(tol=1e-5, multi_class='multinomial', solver='lbfgs') # solveur par defaut



#NB : mutlinomial marche même face à un cas de classification binaire, minimise perte, cf sklearn pour plus d'infos



# Perf a K-fold validation using the accuracy as the performance measure

K = 10 # feel free to adapt the value of $K$



# we will do it on a grid search using n_jobs processors

clf_reg = GridSearchCV(clf_reg, param_grid=parameters, cv=K, scoring="accuracy", verbose=1, n_jobs = 2)

clf_reg.fit(Xa, Ya)
# Calcul des erreurs de classification 



# Get the best parameters

print("\nRegression logistique - optimal hyper-parameters = {}".format(clf_reg.best_params_))

print("Regression logistique - best cross-val accuracy = {} \n".format(clf_reg.best_score_))



y_app_reg = clf_reg.predict(Xa)

err_app_reg = sum(y_app_reg != Ya)/Ya.size

print('Regression logistique : taux d''erreur apprentissage = {}%'.format(100*err_app_reg))



y_val_reg = clf_reg.predict(Xv)

err_val_reg = sum(y_val_reg != Yv)/Yv.size

print('Regression logistique : taux d''erreur validation = {}%'.format(100*err_val_reg))
# Recherche de la valeur optimale du paramètre de régularisation C avec GridSearchCV



C_grid = np.logspace(-1, 1, 7)

#C_grid = [i/100 for i in range (1, 150)]



#from sklearn.model_selection import GridSearchCV



# the grid

parameters = [{"C": C_grid}]



# the classifier

clf_svm_lin = SVC(kernel='linear', probability=True)



# Perf a K-fold validation using the accuracy as the performance measure

K = 5 # feel free to adapt the value of $K$



# we will dot it on a grid search using n_jobs processors

clf_svm_lin = GridSearchCV(clf_svm_lin, param_grid=parameters, cv=K, scoring="accuracy", verbose=1, n_jobs = 2)

clf_svm_lin.fit(Xa, Ya)
# Calcul des erreurs de classification 



# Get the best parameters

print("\nSVM linéaire - optimal hyper-parameters = {}".format(clf_svm_lin.best_params_))

print("SVM linéaire - best cross-val accuracy = {} \n".format(clf_svm_lin.best_score_))



y_app_svm_lin = clf_reg.predict(Xa)

err_app_svm_lin = sum(y_app_svm_lin != Ya)/Ya.size

print('SVM linéaire : taux d''erreur apprentissage = {}%'.format(100*err_app_svm_lin))



y_val_svm_lin = clf_svm_lin.predict(Xv)

err_val_svm_lin = sum(y_val_svm_lin != Yv)/Yv.size

print('SVM linéaire : taux d''erreur validation = {}%'.format(100*err_val_svm_lin))
# Recherche des valeurs optimales des paramètres C et gamma avec GridSearchCV



gamma_grid = np.logspace(-1.5, 0, 3)

C_grid = np.logspace(-1, 1.5, 3)



# the grid

parameters = [{"gamma": gamma_grid, "C": C_grid}]



# the classifier

clf_svm_rbf = SVC(kernel="rbf", tol=0.01, cache_size = 1000, probability=True)



# Perf a K-fold validation using the accuracy as the performance measure

K = 3 # feel free to adapt the value of $K$



# we will do it on a grid search using n_jobs processors

clf_svm_rbf = GridSearchCV(clf_svm_rbf, param_grid=parameters, cv=K, scoring="accuracy", verbose=1, n_jobs = 2)

clf_svm_rbf.fit(Xa, Ya)
# Calcul des erreurs de classification



# Get the best parameters

print("\nSVM non linéaire - optimal hyper-parameters = {}".format(clf_svm_rbf.best_params_))

print("SVM non linéaire - best cross-val accuracy = {} \n".format(clf_svm_rbf.best_score_))



y_app_svm_rbf = clf_reg.predict(Xa)

err_app_svm_rbf = sum(y_app_svm_rbf != Ya)/Ya.size

print('SVM non linéaire : taux d''erreur apprentissage = {}%'.format(100*err_app_svm_rbf))



y_val_svm_rbf = clf_svm_rbf.predict(Xv)

err_val_svm_rbf = sum(y_val_svm_rbf != Yv)/Yv.size

print('SVM non linéaire : taux d''erreur validation = {}%'.format(100*err_val_svm_rbf))
# Kppv et GridSearchCV



k_values = [k for k in range(2,10)]



# the grid

parameters = [{"n_neighbors": k_values}]



# the classifier

clf_knn = KNeighborsClassifier()

# Perf a K-fold validation using the accuracy as the performance measure

K = 5 # feel free to adapt the value of $K$

# we will dot it on a grid search using n_jobs processors

clf_knn = GridSearchCV(clf_knn, param_grid=parameters, cv=K, scoring="accuracy", verbose=1, n_jobs = 2)

clf_knn.fit(Xa, Ya)
# Calcul des erreurs de classification



# Get the best parameters

print("\nKNN - optimal hyper-parameters = {}".format(clf_knn.best_params_))

print("KNN - best cross-val accuracy = {} \n".format(clf_knn.best_score_))



y_app_knn = clf_knn.predict(Xa)

err_app_knn = sum(y_app_knn != Ya)/Ya.size

print('KNN : taux d''erreur apprentissage = {}%'.format(100*err_app_knn))



y_val_knn = clf_knn.predict(Xv)

err_val_knn = sum(y_val_knn != Yv)/Yv.size

print('KNN : taux d''erreur validation = {}%'.format(100*err_val_knn))
# Prédiction sur les données de validation et matrice de confusion



Y_ldav = clf_lda.predict(Xv)

Y_qdav = clf_qda.predict(Xv)

y_val_reg = clf_reg.predict(Xv)

y_val_svm_lin = clf_svm_lin.predict(Xv)

y_val_svm_rbf = clf_svm_rbf.predict(Xv)

y_val_knn = clf_knn.predict(Xv)





# matrices de confusion

confmat1 = confusion_matrix(y_true=Yv, y_pred=Y_ldav)

confmat2 = confusion_matrix(y_true=Yv, y_pred=Y_qdav)

confmat3 = confusion_matrix(y_true=Yv, y_pred=y_val_reg)

confmat4 = confusion_matrix(y_true=Yv, y_pred=y_val_svm_lin)

confmat5 = confusion_matrix(y_true=Yv, y_pred=y_val_svm_rbf)

confmat6 = confusion_matrix(y_true=Yv, y_pred=y_val_knn)

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6,figsize=(20,15))



ax1.matshow(confmat1, cmap=plt.cm.Blues, alpha=0.3)

ax2.matshow(confmat2, cmap=plt.cm.Blues, alpha=0.3)

ax3.matshow(confmat3, cmap=plt.cm.Blues, alpha=0.3)

ax4.matshow(confmat4, cmap=plt.cm.Blues, alpha=0.3)

ax5.matshow(confmat5, cmap=plt.cm.Blues, alpha=0.3)

ax6.matshow(confmat6, cmap=plt.cm.Blues, alpha=0.3)



for i in range(confmat1.shape[0]):

    for j in range(confmat1.shape[1]):

        ax1.text(x=j, y=i, s=confmat1[i, j], va="center", ha="center")

ax1.set(xlabel='Label predit', ylabel='Vrai label')

ax1.set_title('LDA')



for i in range(confmat2.shape[0]):

    for j in range(confmat2.shape[1]):

        ax2.text(x=j, y=i, s=confmat2[i, j], va="center", ha="center")

ax2.set(xlabel='Label predit', ylabel='Vrai label')

ax2.set_title('QDA')



for i in range(confmat3.shape[0]):

    for j in range(confmat3.shape[1]):

        ax3.text(x=j, y=i, s=confmat3[i, j], va="center", ha="center")

ax3.set(xlabel='Label predit', ylabel='Vrai label')

ax3.set_title('Régression logistique')



for i in range(confmat4.shape[0]):

    for j in range(confmat4.shape[1]):

        ax4.text(x=j, y=i, s=confmat4[i, j], va="center", ha="center")

ax4.set(xlabel='Label predit', ylabel='Vrai label')

ax4.set_title('SVM linéaire')



for i in range(confmat5.shape[0]):

    for j in range(confmat5.shape[1]):

        ax5.text(x=j, y=i, s=confmat5[i, j], va="center", ha="center")

ax5.set(xlabel='Label predit', ylabel='Vrai label')

ax5.set_title('SVM non linéaire')



for i in range(confmat6.shape[0]):

    for j in range(confmat6.shape[1]):

        ax6.text(x=j, y=i, s=confmat6[i, j], va="center", ha="center")

ax6.set(xlabel='Label predit', ylabel='Vrai label')

ax6.set_title('KNN')



plt.show()



# accuracy: (tp + tn) / (p + n)

# precision tp / (tp + fp)

# recall: tp / (tp + fn)

# f1: 2 tp / (2 tp + fp + fn)



accuracy_lda = accuracy_score(Yv, Y_ldav)

precision_lda = precision_score(Yv, Y_ldav)

recall_lda = recall_score(Yv, Y_ldav)

f1_lda = f1_score(Yv, Y_ldav)



accuracy_qda = accuracy_score(Yv, Y_qdav)

precision_qda = precision_score(Yv, Y_qdav)

recall_qda = recall_score(Yv, Y_qdav)

f1_qda = f1_score(Yv, Y_qdav)



accuracy_reg = accuracy_score(Yv, y_val_reg)

precision_reg = precision_score(Yv, y_val_reg)

recall_reg = recall_score(Yv, y_val_reg)

f1_reg = f1_score(Yv, y_val_reg)



accuracy_svn_lin = accuracy_score(Yv, y_val_svm_lin)

precision_svn_lin = precision_score(Yv, y_val_svm_lin)

recall_svn_lin = recall_score(Yv, y_val_svm_lin)

f1_svn_lin = f1_score(Yv, y_val_svm_lin)



accuracy_svn_rbf = accuracy_score(Yv, y_val_svm_rbf)

precision_svn_rbf = precision_score(Yv, y_val_svm_rbf)

recall_svn_rbf = recall_score(Yv, y_val_svm_rbf)

f1_svn_rbf = f1_score(Yv, y_val_svm_rbf)



accuracy_knn = accuracy_score(Yv, y_val_knn)

precision_knn = precision_score(Yv, y_val_knn)

recall_knn = recall_score(Yv, y_val_knn)

f1_knn = f1_score(Yv, y_val_knn)



resultats = {'LDA':[accuracy_lda, precision_lda, recall_lda, f1_lda],

        'QDA':[accuracy_qda, precision_qda, recall_qda, f1_reg],

        'Reg. Log':[accuracy_reg, precision_reg, recall_reg, f1_reg],

        'SVM lin':[accuracy_svn_lin, precision_svn_lin, recall_svn_lin, f1_svn_lin], 

        'SVM non lin':[accuracy_svn_rbf, precision_svn_rbf, recall_svn_rbf, f1_svn_rbf], 

        'KNN':[accuracy_knn, precision_knn, recall_knn, f1_knn]}



df_res = pd.DataFrame(resultats, index= ['Accuracy', 'Precision', 'Recall', 'F-mesure'])

df_res
clf_svm_rbf.fit(X, y)

predictions_2 = clf_svm_rbf.predict(X_test)



output_2 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_2})

output_2.to_csv('my_submission_2_NON_LINEAR_SVM.csv', index=False)

print("Your submission was successfully saved!")
clf_knn.fit(X, y)

predictions_3 = clf_knn.predict(X_test)



output_3 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_3})

output_3.to_csv('my_submission_3_KNN.csv', index=False)

print("Your submission was successfully saved!")
clf_reg.fit(X, y)

predictions_4 = clf_reg.predict(X_test)



output_4 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_4})

output_4.to_csv('my_submission_4_LOGISTIC_REGRESSION.csv', index=False)

print("Your submission was successfully saved!")
def plot_roc_curve(fpr, tpr, AUC, modele): 

    plt.plot(fpr, tpr, color='orange', label='ROC - AUC = ' + str(round(AUC,2)))

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve - ' + modele)

    plt.legend()

    plt.show()

    

#fpr : false positive rate

#tpr : true positive rate

    

Y_ldav_proba = clf_lda.predict_proba(Xv)

Y_qdav_proba = clf_qda.predict_proba(Xv)

y_val_reg_proba = clf_reg.predict_proba(Xv)

y_val_svm_lin_proba = clf_svm_lin.predict_proba(Xv)

y_val_svm_rbf_proba = clf_svm_rbf.predict_proba(Xv)

y_val_knn_proba = clf_knn.predict_proba(Xv)



modele = ["LDA", "QDA", "Reg. Log", "SVM Lin", "SVM non lin", "KNN"]

res = [Y_ldav_proba[:,1], Y_qdav_proba[:,1], y_val_reg_proba[:,1], y_val_svm_lin_proba[:,1], y_val_svm_rbf_proba[:,1], y_val_knn_proba[:,1]]



for i, nom_modele in enumerate(modele): 

    print(i)

    auc = roc_auc_score(Yv, res[i])    

    fpr, tpr, thresholds = roc_curve(Yv, res[i])

    plt.figure(figsize=(16, 12))

    plt.subplot(3,2,i+1)

    plot_roc_curve(fpr, tpr, auc, nom_modele)
TC_LDA_REGLOG = contingency_matrix(Y_ldav, y_val_reg )

TC_LDA_SVM_NL = contingency_matrix(Y_ldav, y_val_svm_rbf )

TC_REGLOG_SVM_NL = contingency_matrix(y_val_reg, y_val_svm_rbf )



result_LDA_REGLOG = mcnemar(TC_LDA_REGLOG, exact=True)

result_LDA_SVM_NL = mcnemar(TC_LDA_SVM_NL, exact=True)

result_REGLOG_SVM_NL = mcnemar(TC_REGLOG_SVM_NL, exact=True)



resultats = {'LDA / Reg. log':[result_LDA_REGLOG.statistic, result_LDA_REGLOG.pvalue],

        'LDA / SVM non lin':[result_LDA_SVM_NL.statistic, result_LDA_SVM_NL.pvalue],

        'Reg. Log / SVM non lin':[result_REGLOG_SVM_NL.statistic, result_REGLOG_SVM_NL.pvalue]}



df_res = pd.DataFrame(resultats, index= ['Statistic', 'p-value'])

df_res