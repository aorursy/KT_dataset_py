import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')
# Models

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



# Feature Selection

from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit, KFold, train_test_split, StratifiedKFold



# Auxiliary Scores

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score
df = pd.read_csv('../input/avaliacao.csv')
df = pd.read_csv('../input/avaliacao.csv')

df.drop(axis=1, columns=[

    'shot_zone_range', 

    'shot_zone_area', 

    'action_type',

    'matchup', 

    'team_id', 

    'shot_zone_basic', 

    'shot_type', 

    'game_event_id',

    'game_id',

    'season', 

    'game_date',

    'lat', 

    'lon',

    'playoffs',

    'seconds_remaining', 

    'minutes_remaining', 

    'shot_id',

], inplace=True)

df = df.dropna()

df.head(10)
sns.heatmap(df.corr())
from sklearn.model_selection import cross_val_score

from sklearn import model_selection

X = df.drop(["shot_made_flag"],axis=1)

X = df_dummies_X_train = pd.get_dummies(X)

#juntar a coluna de mins e secs

y = df.shot_made_flag



from sklearn.preprocessing import MinMaxScaler



min_max_scaler = MinMaxScaler()



X[X.columns] = min_max_scaler.fit_transform(X[X.columns])

X.head(10)





X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)
import pandas as pd

from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import GradientBoostingClassifier
model = KNeighborsClassifier()

scores = cross_val_score(model, X, y, cv=5)



print('Acurácia de KNeighbors simples:', scores.mean())





clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)

clf2 = RandomForestClassifier(n_estimators=10, random_state=1)

clf3 = GaussianNB()

clf4 = KNeighborsClassifier()



eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('knn', clf4)], voting='hard')



for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble', 'KNN']):

    scores = cross_val_score(clf, X, y, cv=5)

    print("Acurácia: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#depois do ensemble



from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit

from sklearn import linear_model



%matplotlib inline

import matplotlib.pyplot as plt



import numpy as np

import seaborn as sns



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):



    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,

        scoring='neg_mean_squared_error')

    train_scores_mean = np.mean(train_scores, axis=1) * -1

    train_scores_std = np.std(train_scores, axis=1) * -1

    test_scores_mean = np.mean(test_scores, axis=1) * -1

    test_scores_std = np.std(test_scores, axis=1) * -1

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Treino")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Teste")



    plt.legend(loc="best")

    return plt





title = "Curva de Aprendizagem"

cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)



estimator = linear_model.Lasso(alpha=2.5)

plot_learning_curve(estimator, title, X_train, y_train, cv=cv, n_jobs=4)



plt.show()
# treinando o modelo

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

# predizendo

y_pred = knn.predict(X_test)



# comparando com gabarito

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_r = pca.fit_transform(X)

print(X_r)



plt.scatter(X_r[:,0],X_r[:,1])

plt.show()
best_model = None

best_accuracy = 0



for k in [1,2,3,4,5,6,7,8,9]:



    knn = KNeighborsClassifier(n_neighbors = k) # a cada passo, o parâmetro assume um valor

    knn.fit(X_train, y_train)



    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print('K:', k, '- ACC:', acc)

    

    if acc > best_accuracy:

        best_model = knn

        best_accuracy = acc

        

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)



print()

print('Melhor modelo:')

print('K:', best_model.get_params()['n_neighbors'], '- ACC:', acc * 100) #corrigir
# utilizando validação cruzada com cross_val_score

from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors = 1)

scores = cross_val_score(knn, X, y, cv=5) # 5 execuções diferentes com 20% dos dados para teste



print('Accuracy - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))
# utilizando validação cruzada com KFold

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits = 5)



acc = []

for train_index, test_index in kf.split(X, y): # precisa passar as classes agora para que a divisão aconteça

    knn = KNeighborsClassifier(n_neighbors = 1)

    knn.fit(X[train_index],y[train_index])

    y_pred = knn.predict(x[test_index])

    acc.append(accuracy_score(y_pred,y[test_index]))



acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão

print('Accuracy - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))