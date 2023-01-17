import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA



from sklearn.model_selection import GridSearchCV
def save_answers(answers):

    answers = answers.reshape(10000, 1)

    b = np.arange(1, 10001).reshape(10000, 1)

    result = np.concatenate((b, answers), axis=1)

    df = pd.DataFrame(result)

    df.columns = ['id', 'label']

    df.to_csv("answers.csv", index=False)
path = '/kaggle/input/jds101/'

df = pd.read_csv(path +'fashion-mnist_train.csv')

df_test = pd.read_csv(path + 'new_test.csv')



X = df.iloc[:,1:]

y = df.iloc[:,0]



X = X / 255

df_test = df_test / 255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
decision_tree = DecisionTreeClassifier()
decision_tree.get_params()
parameters = {

    'max_depth': [None, 10, 15, 20],

    'criterion': ['gini', 'entropy'],

    'splitter': ['best', 'random'],

    'min_samples_leaf': [1, 2, 3, 4, 5]

}
clf = GridSearchCV(decision_tree, parameters)
%%time

clf.fit(X_train, y_train)
res = (

    pd.DataFrame({

        "mean_test_score": clf.cv_results_["mean_test_score"],

        "mean_fit_time": clf.cv_results_["mean_fit_time"]})

      .join(pd.json_normalize(clf.cv_results_["params"]).add_prefix("param_"))

)

res
res['mean_test_score'].max()
knn = KNeighborsClassifier()
knn.get_params()
knn_parameters = {

    'algorithm': ['auto', 'ball_tree', 'kd_tree'],

    'weights': ['uniform', 'distance'],

    'n_jobs': [-1],

    'n_neighbors': [4, 11, 15, 20]

}
knn_clf = GridSearchCV(knn, knn_parameters)
%%time

knn_clf.fit(X_train, y_train)
res_knn = (

    pd.DataFrame({

        "mean_test_score": knn_clf.cv_results_["mean_test_score"],

        "mean_fit_time": knn_clf.cv_results_["mean_fit_time"]})

      .join(pd.json_normalize(knn_clf.cv_results_["params"]).add_prefix("param_"))

)

res_knn
good_knn = KNeighborsClassifier(n_neighbors=4, algorithm='kd_tree', weights='distance')
good_knn.fit(X, y)
answers = good_knn.predict(df_test)
save_answers(answers)