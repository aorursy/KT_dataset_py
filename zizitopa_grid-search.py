import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
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

X.shape, y.shape
X = X / 255

df_test = df_test / 255
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA



from sklearn.model_selection import GridSearchCV
pca = PCA()
pca_train = pca.fit(X)
X_pca_train = pca_train.transform(X_train)

X_pca_test = pca_train.transform(X_test)

X_pca_competition = pca_train.transform(df_test)
logistic_regression = LogisticRegression(n_jobs=-1)
logistic_regression.get_params()
parameters = {

    'C': [0.001, 0.01, 0.1, 10],

    'solver': ['saga', 'sag', 'newton-cg']

}
clf = GridSearchCV(logistic_regression, parameters)
clf.fit(X_pca_train, y_train)
sorted(clf.cv_results_)
res = (

    pd.DataFrame({

        "mean_test_score": clf.cv_results_["mean_test_score"],

        "mean_fit_time": clf.cv_results_["mean_fit_time"]})

      .join(pd.json_normalize(clf.cv_results_["params"]).add_prefix("param_"))

)
res
good_log_reg = LogisticRegression(n_jobs=-1, solver="newton-cg", C=0.1)
good_log_reg.fit(X, y)
answers = good_log_reg.predict(df_test)
save_answers(answers)