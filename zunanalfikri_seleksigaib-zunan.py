import pandas as pd

pd.options.display.max_columns = 999

import numpy as np

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

#sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from pandas_profiling import ProfileReport

import warnings

warnings.simplefilter(action='ignore')

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
test = pd.read_csv("../input/seleksidukungaib/test.csv")

train = pd.read_csv("../input/seleksidukungaib/train.csv")

sample_submission = pd.read_csv("../input/seleksidukungaib/sample_submission.csv")
train.columns
dropped_column = ['idx', 'userId', 'num_transfer_trx', 'max_transfer_trx',

                  'min_transfer_trx', 'date', 'date_collected', 'isUpgradedUser']
data = pd.concat([train,test],ignore_index=True)

data = data.drop(dropped_column, axis = 1)
data = data.drop(['average_transfer_trx'], axis = 1)
data.info()
#10 row null values on column isActive, isVerifiedPhone, isVerifiedEmail, blocked, super, userLevel, pinEnabled

data.loc[data.isActive.isnull() == True]

#is churned not null -> di data train -> drop 10 baris tersebut

data = data.dropna(subset=["isActive"])
#isi kolom bertipe object dengan modus

data['premium'] = data['premium'].fillna(data['premium'].mode())
# isi yang lainnya dengan median, kecuali isChurned

for column in data.columns:

    if (column != "isChurned"):

        data[column] = data[column].fillna(data[column].median())
# semua data sudah tidak ada yang null
# encode data yang bertype object agar mudah saat dilakukan modelling (beberapa model hanya bisa menggunakan tipe numeric, seperti regression model)

categorical_features = ['premium', 'super', 'pinEnabled']

le = LabelEncoder()

for col in categorical_features:

    data[col] = le.fit_transform(list(data[col].values))
#cap data ke 0.85 persen
Q3 = data.quantile(0.85)
Q3
data.describe()
numerik_col = ['average_recharge_trx','average_topup_trx','max_recharge_trx','max_topup_trx',

              'min_recharge_trx','min_topup_trx','num_recharge_trx','num_topup_trx','num_transaction',

              'random_number','total_transaction']

# for col in numerik_col:

#     data.loc[data[col] >= Q3[col], col] = Q3[col]
# average_topup_trx berkorelasi tinggi dengan max_topup_trx

# data['avg_topup_plus_max_topup'] = data['average_topup_trx'] + data['max_topup_trx']

# data.drop(['max_topup_trx', 'average_topup_trx'], axis=1, inplace = True)

# berdasar hasil experimen, lebih buruk jika digabung



# num_transaction berkorelasi tinggi dengan num_recharge_trx

data['num_transaction_plus_num_recharge'] = data['num_transaction'] + data['num_recharge_trx']

data.drop(['num_transaction', 'num_recharge_trx'], axis=1, inplace = True)

# karena train dan test adalah data di bulan berbeda, maka normalisasi dilakukan di masing2 data (split dulu)



## split data to train and test:

train = data[~data.isChurned.isnull()]

test = data[data.isChurned.isnull()]



# Normalisasi data numeric data:



numerik_col = ['max_recharge_trx','average_recharge_trx',

               'average_topup_trx', 'max_topup_trx',

              'min_recharge_trx','min_topup_trx','num_topup_trx',

              'random_number','total_transaction',

#               'num_recharge_trx','num_transaction'

              ]



for col in (numerik_col):

#     data[col]=((data[col]-data[col].min())/(data[col].max()-data[col].min()))

    train[col]=((train[col]-train[col].min())/(train[col].max()-train[col].min()))

    test[col]=((test[col]-test[col].min())/(test[col].max()-test[col].min()))

train.describe()
train.info()
print(len(train))

print(len(test))
# cek data duplikat 

train.duplicated().value_counts()



#hasil data duplikat -> 8000 row , action : drop duplikat
#drop data duplikat

train.drop_duplicates(keep = 'first', inplace = True) 

  
### Cari korelasi data

train.corr().style.background_gradient(cmap='coolwarm')
# analisis yang korelasinya kurang dari 0.1 dan lebih dari -0.1 (kecil)

# isActive, isVeriviedPhone, blocked, super, random_number

min_cor = ['isActive', 'isVerifiedPhone', 'blocked', 'super', 'random_number']

for col in min_cor:

    print('====== ', col, " ======")

    print(train[col].value_counts())

#isActive, isVerifiedPhone, blocked dan super memiliki varisi data yang sangat sedikit 90% ++ sama

# random_number tidak memiliki 

# aksi : drop semua kolom tersebut

drop_from_cor = ['isActive', 'isVerifiedPhone', 'blocked', 'super', 'random_number']

train.drop(drop_from_cor, axis = 1, inplace = True)

test.drop(drop_from_cor, axis = 1, inplace = True)
train.head()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.tree import ExtraTreeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm.classes import OneClassSVM

from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from sklearn.neighbors.classification import RadiusNeighborsClassifier

from sklearn.neighbors.classification import KNeighborsClassifier

from sklearn.multioutput import ClassifierChain

from sklearn.multioutput import MultiOutputClassifier

from sklearn.multiclass import OutputCodeClassifier

from sklearn.multiclass import OneVsOneClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model.stochastic_gradient import SGDClassifier

from sklearn.linear_model.ridge import RidgeClassifierCV

from sklearn.linear_model.ridge import RidgeClassifier

from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    

from sklearn.gaussian_process.gpc import GaussianProcessClassifier

# from sklearn.ensemble.voting_classifier import VotingClassifier

from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.ensemble.bagging import BaggingClassifier

from sklearn.ensemble.forest import ExtraTreesClassifier

from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import GaussianNB

from sklearn.semi_supervised import LabelPropagation

from sklearn.semi_supervised import LabelSpreading

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.naive_bayes import MultinomialNB  

from sklearn.neighbors import NearestCentroid

from sklearn.svm import NuSVC

from sklearn.linear_model import Perceptron

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import SVC

import xgboost as xgb

from xgboost import XGBClassifier

# from sklearn.mixture import DPGMM

# from sklearn.mixture import GMM 

# from sklearn.mixture import GaussianMixture

# from sklearn.mixture import VBGMM
Y = train["isChurned"]

X = train.drop(["isChurned"], axis = 1)
random_state = 1

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.2, random_state = random_state)
def get_kfold():

    return KFold(n_splits=5, shuffle=True, random_state=1)
all_model = [RandomForestClassifier(),ExtraTreeClassifier(), LogisticRegression(),RidgeClassifier(),

             DecisionTreeClassifier(), KNeighborsClassifier(), PassiveAggressiveClassifier(),

#              BernoulliNB(), GaussianNB(), CalibratedClassifierCV(), 

#              AdaBoostClassifier(), GradientBoostingClassifier()

#              LinearSVC(), NuSVC() , SVC()

            ]
params = {'loss_function':'Logloss', # objective function

          'eval_metric':'F1', # metric

#           'cat_features' : categorical_features,

          'iterations' : 1000,

          'learning_rate': 0.01, 

#            'task_type': "GPU", # enable GPU

          'verbose': 1000, # output to stdout info about training process every 1000 iterations

          'random_seed': random_state

         }

cbc = CatBoostClassifier(**params)
data_dmatrix = xgb.DMatrix(data=X,label=Y)
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 10, 'alpha': 10}



# cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,

#                     num_boost_round=50,early_stopping_rounds=10,metrics="auc", as_pandas=True, seed=123)
all_model.append(cbc)

all_model.append(XGBClassifier())
for model in all_model:

    scores = cross_val_score(model, X, Y, cv=get_kfold(), scoring='f1')

    print("Model : " , model , " has f1 score : " , scores.mean())
#setelah eksperimen dan percobaan submission, model logistic regression paling nggak overfitting

# choose logistic regression for Hyperparameter tuning
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import GridSearchCV
model = LogisticRegression()

solvers = ['newton-cg', 'lbfgs', 'liblinear']

penalty = ['l2']

c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers,penalty=penalty,C=c_values)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1',error_score=0, verbose = 3)

grid_result = grid_search.fit(X, Y)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
#use all data for final train, use best param from tuning

model =  LogisticRegression(C= 1.0, penalty= 'l2', solver = 'newton-cg')
scores = cross_val_score(model, X, Y, cv=get_kfold(), scoring='f1')

print(scores.mean())
model = model.fit(X,Y)
test = test.drop(['isChurned'], axis = True)
prediction = model.predict(test)
prediction = prediction.astype(int)
len(prediction)
len(sample_submission)
#LogisticRegression

sample_submission["isChurned"] = prediction

sample_submission["isChurned"].value_counts()
sample_submission.to_csv("submission.csv", index = False)
#best : logreg kfold => 0.894