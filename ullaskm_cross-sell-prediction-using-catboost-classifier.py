import pandas as pd

import numpy  as np

import math

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split,KFold,cross_val_score

from sklearn.metrics import confusion_matrix,roc_auc_score

from pandas_profiling import ProfileReport

from imblearn.over_sampling import SMOTE,SMOTENC,ADASYN

from sklearn.model_selection import StratifiedKFold

# Comment this if the data visualisations doesn't work on your side

%matplotlib inline



plt.style.use('bmh')
import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')

df_test = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')

df_train.head()
# profile = ProfileReport(df_train,title='Profile Report',explorative=True)

# profile.to_widgets()
df_train.info()
df_test.info()
def label_encoder(data):

    le = preprocessing.LabelEncoder()

    data = le.fit_transform(data)

    return data
df_train['Gender'] = label_encoder(df_train['Gender'])

df_train['Vehicle_Age'] = label_encoder(df_train['Vehicle_Age'])

df_train['Vehicle_Damage'] = label_encoder(df_train['Vehicle_Damage'])

df_train.head()
df_train['Policy_Sales_Channel']  = df_train['Policy_Sales_Channel'].astype(int)

df_train['Region_Code']  = df_train['Region_Code'].astype(int)

df_train.head()
category_col=['Gender','Driving_License', 'Region_Code', 'Previously_Insured','Vehicle_Damage',

         'Vehicle_Age','Vintage','Policy_Sales_Channel']



X = df_train.copy()

y = X['Response']

X.drop(['id','Response'],axis=1,inplace=True)

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25,shuffle=True,random_state=99,stratify=y)

print("Size of X_train ", X_train.shape)

print("Size of X_cv ", X_cv.shape)

print("Size of y_train ", y_train.shape)

print("Size of y_cv ", y_cv.shape)

from catboost import CatBoostClassifier

params = {'iterations':1000,

        'learning_rate':0.1,

        'cat_features': category_col,

        'depth':7,

        'eval_metric':'AUC',

        'loss_function':'Logloss',

        'verbose':200,

        'od_type':"Iter", # overfit detector

        'od_wait':300, # most recent best iteration to wait before stopping

        'random_seed': 99,

        'l2_leaf_reg' : 11

          }



cat_model = CatBoostClassifier(**params)

cat_model.fit(X_train, y_train,   

          eval_set=(X_cv, y_cv), 

          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score

          plot=True  

         );
import statsmodels.api as sm

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary())
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y, cat_model.predict_proba(X)[:,1])

fpr, tpr, thresholds = roc_curve(y, cat_model.predict_proba(X)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Cataboost (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
X_test = df_test.copy()



# Encoding category using Label Encoder

X_test['Gender'] = label_encoder(X_test['Gender'])

X_test['Vehicle_Damage'] = label_encoder(X_test['Vehicle_Damage'])

X_test['Vehicle_Age'] = label_encoder(X_test['Vehicle_Age'])

X_test.drop(['id'],axis=1, inplace=True)

X_test['Policy_Sales_Channel']  = X_test['Policy_Sales_Channel'].astype(int)

X_test['Region_Code']  = X_test['Region_Code'].astype(int)

X_test.head()
response = cat_model.predict_proba(X_test)

response
submission = pd.DataFrame(df_test['id'],columns=['id',])

submission['Response'] = response[:, 1]

submission.head(10)