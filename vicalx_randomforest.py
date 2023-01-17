# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

os.chdir("/kaggle/input")

print(os.listdir("../input"))

import pandas as pd

import numpy as np

import datetime



from sklearn.ensemble import *

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import *

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV



from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score

from sklearn.metrics.scorer import make_scorer

import category_encoders as ce

from sklearn.metrics import *
import pandas as pd

import numpy as np

import datetime

dtype = {'DayOfWeek': np.uint8, 'DayOfMonth': np.uint8, 'Month': np.uint8 ,

         'Distance': np.float32, 'UniqueCarrier': str, 

         'Origin': str, 'Dest': str,

         'DepHour': np.uint8,'DepTime':np.float32,'DelayTime':int}



df = pd.read_csv('../input/finalset/Train.csv',dtype=dtype)



df.head(2)
df=df.drop(df.columns[0],axis=1)
df=df.drop(df.columns[3],axis=1)


from bs4 import BeautifulSoup

raw_html = open("../input/codescrap/codes.html").read()

html = BeautifulSoup(raw_html, 'html.parser')

col=[]

for li in html.select('tr'):

    if len(li.select('td'))>1:

        

        r1="{}".format(li.select('td')[1].text)

        r2="{}".format(li.select('td')[2].text)

        col.append([r1,r2])

import pandas as pd

pdf=pd.DataFrame(col,columns=["country","code"])

df=df.merge(pdf,left_on='Dest',right_on='code')

df=df.rename(index=str, columns={"country": "Cdest"})

df=df.drop('code',axis=1)

df=df.merge(pdf,left_on='Origin',right_on='code')

df=df.rename(index=str, columns={"country": "Corigin"})

df=df.drop('code',axis=1)



# We create the preprocessing pipelines for both numeric and categorical data.

numeric_features = ['Month', 'DayofMonth', 'DayOfWeek','Distance', 'DepHour']

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])



categorical_features = ['UniqueCarrier', 'Corigin', 'Cdest']

categorical_transformer = Pipeline(steps=[('labelencoder',OrdinalEncoder())])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])



# Append classifier to preprocessing pipeline.

# Now we have a full prediction pipeline.





X = df.drop('DelayTime', axis=1).drop('Origin',axis=1).drop('Dest',axis=1)

X_train=preprocessor.fit_transform(X)

rforest = RandomForestClassifier(n_estimators=250,max_depth=26, random_state=0)

rforest.fit(X_train, y)

rf = rforest





# Supervised transformation based on gradient boosted trees

grd = GradientBoostingClassifier(n_estimators=8)



grd.fit(X_train, y)









importances = rforest.feature_importances_

std = np.std([tree.feature_importances_ for tree in rforest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    

    print("%d. feature %s (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

imp_df = pd.DataFrame({'feature': X.columns.values,

                       'importance': forest.feature_importances_})

 

# Reorder by importance

ordered_df = imp_df.sort_values(by='importance')

imp_range=range(1,len(imp_df.index)+1)

 

## Barplot with confidence intervals

height = ordered_df['importance']

bars = ordered_df['feature']

y_pos = np.arange(len(bars))



# Create horizontal bars

plt.barh(y_pos, height)

 

# Create names on the y-axis

plt.yticks(y_pos, bars)



plt.xlabel("Mean reduction in tree impurity in random forest")



plt.tight_layout()

# Show graphic

plt.rcParams['figure.figsize'] = (8,8)

plt.show()
tf = pd.read_csv('../input/finalset/Test.csv',dtype=dtype)

tf=tf.merge(pdf,left_on='Dest',right_on='code')

tf=tf.rename(index=str, columns={"country": "Cdest"})

tf=tf.drop('code',axis=1)

tf=tf.merge(pdf,left_on='Origin',right_on='code')

tf=tf.rename(index=str, columns={"country": "Corigin"})

tf=tf.drop('code',axis=1)

Xtest = tf[['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 'Corigin', 'Cdest',

       'Distance', 'DepHour']]

ytest = tf["DelayTime"]


X_test=preprocessor.fit_transform(Xtest)



grd_prd=grd.predict(X_test)

rf_prd=rforest.predict(X_test)



y_pred_grd = grd.predict_proba(X_test)[:, 1]

fpr_grd, tpr_grd, _ = roc_curve(ytest, y_pred_grd)



# The random forest model by itself

y_pred_rf = rf.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(ytest, y_pred_rf)





plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')



plt.plot(fpr_rf, tpr_rf, label='RF')



plt.plot(fpr_grd, tpr_grd, label='GBT')



plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()



rforest.score(X_test,ytest)
grd.score(X_test,ytest)
##calculating metrics



["AUC SCORES",roc_auc_score(ytest, y_pred_grd,average='weighted'),roc_auc_score(ytest, y_pred_rf,average='weighted')]

["AUC SCORES",balanced_accuracy_score(ytest, grd_prd),balanced_accuracy_score(ytest, rf_prd)] 