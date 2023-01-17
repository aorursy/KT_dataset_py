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
file = "/kaggle/input/digit-recognizer/train.csv"
df = pd.read_csv(file)
df.head()
df.shape
file = "/kaggle/input/digit-recognizer/test.csv"
test = pd.read_csv(file)
test.head()
test.shape
X = df.drop(['label'],axis=1)
y = df['label']
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True,style='darkgrid')
some_digit = np.array(X[25000:25001])
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap=matplotlib.cm.binary,interpolation="nearest")
plt.axis("off")
plt.show()
y[25000:25001]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape
X_test.shape

df.isnull().any().any()
test.isnull().any().any()
from sklearn.preprocessing import Normalizer,StandardScaler
normalizer = Normalizer()
scaler = StandardScaler()
Xn_train = normalizer.fit_transform(X_train)
Xn_test = normalizer.fit_transform(X_test)
Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.fit_transform(X_test)
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
model1 = sgd.fit(Xn_train,y_train)
y_pred1 = model1.predict(Xn_test)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
accuracy_score(y_test,y_pred1)
cross_val_score(sgd,Xn_train,y_train,scoring = "accuracy")
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(random_state=0)
model2 = dtree.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
accuracy_score(y_test,y_pred2)
X_pred = model2.predict(X_train)
accuracy_score(y_train,X_pred)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,max_features='auto',bootstrap=True,n_jobs=-1,random_state=0)
model3 = rfc.fit(X_train,y_train)
y_pred3 = model3.predict(X_test)
accuracy_score(y_test,y_pred3)
from sklearn.ensemble import BaggingClassifier
bgcl = BaggingClassifier(dtree,n_estimators=100,max_features=0.3,n_jobs=-1)
model4 = bgcl.fit(X_train,y_train)
y_pred4 = model4.predict(X_test)
accuracy_score(y_test,y_pred4)
from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7),n_estimators=500,learning_rate=0.1)
model5 = abcl.fit(X_train,y_train)
y_pred5 = model5.predict(X_test)
accuracy_score(y_test,y_pred5)
X_pred5 = model5.predict(X_train)
accuracy_score(y_train,X_pred5)
from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(random_state=1)
model6 = gbcl.fit(X_train,y_train)
y_pred6 = model6.predict(X_test)
accuracy_score(y_test,y_pred6 )
X_pred6 = model6.predict(X_train)
accuracy_score(y_train,X_pred6)
gb = GradientBoostingClassifier(n_estimators=30,learning_rate=0.3,max_depth=8,random_state=1)
model7 = gb.fit(X_train,y_train)
y_pred7 = model7.predict(X_test)
accuracy_score(y_test,y_pred7)
X_pred7 = model7.predict(X_train)
accuracy_score(y_train,X_pred7)
import xgboost as xgb
xgbc = xgb.XGBClassifier(objective='multi:softprob',verbosity=3,n_jobs=-1,subsample=0.8,colsample_bytree=0.5)
model8 = xgbc.fit(X_train,y_train)
y_pred8 = model8.predict(X_test)
accuracy_score(y_test,y_pred8)
sample_submission = model8.predict(test)
sample_submission
test_labels = pd.DataFrame(sample_submission,columns=['Label'])
test_labels.reset_index(inplace=True)
test_labels.columns = ['ImageId','Label']
test_labels.head()

test_labels.to_csv('Sample_submission.csv')
import os
os.chdir(r'../working')
from IPython.display import FileLink
FileLink(r'Sample_submission.csv')
