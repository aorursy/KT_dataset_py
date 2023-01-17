# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

from sklearn import pipeline,ensemble,preprocessing,feature_extraction,cross_validation,metrics
train=pd.read_json('../input/train.json')
train.head()
train.ingredients=train.ingredients.apply(' '.join)
train.head()
clf=pipeline.Pipeline([
        ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=-1))
    ])
# step 1: testing
X_train,X_test,y_train,y_test=cross_validation.train_test_split(train.ingredients,train.cuisine, test_size=0.2)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
metrics.confusion_matrix(y_test,y_pred)
metrics.accuracy_score(y_test,y_pred)
# step 2: real training
test=pd.read_json('../input/test.json')
test.ingredients=test.ingredients.apply(' '.join)
test.head()
clf.fit(train.ingredients,train.cuisine)
pred=clf.predict(test.ingredients)
df=pd.DataFrame({'id':test.id,'cuisine':pred})
df.to_csv('rf_tfidf.csv', columns=['id','cuisine'],index=False)
# LB score ~ 0.753