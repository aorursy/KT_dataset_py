# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test_ids = test.id 

del train['id']

del test['id']

y_train = train['species'].as_matrix()

del train['species']

X_test = test.as_matrix()

X_train = train.as_matrix()



# Any results you write to the current directory are saved as output.
pipe = Pipeline([('a', StandardScaler()), ('pca', TruncatedSVD(n_components = 10)), ('b', LinearDiscriminantAnalysis())])

pipe.fit(X_train, y_train)

classes = pipe.classes_

data = pipe.predict_proba(X_test)
sub = pd.DataFrame(data, columns = classes)

sub.insert(0,'id', test_ids)

sub.to_csv('submission.csv', index = False)