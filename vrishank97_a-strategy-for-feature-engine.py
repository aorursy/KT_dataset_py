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



import warnings

warnings.filterwarnings('ignore')

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import r2_score
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Remove the outlier

train=train[train.y<250]
# Check no. of rows greater than equal to 100

len(train['y'][(train.y>=100)])
# Check no. of rows less than 100

len(train['y'][(train.y<100)])
train['y_class'] = train.y.apply(lambda x: 0 if x<100  else 1 )
# Concat the datasets

data = pd.concat([train,test])
# Removing object type vars as I am more interested in binary ones

data = data.drop(data.select_dtypes(include = ['object']).columns,axis=1)
feat = list(data.drop(['y','y_class'],axis=1).columns.values)
train_df = (data[:train.shape[0]])

test_df = (data[train.shape[0]:])
# I have not removed zero valued columns for now

len(feat)
# Remove ID as we want some honest features :)

feat.remove('ID')
from sklearn.metrics import f1_score as f1
# Calculating CV score

def cv_score(model):

    return cross_val_score(model,train_df[feat],train_df['y_class'],cv=10,scoring = 'f1').mean()
from sklearn.tree import DecisionTreeClassifier as DTC
model = DTC(max_depth = 5,min_samples_split=200) # We don't want to overfit
cv_score(model) 
model.fit(train_df[feat],train_df.y_class)
# Graphviz is used to build decision trees

from sklearn.tree import export_graphviz

from sklearn import tree
# This statement builds a dot file.

tree.export_graphviz(model, out_file='tree.dot',feature_names  = feat)  
# This will bring the image to the notebook (or you can view it locally)

from IPython.display import Image

#Image("tree.png") # Uncomment if you are trying this on local

# Can't read the image in kernal. Anyone know how? Will try and add image in comments