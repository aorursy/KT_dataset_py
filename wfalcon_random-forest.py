# import libs
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
# read training
train_df = pd.read_csv('../input/train.csv')

# extract the x and y for the models
X_tr = train_df.values[:, 1:].astype(float)
Y_tr = train_df.values[:, 0]
# train neural network
print ('training...')
clf = RandomForestClassifier(100)
clf = clf.fit(X_tr, Y_tr)
print ('training complete...')
scores = cross_val_score(clf, X_tr, Y_tr)
print ('Accuracy {0}'.format(np.mean(scores)))
# Read test data
test_df = pd.read_csv('../input/test.csv')
X_test = test_df.values.astype(float)

# make predictions
Y_test = clf.predict(X_test)
# make DF to print easily
ans = pd.DataFrame(data={'ImageId':range(1,len(Y_test)+1), 'Label':Y_test})

# save to csv
# ans.to_csv('/Users/.../randforest.csv', index=False)