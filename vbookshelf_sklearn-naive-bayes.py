import pandas as pd
# read the data into a pandas datafrome
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print(df_train.shape)
print(df_test.shape)
X = df_train.drop('label', axis=1)
y = df_train['label']

X_test = df_test

print(X.shape)
print(y.shape)
print(X_test.shape)
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X, y)

y_pred = nb.predict(X_test)

# The index should start from 1 instead of 0
df = pd.Series(range(1,28001),name = "ImageId")

ID = df

submission = pd.DataFrame({'ImageId':ID, 
                           'Label':y_pred, 
                          }).set_index('ImageId')

submission.to_csv('mnist_nb.csv', columns=['Label']) 
