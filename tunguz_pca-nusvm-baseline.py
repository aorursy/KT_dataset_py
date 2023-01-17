import pandas as pd

from sklearn.decomposition import PCA

from sklearn import svm



# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test  = pd.read_csv("../input/Kannada-MNIST/test.csv")

submission  = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")





train_x = train.values[:,1:]

train_y = train.values[:,0]

test_x = test.values[:,1:]



pca = PCA(n_components=0.8,whiten=True)

train_x = pca.fit_transform(train_x)

test_x = pca.transform(test_x)





svc = svm.NuSVC()

svc.fit(train_x, train_y)



preds = svc.predict(test_x)

submission['label'] = preds

submission.to_csv('submission.csv', index=False)