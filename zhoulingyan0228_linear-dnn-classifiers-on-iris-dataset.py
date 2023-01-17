# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.estimator import LinearClassifier 
from tensorflow.estimator import BaselineClassifier
from tensorflow.estimator import DNNClassifier
from tensorflow.estimator.inputs import numpy_input_fn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def print_metrics(labels, tfpredictions):
    prediction_classes = list(map(lambda x: x['class_ids'][0], tfpredictions))
    print('Recall=', sklearn.metrics.recall_score(labels, prediction_classes, average='micro'))
    print('Precision=', sklearn.metrics.precision_score(labels, prediction_classes, average='micro'))
    print('Confution Matrix=\n', sklearn.metrics.confusion_matrix(labels, prediction_classes))
def plot_confusion_matrix(labels, tfpredictions) :
    prediction_classes = list(map(lambda x: x['class_ids'][0], tfpredictions))
    tmp = pd.DataFrame(sklearn.metrics.confusion_matrix(y_validate, prediction_classes))
    tmp.columns = speciedEncoder.inverse_transform(tmp.columns)
    tmp.index = speciedEncoder.inverse_transform(tmp.index)
    sns.heatmap(tmp, annot=True)
df_raw = pd.read_csv('../input/Iris.csv')
speciedEncoder = preprocessing.LabelEncoder()
speciedEncoder.fit(df_raw['Species'])
features = df_raw.copy()
labels_real = speciedEncoder.transform(df_raw['Species'])
features = features.drop(['Id', 'Species'], axis=1)
pd.plotting.scatter_matrix(df_raw, c=labels_real, figsize = [16,16])
features.head()
scaler = preprocessing.StandardScaler()
scaler.fit(features)
features_normalized = scaler.transform(features)
X_train, X_validate, y_train, y_validate = train_test_split(features_normalized, labels_real, test_size=0.2)
baselineClassifier = BaselineClassifier(
    n_classes=np.max(y_train)+1
)
baselineClassifier.train(input_fn=numpy_input_fn({'SepalLengthCm': X_train.T[0], 'SepalWidthCm': X_train.T[1], 'PetalLengthCm': X_train.T[2], 'PetalWidthCm': X_train.T[3]}, 
                                         y_train, 
                                         shuffle=True,
                                         batch_size=y_train.size))
print('Baseline Performance')
predictions = list(baselineClassifier.predict(input_fn=numpy_input_fn({'SepalLengthCm': X_validate.T[0], 'SepalWidthCm': X_validate.T[1], 'PetalLengthCm': X_validate.T[2], 'PetalWidthCm': X_validate.T[3]}, 
                                               shuffle=False)))
print_metrics(y_validate, predictions)
linearClassifier = LinearClassifier(
    feature_columns=[
        tf.feature_column.numeric_column('SepalLengthCm'),
        tf.feature_column.numeric_column('SepalWidthCm'),
        tf.feature_column.numeric_column('PetalLengthCm'),
        tf.feature_column.numeric_column('PetalWidthCm')],
    n_classes=np.max(y_train)+1,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
)
linearClassifier.train(input_fn=numpy_input_fn({'SepalLengthCm': X_train.T[0], 'SepalWidthCm': X_train.T[1], 'PetalLengthCm': X_train.T[2], 'PetalWidthCm': X_train.T[3]}, 
                                            y_train, 
                                            shuffle=True,
                                            batch_size=20,
                                            num_epochs=None),
                           steps=1000)
print('Linear Classifier Performance')
predictions = list(linearClassifier.predict(input_fn=numpy_input_fn({'SepalLengthCm': X_validate.T[0], 'SepalWidthCm': X_validate.T[1], 'PetalLengthCm': X_validate.T[2], 'PetalWidthCm': X_validate.T[3]}, 
                                               shuffle=False)))
print_metrics(y_validate, predictions)
dnnClassifier = DNNClassifier(
    feature_columns=[
        tf.feature_column.numeric_column('SepalLengthCm'),
        tf.feature_column.numeric_column('SepalWidthCm'),
        tf.feature_column.numeric_column('PetalLengthCm'),
        tf.feature_column.numeric_column('PetalWidthCm')],
    n_classes=int(np.max(y_train)+1),
    hidden_units=[16, 8],
    optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
)

dnnClassifier.train(input_fn=numpy_input_fn({'SepalLengthCm': X_train.T[0], 'SepalWidthCm': X_train.T[1], 'PetalLengthCm': X_train.T[2], 'PetalWidthCm': X_train.T[3]}, 
                                           y_train.astype('int'), 
                                           shuffle=True,
                                           batch_size=20,
                                           num_epochs=None),
                   steps=1000)
print('DNN Classifier Performance')
predictions = list(linearClassifier.predict(input_fn=numpy_input_fn({'SepalLengthCm': X_validate.T[0], 'SepalWidthCm': X_validate.T[1], 'PetalLengthCm': X_validate.T[2], 'PetalWidthCm': X_validate.T[3]}, 
                                               shuffle=False)))
print_metrics(y_validate, predictions)
plot_confusion_matrix(y_validate, predictions)