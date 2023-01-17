import tensorflow.contrib.learn as skflow

import tensorflow as tf

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import train_test_split



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



def encode(train, test):

    label_encoder = LabelEncoder().fit(train.species)

    labels = label_encoder.transform(train.species)

    classes = list(label_encoder.classes_)



    train = train.drop(['species', 'id'], axis=1)

    scaler = StandardScaler().fit(train.values)

    scaled_train = scaler.transform(train.values)

    

    test = test.drop('id', axis=1)

    scaled_test = scaler.transform(test.values)



    return scaled_train, labels, scaled_test, classes



train_values, labels, test_values, classes = encode(train, test)



X_train, X_test, y_train, y_test = train_test_split(train_values, labels)



feature_columns = [tf.contrib.layers.real_valued_column("", dimension=X_train.shape[0])]

n_classes = len(classes)

nn_shape = [

    int(n_classes*1.5),

    int(n_classes*2.5),

    int(n_classes*1.5),

]



classifier = skflow.DNNClassifier(hidden_units=nn_shape, n_classes=n_classes, feature_columns=feature_columns)

classifier.fit(X_train, y_train, steps=1000)

score = metrics.accuracy_score(y_test, list(classifier.predict(X_test)))

print("Accuracy: %f" % score)