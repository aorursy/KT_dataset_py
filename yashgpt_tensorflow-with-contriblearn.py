from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target
y
y.dtype
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
import tensorflow as tf
import tensorflow.contrib.learn.python



from tensorflow.contrib.learn.python import learn as learn
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input( X_train)

classifier=learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20,10],n_classes=3)
classifier.fit(X_train, y_train, steps=200, batch_size=32)
predictions = list(classifier.predict(X_test, as_iterable=True))
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input( X_train)

classifier=learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[20,20,20],n_classes=3)

classifier.fit(X_train, y_train, steps=300, batch_size=32)

predictions = list(classifier.predict(X_test, as_iterable=True))
print(classification_report(y_test,predictions))