import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.preprocessing as StandardScaler # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns 
import matplotlib.pyplot as plt 
import tensorflow as tf 
tf.logging.set_verbosity(tf.logging.ERROR)
df_raw = pd.read_csv('../input/train.csv')
df_raw.head()
df_features = df_raw[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
fillval = df_features.median()
df_features = df_features.fillna(fillval)
labels = df_raw['Survived']
df_features = pd.concat([df_features, pd.get_dummies(df_features['Pclass'], prefix='Pclass')], axis = 1)
df_features = pd.concat([df_features, pd.get_dummies(df_features['Sex'])], axis = 1)
df_features['FamilySize'] = df_features['SibSp'] + df_features['Parch']
df_features = df_features.drop(['Pclass', 'Sex', 'female'], axis=1)
df_features.head()
pd.DataFrame({'Non-Survivors': pd.concat([df_raw['Pclass'], labels], axis=1).groupby('Survived').get_group(0)['Pclass'],
              'Survivors':   pd.concat([df_raw['Pclass'], labels], axis=1).groupby('Survived').get_group(1)['Pclass']}).plot.hist(stacked=False, alpha=0.5);
pd.DataFrame({'Non-Survivors': pd.concat([df_features, labels], axis=1).groupby('Survived').get_group(0)['Age'],
              'Survivors':   pd.concat([df_features, labels], axis=1).groupby('Survived').get_group(1)['Age']}).plot.hist(stacked=False, alpha=0.5);
pd.DataFrame({'Non-Survivors': pd.concat([df_features, labels], axis=1).groupby('Survived').get_group(0)['male'],
              'Survivors':   pd.concat([df_features, labels], axis=1).groupby('Survived').get_group(1)['male']}).plot.hist(stacked=False, alpha=0.5);
pd.DataFrame({'Non-Survivors': pd.concat([df_features, labels], axis=1).groupby('Survived').get_group(0)['Fare'],
              'Survivors':   pd.concat([df_features, labels], axis=1).groupby('Survived').get_group(1)['Fare']}).plot.hist(stacked=False, alpha=0.5);
pd.DataFrame({'Non-Survivors': pd.concat([df_features, labels], axis=1).groupby('Survived').get_group(0)['FamilySize'],
              'Survivors':   pd.concat([df_features, labels], axis=1).groupby('Survived').get_group(1)['FamilySize']}).plot.hist(stacked=False, alpha=0.5);
df_feature_train, df_feature_validate, labels_train, labels_validate = train_test_split(df_features, labels, test_size=0.2)
                 
feature_scaler = sklearn.preprocessing.StandardScaler(copy=True)
feature_scaler.fit(df_feature_train.values)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': feature_scaler.transform(df_feature_train.values)},
    y=labels_train.values,
    batch_size=32,
    num_epochs=5000,
    shuffle=True)

validate_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': feature_scaler.transform(df_feature_validate.values)},
    y=labels_validate.values,
    shuffle=False)
def model_fn(features, labels, mode, params):
    layer = features['x']
    if mode == tf.estimator.ModeKeys.TRAIN:
        layer = tf.layers.dropout(inputs=layer, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
    layer = tf.layers.dense(inputs=layer, units=64, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=32, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=8)
    if mode == tf.estimator.ModeKeys.TRAIN:
        layer = tf.layers.dropout(inputs=layer, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Logits Layer
    logits = tf.layers.dense(inputs=layer, units=params['num_classes'])

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    weights = tf.gather(params['weights'], labels)
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights) 

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]),
        "recall": tf.metrics.recall(
            labels=labels, predictions=predictions["classes"]),
        "precision": tf.metrics.precision(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

classifier = tf.estimator.Estimator(
    model_fn=model_fn,
    params={'num_classes': 2,
           'weights': [1., 1]})
classifier.train(input_fn=train_input_fn)
print(classifier.evaluate(input_fn=validate_input_fn))
predicted = np.array(list(map(lambda x: x['classes'], classifier.predict(input_fn=validate_input_fn))))
tmp = pd.DataFrame(sklearn.metrics.confusion_matrix(labels_validate.values, predicted.astype(np.int32)))
plt.subplots(figsize=(4,4)) 
sns.heatmap(tmp, annot=True, fmt='.1f');
df_test_raw= pd.read_csv('../input/test.csv')
df_test = df_test_raw.fillna(fillval)
df_test.describe()
df_test = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
df_test = pd.concat([df_test, pd.get_dummies(df_test['Pclass'], prefix='Pclass')], axis = 1)
df_test = pd.concat([df_test, pd.get_dummies(df_test['Sex'])], axis = 1)
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']
df_test = df_test.drop(['Pclass', 'Sex', 'female'], axis=1)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': feature_scaler.transform(df_test.values)},
    shuffle=False)

predicted = list(map(lambda x: x['classes'], classifier.predict(input_fn=test_input_fn)))
df_result = df_test_raw[['PassengerId']]
df_result['Survived'] = predicted
df_result.to_csv('submission.csv', index=False, header=True)