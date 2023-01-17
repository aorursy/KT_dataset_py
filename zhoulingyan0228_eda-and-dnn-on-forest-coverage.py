# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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
df_raw = pd.read_csv('../input/train.csv').drop(['Id'],axis=1)
df_raw.head()
df_features = df_raw.copy()
df_features['Distance_To_Hydrology'] = np.sqrt(df_raw['Horizontal_Distance_To_Hydrology'].values ** 2 + df_raw['Vertical_Distance_To_Hydrology'] ** 2)
pd.plotting.scatter_matrix(df_features[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']], c=df_raw['Cover_Type'], figsize=(20,20));
tmp = df_features[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Cover_Type']].groupby('Cover_Type').sum()
sns.heatmap(tmp, annot=True, fmt='d');
tmp = df_features[['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
       'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
       'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
       'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
       'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
       'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
       'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
       'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
       'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
       'Soil_Type39', 'Soil_Type40', 'Cover_Type']].groupby('Cover_Type').sum()
plt.subplots(figsize=(20,20))
sns.heatmap(tmp, annot=True, fmt='d', square=False);
plt.subplots(figsize=(20,20))
for i, c in enumerate(['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                    'Vertical_Distance_To_Hydrology', 'Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                    'Horizontal_Distance_To_Fire_Points']):
    plt.subplot(6, 2, i+1)
    sns.violinplot(x='Cover_Type', y=c, data=df_features[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
           'Vertical_Distance_To_Hydrology', 'Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
           'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
           'Horizontal_Distance_To_Fire_Points', 'Cover_Type']], scale="width");
df_train, df_validate = train_test_split(df_raw, test_size=0.2)

label_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 6), copy=True)
label_scaler.fit([[1],[7]])
                 
feature_scaler = sklearn.preprocessing.StandardScaler(copy=True)
feature_scaler.fit(df_train.drop('Cover_Type', axis=1).values)

#feature_selection = SelectKBest(mutual_info_classif, k=50)
#feature_selection.fit(df_train.drop('Cover_Type', axis=1).values, df_train['Cover_Type'].values)
#feature_selection.transform(df_train.drop('Cover_Type', axis=1).values)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': feature_scaler.transform(df_train.drop('Cover_Type', axis=1).values)},
    y=label_scaler.transform(df_train['Cover_Type'].values.reshape(-1, 1)).astype(np.int32).flatten(),
    batch_size=32,
    num_epochs=300,
    shuffle=True)

validate_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': feature_scaler.transform(df_validate.drop('Cover_Type', axis=1).values)},
    y=label_scaler.transform(df_validate['Cover_Type'].values.reshape(-1, 1)).astype(np.int32).flatten(),
    shuffle=False)
def model_fn(features, labels, mode, params):
    layer = tf.layers.dense(inputs=features['x'], units=512, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=128, activation=tf.nn.relu)
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
    params={'num_classes': 7,
           'weights': [3, 3, 1., 1., 1., 1., 1.]})
classifier.train(input_fn=train_input_fn)
print(classifier.evaluate(input_fn=validate_input_fn))
predicted = np.array(list(map(lambda x: x['classes'], classifier.predict(input_fn=validate_input_fn))))
df_validate['Cover_Type'].values.shape
df_validate['Cover_Type'].shape
tmp = pd.DataFrame(sklearn.metrics.confusion_matrix(df_validate['Cover_Type'].values, label_scaler.inverse_transform(predicted.reshape(-1,1)).flatten().astype(np.int32)))
plt.subplots(figsize=(10,10)) 
sns.heatmap(tmp, annot=True, fmt='.1f');
test_raw = pd.read_csv('../input/test.csv')
X_test = test_raw.drop(['Id'],axis=1)
ids = test_raw[['Id']]
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': feature_scaler.transform(X_test.values)},
    shuffle=False)
predicted = np.array(list(map(lambda x: x['classes'], classifier.predict(input_fn=test_input_fn))))

out_df = ids.copy()
out_df['Cover_Type'] = label_scaler.inverse_transform(predicted.reshape(-1,1)).flatten().astype(np.int32)
out_df.to_csv('./submission.csv', index=False)