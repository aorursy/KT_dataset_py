import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
data_raw = pd.read_csv("../input/train.csv")
data_raw["color"] = data_raw["color"].astype('category')
data_raw["type"] = data_raw["type"].astype('category')
data_raw.describe(include='all')
plt.subplots(figsize=(10,10))
plt.subplot(2,2,1)
sns.stripplot(x="type", y="bone_length", data=data_raw, jitter=True);
plt.subplot(2,2,2)
sns.stripplot(x="type", y="rotting_flesh", data=data_raw, jitter=True);
plt.subplot(2,2,3)
sns.stripplot(x="type", y="hair_length", data=data_raw, jitter=True);
plt.subplot(2,2,4)
sns.stripplot(x="type", y="has_soul", data=data_raw, jitter=True);
sns.pairplot(data_raw.drop('id', axis=1), hue="type");
sns.countplot(x='color', hue='type', data=data_raw);
typeEncoder = sklearn.preprocessing.LabelEncoder()
colorEncoder = sklearn.preprocessing.LabelBinarizer()
typeEncoder.fit(data_raw['type'])
colorEncoder.fit(data_raw['color'])

def encodeFeatures(dt) :
    return pd.concat([dt.drop(['color'], axis=1), pd.DataFrame(colorEncoder.transform(dt['color']), columns=colorEncoder.classes_)], axis=1)
labelsEncoded = typeEncoder.transform(data_raw['type'])
featuresEncoded = encodeFeatures(data_raw.drop(['type', 'id'], axis=1))

featureScaler = sklearn.preprocessing.StandardScaler()
featureScaler.fit(featuresEncoded)
featuresScaled = featureScaler.transform(featuresEncoded)
all_feature_column = tf.feature_column.numeric_column(key='x', shape=featuresScaled.shape[1])

kfold = sklearn.model_selection.KFold(n_splits=5)
for train_indices, validate_indices in kfold.split(featuresScaled):
    classifier = tf.estimator.DNNClassifier(
        feature_columns=[all_feature_column],
        n_classes=3,
        hidden_units=[16,8],
        dropout=0.5
    )
    classifier.train(input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x': featuresScaled[train_indices]},
        y=labelsEncoded[train_indices],
        batch_size=32,
        num_epochs=400,
        shuffle=True
    ))
    print(classifier.evaluate(input_fn=tf.estimator.inputs.numpy_input_fn(
        x={'x': featuresScaled[validate_indices]},
        y=labelsEncoded[validate_indices],
        shuffle=False
    )))

data_test = pd.read_csv('../input/test.csv')
featuresTest = featureScaler.transform(encodeFeatures(data_test.drop(['id'], axis=1)))
classifier = tf.estimator.DNNClassifier(
    feature_columns=[all_feature_column],
    n_classes=3,
    hidden_units=[16,8],
    dropout=0.5
)
classifier.train(input_fn=tf.estimator.inputs.numpy_input_fn(
    x={'x': featuresScaled},
    y=labelsEncoded,
    batch_size=32,
    num_epochs=400,
    shuffle=True
))
predictions_raw = list(classifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(
    x={'x': featuresTest},
    shuffle=False
)))
predicted_labels = typeEncoder.inverse_transform(list(map(lambda x: x['class_ids'][0], predictions_raw)))

pd.concat([data_test[['id']], pd.DataFrame({'type': predicted_labels})], axis=1).to_csv('submission.csv', index=False)