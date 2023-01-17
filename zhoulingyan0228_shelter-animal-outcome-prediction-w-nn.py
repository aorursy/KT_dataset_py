import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
tf.logging.set_verbosity(tf.logging.ERROR)

df = pd.read_csv('../input/train.csv')
df.head()
features = df.drop(['AnimalID', 'OutcomeSubtype', 'Name'],axis=1).copy()
labels = df[['OutcomeType']].copy()
features.head()
def mapDateTime(row):
    try:
        dt = datetime.strptime(row['DateTime'], '%Y-%m-%d %H:%M:%S')
        return pd.Series([dt.year, dt.month, dt.day], index=['year', 'month', 'day'])
    except:
        return pd.Series([2015, 1, 2], index=['year', 'month', 'day'])
    
def mapSexuponOutcome(row):
    try:
        intactness, gender = row['SexuponOutcome'].split(' ')
        intactness_val = 1
        gender_val = 1
        if intactness in ('Neutered', 'Spayed', 'neutered', 'spayed'):
            intactness_val = 0
        if gender in ('female', 'Female'):
            gender_val = 0
        return pd.Series([intactness_val, 1-intactness_val, 1-gender_val, gender_val], index=['intact', 'notintact', 'female', 'male'])
    except:
        return pd.Series([0, 1, 0, 1], index=['intact', 'notintact', 'female', 'male'])
    
def mapAgeuponOutcome(row):
    try:
        digit, unit = row['AgeuponOutcome'].split(' ')
        unitDict = {'day':1, 'days':1, 'week':7, 'weeks':7, 'month':30, 'months':30, 'year':365, 'years':365}
        return pd.Series([int(digit)*unitDict[unit]], index=['age'])
    except:
        return pd.Series([0], index=['age'])
    
def mapBreed(row):
    try:
        breeds = row['Breed'].replace(' Mix', '').split('/')
        return pd.Series([breeds[0], breeds[-1]], index=['breed1', 'breed2'])
    except:
        return pd.Series(['Domestic Shorthair', 'Domestic Shorthair'], index=['breed1', 'breed2'])
    
def mapColor(row):
    try:
        colors = row['Color'].split('/')
        return pd.Series([colors[0], colors[-1]], index=['color1', 'color2'])
    except:
        return pd.Series(['Brown', 'Brown'], index=['color1', 'color2'])
    
datetimeDf = features.apply(mapDateTime, axis=1)
sexuponOutcomeDf = features.apply(mapSexuponOutcome, axis=1)
ageuponOutcomeDf = features.apply(mapAgeuponOutcome, axis=1)
breedLabelDf = features.apply(mapBreed, axis=1)
colorLabelDf = features.apply(mapColor, axis=1)
breedEncoder = sklearn.preprocessing.LabelBinarizer()
breedEncoder.fit(pd.concat([breedLabelDf['breed1'], breedLabelDf['breed2']]))
animalTypeEncoder = sklearn.preprocessing.LabelBinarizer()
animalTypeEncoder.fit(features['AnimalType'])
colorEncoder = sklearn.preprocessing.LabelBinarizer()
colorEncoder.fit(pd.concat([colorLabelDf['color1'], colorLabelDf['color2']]))
labelEncoder = sklearn.preprocessing.LabelEncoder()
labelEncoder.fit(labels)
color1Df = pd.DataFrame(colorEncoder.transform(colorLabelDf['color1']))
color1Df.columns = colorEncoder.classes_
color1Df = color1Df.add_prefix('color1_')
color2Df = pd.DataFrame(colorEncoder.transform(colorLabelDf['color2']))
color2Df.columns = colorEncoder.classes_
color2Df = color2Df.add_prefix('color2_')
breed1Df = pd.DataFrame(breedEncoder.transform(breedLabelDf['breed1']))
breed1Df.columns = breedEncoder.classes_
breed1Df = breed1Df.add_prefix('breed1_')
breed2Df = pd.DataFrame(breedEncoder.transform(breedLabelDf['breed2']))
breed2Df.columns = breedEncoder.classes_
breed2Df = breed2Df.add_prefix('breed2_')
animalTypeDf = pd.DataFrame(animalTypeEncoder.transform(features['AnimalType']))
animalTypeDf.columns = animalTypeEncoder.classes_[[1]]
animalTypeDf = animalTypeDf.add_prefix('type_')
plt.subplots(figsize=(5,5))
labels.groupby('OutcomeType').size().plot.bar();

plt.subplots(figsize=(20,20))
tmp = pd.concat([color1Df, labels], axis=1).groupby('OutcomeType').sum()
tmp = tmp / tmp.sum()
sns.heatmap(tmp.T, annot=True, fmt='0.1f');

plt.subplots(figsize=(20,60))
tmp = pd.concat([breed1Df, labels], axis=1).groupby('OutcomeType').sum()
tmp = tmp / tmp.sum()
sns.heatmap(tmp.T, annot=True, fmt='0.1f');

plt.subplots(figsize=(5,5))
tmp = pd.concat([sexuponOutcomeDf[['intact', 'notintact']], labels], axis=1).groupby('OutcomeType').sum()
tmp = tmp / tmp.sum()
sns.heatmap(tmp.T, annot=True, fmt='0.2f');

plt.subplots(figsize=(5,5))
tmp = pd.concat([sexuponOutcomeDf[['male', 'female']], labels], axis=1).groupby('OutcomeType').sum()
tmp = tmp / tmp.sum()
sns.heatmap(tmp.T, annot=True, fmt='0.2f');

plt.subplots(figsize=(5,5))
tmp = pd.concat([animalTypeDf, 1-animalTypeDf.rename({'type_Dog': 'type_Cat','type_Cat': 'type_Dog'}, axis=1), labels], axis=1).groupby('OutcomeType').sum()
tmp = tmp / tmp.sum()
sns.heatmap(tmp.T, annot=True, fmt='0.2f');
labelsEncoded = pd.DataFrame(labelEncoder.transform(labels.values.flatten()), columns=['OutcomeType'])
featuresExtended = pd.concat([animalTypeDf, datetimeDf, sexuponOutcomeDf, ageuponOutcomeDf, color1Df, color2Df, breed1Df, breed2Df], axis=1)
featureScaler = sklearn.preprocessing.StandardScaler()
featuresProcessed = featureScaler.fit_transform(featuresExtended)
featuresTrain, featuresValidate, labelsTrain, labelsValidate = sklearn.model_selection.train_test_split(featuresProcessed, labelsEncoded.values, test_size=0.2)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': featuresTrain},
    y=labelsTrain,
    batch_size=128,
    num_epochs=50,
    shuffle=True)

validate_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': featuresValidate},
    y=labelsValidate,
    shuffle=False)
def model_fn(features, labels, mode, params):
    layer = features['x']
    layer = tf.layers.dense(inputs=layer, units=1024, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=512, activation=tf.nn.relu)
    layer = tf.layers.dense(inputs=layer, units=256, activation=tf.nn.relu)
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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
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
    params={'num_classes': 5,
           'weights': [1., 1., 1., 1., 1.]})
classifier.train(input_fn=train_input_fn)
print(classifier.evaluate(input_fn=validate_input_fn))
raw_predictions = list(classifier.predict(input_fn=validate_input_fn))
predicted_classes = list(map(lambda x: x['classes'], raw_predictions))
sns.heatmap(sklearn.metrics.confusion_matrix(labelsValidate, np.array(predicted_classes).reshape((-1,1))), annot=True);
from sklearn.metrics import log_loss

predicted_probs = list(map(lambda x: x['probabilities'], raw_predictions))
print(log_loss(labelsValidate, predicted_probs) )
testDf = pd.read_csv('../input/test.csv')
ids = testDf[['ID']].copy()
featuresTest = testDf[['DateTime', 'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']]

def preprocessFeatures(featuresTest):
    datetimeDf = featuresTest.apply(mapDateTime, axis=1)
    sexuponOutcomeDf = featuresTest.apply(mapSexuponOutcome, axis=1)
    ageuponOutcomeDf = featuresTest.apply(mapAgeuponOutcome, axis=1)
    breedLabelDf = featuresTest.apply(mapBreed, axis=1)
    colorLabelDf = featuresTest.apply(mapColor, axis=1)
    
    color1Df = pd.DataFrame(colorEncoder.transform(colorLabelDf['color1']))
    color1Df.columns = colorEncoder.classes_
    color1Df = color1Df.add_prefix('color1_')
    color2Df = pd.DataFrame(colorEncoder.transform(colorLabelDf['color2']))
    color2Df.columns = colorEncoder.classes_
    color2Df = color2Df.add_prefix('color2_')
    breed1Df = pd.DataFrame(breedEncoder.transform(breedLabelDf['breed1']))
    breed1Df.columns = breedEncoder.classes_
    breed1Df = breed1Df.add_prefix('breed1_')
    breed2Df = pd.DataFrame(breedEncoder.transform(breedLabelDf['breed2']))
    breed2Df.columns = breedEncoder.classes_
    breed2Df = breed2Df.add_prefix('breed2_')
    animalTypeDf = pd.DataFrame(animalTypeEncoder.transform(featuresTest['AnimalType']))
    animalTypeDf.columns = animalTypeEncoder.classes_[[1]]
    animalTypeDf = animalTypeDf.add_prefix('type_')
    
    featuresExtended = pd.concat([animalTypeDf, datetimeDf, sexuponOutcomeDf, ageuponOutcomeDf, color1Df, color2Df, breed1Df, breed2Df], axis=1)
    featuresProcessed = featureScaler.transform(featuresExtended)
    return featuresProcessed

featuresProcessed = preprocessFeatures(featuresTest)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': featuresProcessed},
    shuffle=False)

raw_predictions = list(classifier.predict(input_fn=test_input_fn))
predicted_probs = list(map(lambda x: x['probabilities'], raw_predictions))
pd.concat([ids, pd.DataFrame(predicted_probs, columns=labelEncoder.inverse_transform(np.arange(0, 5)))], axis=1).to_csv('submission.csv', index=False)