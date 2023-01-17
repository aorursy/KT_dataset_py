import time

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO) 
# Set to INFO for tracking training, default is WARN 

print("Using TensorFlow version %s" % (tf.__version__)) 

CATEGORICAL_COLUMNS = ["workclass", "education", 
                       "marital.status", "occupation", 
                       "relationship", "race", 
                       "sex", "native.country"]

# Columns of the input csv file
COLUMNS = ["age", "workclass", "fnlwgt", "education", 
           "education.num", "marital.status",
           "occupation", "relationship", "race", 
           "sex", "capital.gain", "capital.loss",
           "hours.per.week", "native.country", "income"]

# Feature columns for input into the model
FEATURE_COLUMNS = ["age", "workclass", "education", 
                   "education.num", "marital.status",
                   "occupation", "relationship", "race", 
                   "sex", "capital.gain", "capital.loss",
                   "hours.per.week", "native.country"]
import pandas as pd

df = pd.read_csv("../input/adult.csv")#, header=None, names=COLUMNS)
df.head(5)
df.describe(include=[np.number])
df.describe(include=[np.object])
df.columns
df.dtypes
from sklearn.model_selection import train_test_split
BATCH_SIZE = 40

num_epochs = 1
shuffle = True

# df = pd.read_csv("../input/adult.csv")
y = df["income"].apply(lambda x: ">50K" in x).astype(int)
del df["fnlwgt"] # Unused column
del df["income"] # Labels column, already saved to labels variable
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        num_epochs=num_epochs,
        shuffle=shuffle)

eval_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=X_test,
        y=y_test,
        batch_size=BATCH_SIZE,
        num_epochs=num_epochs,
        shuffle=shuffle)

print('input functions configured')

def generate_input_fn(filename, num_epochs=None, shuffle=True, batch_size=BATCH_SIZE):
    df = pd.read_csv(filename)#, header=None, names=COLUMNS)
    labels = df["income"].apply(lambda x: ">50K" in x).astype(int)
    del df["fnlwgt"] # Unused column
    del df["income"] # Labels column, already saved to labels variable
    
    type(df['age'].iloc[3])
    
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle)

# The layers module contains many utilities for creating feature columns.

# Categorical base columns.
sex = tf.feature_column.categorical_column_with_vocabulary_list(
    key="sex",                                                           
    vocabulary_list=["female", "male"])
race = tf.feature_column.categorical_column_with_vocabulary_list(
    key="race",                                                             
    vocabulary_list=["Amer-Indian-Eskimo",
                     "Asian-Pac-Islander",
                     "Black", "Other", "White"])

education = tf.feature_column.categorical_column_with_hash_bucket(
  "education", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket(
  "marital.status", hash_bucket_size=100)
relationship = tf.feature_column.categorical_column_with_hash_bucket(
  "relationship", hash_bucket_size=100)
workclass = tf.feature_column.categorical_column_with_hash_bucket(
  "workclass", hash_bucket_size=100)
occupation = tf.feature_column.categorical_column_with_hash_bucket(
  "occupation", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
  "native.country", hash_bucket_size=1000)

print('Categorical columns configured')
# Continuous base columns.
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education.num")
capital_gain = tf.feature_column.numeric_column("capital.gain")
capital_loss  = tf.feature_column.numeric_column("capital.loss")
hours_per_week = tf.feature_column.numeric_column("hours.per.week")

print('Continuous columns configured')
# Wide columns and deep columns.
wide_columns = [sex, race, native_country,
      education, occupation, workclass,
      marital_status, relationship]

deep_columns = [
    # Multi-hot indicator columns for columns with fewer possibilities
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(marital_status),
    tf.feature_column.indicator_column(sex),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.indicator_column(race),
    # Embeddings for categories with more possibilities. Should have at least (possibilties)**(0.25) dims
    tf.feature_column.embedding_column(education, dimension=8),
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
    # Numerical columns
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
]

print('wide and deep columns configured')
def create_model_dir(model_type):
    return 'models/model_' + model_type + '_' + str(int(time.time()))

# If new_model=False, pass in the desired model_dir 
def get_model(model_type, wide_columns=None, deep_columns=None, new_model=False, model_dir=None):
    if new_model or model_dir is None:
        model_dir = create_model_dir(model_type) # Comment out this line to continue training a existing model
    print("Model directory = %s" % model_dir)
    
    m = None
    
    # Linear Classifier
    if model_type == 'WIDE':
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir, 
            feature_columns=wide_columns)

    # Deep Neural Net Classifier
    if model_type == 'DEEP':
        m = tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50])

    # Combined Linear and Deep Classifier
    if model_type == 'WIDE_AND_DEEP':
        m = tf.estimator.DNNLinearCombinedClassifier(
                model_dir=model_dir,
                linear_feature_columns=wide_columns,
                dnn_feature_columns=deep_columns,
                dnn_hidden_units=[100, 70, 50, 25])
        
    print('estimator built')
    
    return m, model_dir
    
MODEL_TYPE = 'WIDE_AND_DEEP'
model_dir = create_model_dir(model_type=MODEL_TYPE)
m, model_dir = get_model(model_type = MODEL_TYPE, wide_columns=wide_columns, deep_columns=deep_columns, model_dir=model_dir)
%%time 

m.train(input_fn=train_input_fn)

print('training done')
results = m.evaluate(input_fn=eval_input_fn)
print('evaluate done')
print('\nAccuracy: %s' % results['accuracy'])
# Transformations.
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[ 18, 25, 30, 35, 40, 45, 50, 55, 60, 65 ])

education_occupation = tf.feature_column.crossed_column(
    ["education", "occupation"], hash_bucket_size=int(1e4))

age_race_occupation = tf.feature_column.crossed_column(
    [age_buckets, "race", "occupation"], hash_bucket_size=int(1e6))

country_occupation = tf.feature_column.crossed_column(
    ["native.country", "occupation"], hash_bucket_size=int(1e4))

print('Transformations complete')
# Update the wide columns
wide_columns += [age_buckets, education_occupation,
      age_race_occupation, country_occupation]

print('wide and deep columns configured')
# Create a new model
model_dir = create_model_dir(model_type=MODEL_TYPE)
m2, model_dir = get_model(model_type = MODEL_TYPE, wide_columns=wide_columns, deep_columns=deep_columns, model_dir=model_dir)

# train
m2.train(input_fn=train_input_fn)
print('training done')

# Eval
results = m2.evaluate(input_fn=eval_input_fn)
print('evaluate done')
print('\nAccuracy: %s' % results['accuracy'])
# Create a dataframe to wrap in an input function
# df = pd.read_csv("adult.test.csv")
start,end = 500,510
data_predict = X_test.iloc[start:end]
predict_labels = y_test.iloc[start:end]
print(predict_labels)
data_predict.head(10) # show this before deleting, so we know what the labels are
predict_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=data_predict,
        batch_size=1,
        num_epochs=1,
        shuffle=False)
predictions = m.predict(input_fn=predict_input_fn)

for prediction in predictions:
    print("Predictions:    {} with probabilities {}\n".format(prediction["classes"], prediction["probabilities"]))
predictions = m2.predict(input_fn=predict_input_fn)

for prediction in predictions:
    print("Predictions:    {} with probabilities {}\n".format(prediction["classes"], prediction["probabilities"]))
def column_to_dtype(column):
    if column in CATEGORICAL_COLUMNS:
        return tf.string
    else:
        return tf.float32
    
feature_spec = {
    column: tf.FixedLenFeature(shape=[1], dtype=column_to_dtype(column))
        for column in FEATURE_COLUMNS
}
serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
export_dir = m2.export_savedmodel(export_dir_base="models/export", 
                            serving_input_receiver_fn=serving_fn)
export_dir = export_dir.decode("utf8")
export_dir
!ls {export_dir}
!tar -zcvf exported_model.tgz {export_dir}
!ls