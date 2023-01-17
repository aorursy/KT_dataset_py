# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# See https://github.com/vienai8d/tfrecordutil/wiki.

!pip install git+https://github.com/vienai8d/tfrecordutil.git@v0.1
# Convert to TFRecord.

!tfrecordutil-csv2tfrecord /kaggle/input/hotel-booking-demand/hotel_bookings.csv hotel_bookings.tfrecord
import tensorflow as tf

tf.__version__
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

df
from tfrecordutil import create_example_schema, read_example_tfrecord

schema = create_example_schema(df)

schema
examples = len(df)

batch_size = 100

epoch = 100

shuffle_buffer_size = examples

max_steps = epoch * examples / batch_size



def dataset():

    def parse(feature):

        label = feature.pop('is_canceled')

        return (feature, label)

    return read_example_tfrecord('hotel_bookings.tfrecord', schema).map(parse)



def train_input_fn():

    return dataset().repeat().shuffle(shuffle_buffer_size).batch(batch_size)



def eval_input_fn():

    return dataset().batch(batch_size)



train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)

eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
hidden_units=[128, 64, 32]

feature_columns = [

    tf.feature_column.embedding_column(

        tf.feature_column.categorical_column_with_vocabulary_list('hotel', df['hotel'].unique().tolist()), 2

    ),

    tf.feature_column.embedding_column(

        tf.feature_column.categorical_column_with_vocabulary_list('country', df['country'].dropna().unique().tolist()), 4

    ),

    tf.feature_column.embedding_column(

        tf.feature_column.categorical_column_with_vocabulary_list('market_segment', df['distribution_channel'].unique().tolist()), 2

    ),

    tf.feature_column.embedding_column(

        tf.feature_column.categorical_column_with_vocabulary_list('distribution_channel', df['distribution_channel'].unique().tolist()), 2

    ),

    tf.feature_column.embedding_column(

        tf.feature_column.categorical_column_with_vocabulary_list('reserved_room_type', df['reserved_room_type'].unique().tolist()), 2

    ),

    tf.feature_column.embedding_column(

        tf.feature_column.categorical_column_with_vocabulary_list('customer_type', df['customer_type'].unique().tolist()), 2

    ),

    tf.feature_column.numeric_column('lead_time'),

    tf.feature_column.numeric_column('stays_in_weekend_nights'),

    tf.feature_column.numeric_column('stays_in_week_nights'),

    tf.feature_column.numeric_column('adults'),

    tf.feature_column.numeric_column('previous_cancellations'),

    tf.feature_column.numeric_column('previous_bookings_not_canceled'),

    tf.feature_column.numeric_column('booking_changes'),

    tf.feature_column.numeric_column('required_car_parking_spaces'),

]

estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=hidden_units)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec )