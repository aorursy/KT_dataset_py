import os

import tensorflow as tf
tfrec_dir = '/kaggle/input/ednet-tfrecords-sequential/'

tfrec_fns = os.listdir(tfrec_dir)



# Sort the file - For verification purpose

tfrec_fns = sorted(tfrec_fns, key=lambda x: int(x.replace('EdNet-user-history-', '').replace('.tfrecord', '')))



tfrec_paths = [os.path.join(tfrec_dir, fn) for fn in tfrec_fns]

tfrec_paths
features = {

    'user_id': tf.io.FixedLenFeature([], dtype=tf.int64),

    # Zero partitions: returns 1D tf.Tensor for each Example.

    'row_id': tf.io.RaggedFeature(value_key='row_id', dtype=tf.int64),

    'timestamp': tf.io.RaggedFeature(value_key='timestamp', dtype=tf.int64),

    'content_id': tf.io.RaggedFeature(value_key='content_id', dtype=tf.int64),

    'content_type_id': tf.io.RaggedFeature(value_key='content_type_id', dtype=tf.int64),

    'task_container_id': tf.io.RaggedFeature(value_key='task_container_id', dtype=tf.int64),

    'user_answer': tf.io.RaggedFeature(value_key='user_answer', dtype=tf.int64),

    'answered_correctly': tf.io.RaggedFeature(value_key='answered_correctly', dtype=tf.int64),

    'prior_question_elapsed_time': tf.io.RaggedFeature(value_key='prior_question_elapsed_time', dtype=tf.float32),

    'prior_question_had_explanation': tf.io.RaggedFeature(value_key='prior_question_had_explanation', dtype=tf.int64),

}





def parse_example(example):



    return tf.io.parse_single_example(example, features)





# num_parallel_reads=1 to avoid reading in parallel, so the order is kept - for verification purpose here.

# In real appliation where the order is not important, set it to be > 1 to gain performance.

ds = tf.data.TFRecordDataset(tfrec_paths, num_parallel_reads=1)

ds = ds.map(parse_example)

for x in ds.take(3):

    print(x)

    print('-' * 40)
batched_ds = ds.apply(

    tf.data.experimental.dense_to_ragged_batch(batch_size=2)

)



for x in batched_ds.take(1):

    print(x)