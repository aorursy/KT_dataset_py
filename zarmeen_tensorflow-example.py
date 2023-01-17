import numpy as np

import pandas as pd

import tensorflow as tf

import math
flags = tf.app.flags

FLAGS = flags.FLAGS



flags.DEFINE_integer('num_classes', 99, 'Number of classes.')

flags.DEFINE_integer('num_variables', 192, 'Number of variables.')



# Hyper Parameters

flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')

flags.DEFINE_integer('hidden2', 1024, 'Number of units in hidden layer 2.')

flags.DEFINE_integer('num_epochs', 200, 'Number of learning epochs.')

flags.DEFINE_integer('batch_size', 90, 'Batch size.')

flags.DEFINE_float('keep_prob', 0.5, 'Keep probability for drop out.')

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
def inference(data, data_size, keep_prob):

    # Hidden layer 1

    with tf.name_scope('hidden1'):

        weights = tf.Variable(tf.truncated_normal([data_size, FLAGS.hidden1],

                                                  stddev=1.0 / math.sqrt(float(data_size))), name='weights1')

        biases = tf.Variable(tf.zeros([FLAGS.hidden1]), name='biases1')

        hidden1 = tf.nn.relu(tf.matmul(data, weights) + biases)



        # Dropout before layer 2

        hidden1_drop = tf.nn.dropout(hidden1, keep_prob, name='layer1_dropout')



    # Hidden layer 2

    with tf.name_scope('hidden2'):

        weights = tf.Variable(tf.truncated_normal([FLAGS.hidden1, FLAGS.hidden2],

                                                  stddev=1.0 / math.sqrt(float(FLAGS.hidden1))), name='weights2')

        biases = tf.Variable(tf.zeros([FLAGS.hidden2]), name='biases2')

        hidden2 = tf.nn.sigmoid(tf.matmul(hidden1_drop, weights) + biases)



        # Dropout before linear reading out

        hidden2_drop = tf.nn.dropout(hidden2, keep_prob, name='layer2_dropout')



    # Read out

    with tf.name_scope('softmax_linear'):

        weights = tf.Variable(tf.truncated_normal([FLAGS.hidden2, FLAGS.num_classes],

                                                  stddev=1.0 / math.sqrt(float(FLAGS.hidden2))), name='weights')

        biases = tf.Variable(tf.zeros([FLAGS.num_classes]), name='biases')

        logits = tf.matmul(hidden2_drop, weights) + biases



        return logits
def loss(logits, labels):

    labels = tf.to_int64(labels)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')

    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss
def training(loss):

    tf.summary.scalar(loss.op.name, loss)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(loss, global_step=global_step)



    return train_op
def evaluation(logits, labels):

    correct = tf.nn.in_top_k(logits, labels, 1)

    return tf.reduce_sum(tf.cast(correct, tf.int32))
def preprocess_data(data):

    _data = data.copy()



    # delete id column

    del _data['id']



    # normalize float type columns 

    for column in _data:

        if _data[column].dtypes == float:

            _data[column] = z_score_normalization(_data[column])



    # categorize 'species' column and insert 'species_cat' column

    if 'species' in _data.columns:

        _data.insert(2, 'species_cat', _data['species'].astype('category').cat.codes)

        _data.drop('species', axis=1, inplace=True)

    _data = _data.astype(float)



    return _data, _data.columns.size-1
def load_data_and_labels(file, is_labels_exist=True):

    df = pd.read_csv(file)

    processed_df, variables_size = preprocess_data(df)

    print('Reading %s' % file)

    print('N=%d' % len(df))



    if is_labels_exist is True:

        labels = list(processed_df['species_cat'])

    else:

        labels = None



    if is_labels_exist is True:

        del processed_df['species_cat']



    # transform DataFrame into list 

    new_rows = []

    for index, row in processed_df.iterrows():

        _list_row = []

        for col in processed_df:

            _list_row.append(row[col])

        new_rows.append(_list_row)



    if is_labels_exist is True:

        return new_rows, labels

    else:

        return new_rows
def z_score_normalization(series_of_values):

    _series_of_values = (series_of_values - series_of_values.mean()) / series_of_values.std()

    return _series_of_values
def shuffle_data(data, labels):

    # transform list into DataFrame 

    new_df = pd.DataFrame(data)

    new_df['__labels__'] = labels

    new_df = new_df.reindex(np.random.permutation(new_df.index))



    new_labels = list(new_df['__labels__'])

    del new_df['__labels__']



    # transform DataFrame into list 

    new_row = []

    for index, row in new_df.iterrows():

        _list_row = []

        for col in new_df:

            _list_row.append(row[col])

        new_row.append(_list_row)



    return new_row, new_labels
def run_training(data, labels):

    # Tell TensorFlow that the model will be built into the default Graph.

    with tf.Graph().as_default():

        data_size = FLAGS.num_variables

        num_classes = FLAGS.num_classes



        data_placeholder = tf.placeholder("float", shape=(None, data_size))

        labels_placeholder = tf.placeholder("int32", shape=None)

        keep_prob = tf.placeholder("float")



        # Build a Graph that computes predictions from the inference model.

        logits = inference(data_placeholder, data_size, keep_prob)



        # Add to the Graph the loss calculation.

        loss_op = loss(logits, labels_placeholder)



        # Add to the Graph operations that train the model.

        train_op = training(loss_op)



        # Add the Op to compare the logits to the labels during evaluation.

        eval_correct = evaluation(logits, labels_placeholder)



        # Build the summary Tensor based on the TF collection of Summaries.

        summary = tf.summary.merge_all()



        # The op for initializing the variables.

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



        # check point 

        saver = tf.train.Saver()



        # Create a session for running operations in the Graph.

        sess = tf.Session()



        # Instantiate a SummaryWriter to output summaries and the Graph.

        summary_writer = tf.summary.FileWriter('.', sess.graph)



        # Initialize the variables (the trained variables and the epoch counter).

        sess.run(init_op)



        # training

        for epoch in range(FLAGS.num_epochs):

            for i in range(int(len(data)/FLAGS.batch_size)):

                batch = FLAGS.batch_size*i

                sess.run(train_op, feed_dict={

                    data_placeholder: data[batch:batch + FLAGS.batch_size],

                    labels_placeholder: labels[batch:batch + FLAGS.batch_size],

                    keep_prob: FLAGS.keep_prob})



            # calculate accuracy in every epoch

            train_accuracy = sess.run(eval_correct, feed_dict={

                data_placeholder: data,

                labels_placeholder: labels,

                keep_prob: FLAGS.keep_prob})

            print("epoch %d, acc %g" % (epoch, train_accuracy / len(labels)))



            # update TensorBoard

            summary_str = sess.run(summary, feed_dict={

                data_placeholder: data,

                labels_placeholder: labels,

                keep_prob: 1.0})

            summary_writer.add_summary(summary_str, epoch)



            # shuffling data for next epoch

            data, labels = shuffle_data(data, labels)



        # Save a checkpoint and evaluate the model periodically.

        # Create a saver for writing training checkpoints.

        path = saver.save(sess, 'models.ckpt')

        print('checkpoint is saved at ' + path)



        sess.close()
train_data, train_labels = load_data_and_labels('../input/train.csv', is_labels_exist=True)
run_training(train_data, train_labels)
def run_classifier(data):

    with tf.Graph().as_default():

        data_size = FLAGS.num_variables

        num_classes = FLAGS.num_classes

        data_placeholder = tf.placeholder("float", shape=(None, data_size))



        logits = inference(data_placeholder, data_size, 1.0)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())



        sess = tf.Session()

        sess.run(init_op)



        saver = tf.train.Saver()

        saver.restore(sess, './models.ckpt')



        prediction_list = []

        total = len(data)

        with sess.as_default():

            for i in range(total):

                # print('#%d : ' % i, end='')

                v = logits.eval(feed_dict={data_placeholder: [data[i]]})

                sm = tf.nn.softmax(v)

                smv = sm.eval()

                prediction_list.append(smv[0])



                cls = np.argmax(smv)   # get the class number with largest value by argmax

                # print('prediction=%d (%f)' % (cls, smv[0][cls]))



        return prediction_list
def make_output(class_list):

    result_df = pd.DataFrame(class_list)

    train_df = (pd.read_csv('../input/train.csv'))

    test_df = (pd.read_csv('../input/test.csv'))



    cat_label_list = train_df['species'].astype('category').cat.categories

    new_columns = ['id']

    new_columns.extend(list(cat_label_list))



    # insert id field on left most columns

    result_df.insert(0, 'id', test_df['id'])



    # set columns names

    result_df.columns = new_columns



    # write out csv

    result_df.to_csv(path_or_buf='predicted.csv', index=False, encoding='utf-8')
test_data = load_data_and_labels('../input/test.csv', is_labels_exist=False)

result_list = run_classifier(test_data)

make_output(result_list)
# predicted.csv is created in the current folder.