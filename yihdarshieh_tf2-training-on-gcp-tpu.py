import os

import sys

import json

import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import absl

import datetime

from tensorflow.keras.optimizers import Adam

from adamw_optimizer import AdamW

from tensorflow.python.lib.io.file_io import recursive_create_dir



print(tf.__version__)





# # ----------------------------------------------------------------------------------------

# Ref: https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/custom_training.ipynb#scrollTo=jwJtsCQhHK-E





# Your TPU node internal ip

TPU_WORKER = 'grpc://XXX.XXX.XXX.XXX:8470'



# Your TPU Zone, for example 'europe-west4-a'

ZONE = ''



# Your project name, for example, 'kaggle-nq-123456'

PROJECT = ''



# Your training tf record file on Google Storage bucket. For example, gs://kaggle-my-nq-competition/nq_train.tfrecord

TRAIN_TF_RECORD = ''



# Your checkpoint dir on Google Storage bucket. For example, "gs://kaggle-my-nq-competition/checkpoints/"

CHECKPOINT_DIR = ''





tf.keras.backend.clear_session()



# # ----------------------------------------------------------------------------------------





IS_KAGGLE = True

INPUT_DIR = "/kaggle/input/"



# The original Bert Joint Baseline data.

BERT_JOINT_BASE_DIR = os.path.join(INPUT_DIR, "bertjointbaseline")



# This nq dir contains all files for publicly use.

NQ_DIR = os.path.join(INPUT_DIR, "nq-competition")





MY_OWN_NQ_DIR = NQ_DIR



# For local usage.

if not os.path.isdir(INPUT_DIR):

    IS_KAGGLE = False

    INPUT_DIR = "./"

    NQ_DIR = "./"

    MY_OWN_NQ_DIR = "./"





for dirname, _, filenames in os.walk(INPUT_DIR):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# NQ_DIR contains some packages / modules

sys.path.append(NQ_DIR)

sys.path.append(os.path.join(NQ_DIR, "transformers"))



from nq_flags import DEFAULT_FLAGS as FLAGS

from nq_flags import del_all_flags

from nq_dataset_utils import *



import sacremoses as sm

import transformers

from adamw_optimizer import AdamW



from transformers import TFBertModel

from transformers import TFBertMainLayer, TFBertPreTrainedModel

from transformers.modeling_tf_utils import get_initializer





from transformers import BertTokenizer

from transformers import TFBertModel, TFDistilBertModel

from transformers import TFBertMainLayer, TFDistilBertMainLayer, TFBertPreTrainedModel, TFDistilBertPreTrainedModel

from transformers.modeling_tf_utils import get_initializer





PRETRAINED_MODELS = {

    "BERT": [

        'bert-base-uncased',

        'bert-large-uncased-whole-word-masking-finetuned-squad',

    ],

    "DISTILBERT": [

        'distilbert-base-uncased-distilled-squad'

    ]

}





flags = absl.flags

del_all_flags(flags.FLAGS)



flags.DEFINE_bool(

    "do_lower_case", True,

    "Whether to lower case the input text. Should be True for uncased "

    "models and False for cased models.")



vocab_file = os.path.join(NQ_DIR, "vocab-nq.txt")



flags.DEFINE_string("vocab_file", vocab_file,

                    "The vocabulary file that the BERT model was trained on.")



flags.DEFINE_integer(

    "max_seq_length_for_training", 512,

    "The maximum total input sequence length after WordPiece tokenization for training examples. "

    "Sequences longer than this will be truncated, and sequences shorter "

    "than this will be padded.")



flags.DEFINE_integer(

    "max_seq_length", 512,

    "The maximum total input sequence length after WordPiece tokenization. "

    "Sequences longer than this will be truncated, and sequences shorter "

    "than this will be padded.")



flags.DEFINE_integer(

    "doc_stride", 128,

    "When splitting up a long document into chunks, how much stride to "

    "take between chunks.")



flags.DEFINE_float(

    "include_unknowns_for_training", 0.02,

    "If positive, for converting training dataset, probability of including answers of type `UNKNOWN`.")



flags.DEFINE_float(

    "include_unknowns", -1.0,

    "If positive, probability of including answers of type `UNKNOWN`.")



flags.DEFINE_boolean(

    "skip_nested_contexts", True,

    "Completely ignore context that are not top level nodes in the page.")



flags.DEFINE_integer("max_contexts", 48,

                     "Maximum number of contexts to output for an example.")



flags.DEFINE_integer(

    "max_position", 50,

    "Maximum context position for which to generate special tokens.")



flags.DEFINE_integer(

    "max_query_length", 64,

    "The maximum number of tokens for the question. Questions longer than "

    "this will be truncated to this length.")







    

flags.DEFINE_string("train_tf_record", TRAIN_TF_RECORD,

                    "Precomputed tf records for training dataset.")



flags.DEFINE_bool("do_train", False, "Whether to run training dataset.")





flags.DEFINE_string(

    "input_checkpoint_dir", CHECKPOINT_DIR,

    "The root directory that contains checkpoints to be loaded of all trained models.")



flags.DEFINE_string("model_dir", NQ_DIR, "Root dir of all Hugging Face's models")



flags.DEFINE_string("model_name", "distilbert-base-uncased-distilled-squad", "Name of Hugging Face's model to use.")



flags.DEFINE_integer("epochs", 0, "Total epochs for training.")



flags.DEFINE_integer("train_batch_size", 64 * 8, "Batch size for training.")



flags.DEFINE_integer("shuffle_buffer_size", 100000, "Shuffle buffer size for training.")



flags.DEFINE_float("init_learning_rate", 5e-5, "The initial learning rate for AdamW optimizer.")



flags.DEFINE_bool("cyclic_learning_rate", True, "If to use cyclic learning rate.")



flags.DEFINE_float("init_weight_decay_rate", 0.01, "The initial weight decay rate for AdamW optimizer.")



flags.DEFINE_integer("num_warmup_steps", 0, "Number of training steps to perform linear learning rate warmup.")



flags.DEFINE_integer("num_train_examples", None, "Number of precomputed training steps in 1 epoch.")



# Make the default flags as parsed flags

FLAGS.mark_as_parsed()



NB_SHORT_ANSWER_TYPES = 5



# ----------------------------------------------------------------------------------------





def get_dataset(tf_record_file, seq_length, batch_size=1, shuffle_buffer_size=0, is_training=False):



    if is_training:

        features = {

            "unique_ids": tf.io.FixedLenFeature([], tf.int64),

            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),

            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),

            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),

            "start_positions": tf.io.FixedLenFeature([], tf.int64),

            "end_positions": tf.io.FixedLenFeature([], tf.int64),

            "answer_types": tf.io.FixedLenFeature([], tf.int64)

        }

    else:

        features = {

            "unique_ids": tf.io.FixedLenFeature([], tf.int64),

            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),

            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),

            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64)

        }        



    def decode_record(record, features):

        """Decodes a record to a TensorFlow example."""

        example = tf.io.parse_single_example(record, features)



        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.

        # So cast all int64 to int32.

        for name in list(example.keys()):

            t = example[name]

            if t.dtype == tf.int64:

                t = tf.cast(t, tf.int32)

            example[name] = t

        return example



    def select_data_from_record(record):

        

        x = {

            'unique_ids': record['unique_ids'],

            'input_ids': record['input_ids'],

            'input_mask': record['input_mask'],

            'segment_ids': record['segment_ids']

        }



        if is_training:

            y = {

                'short_start_positions': record['start_positions'],

                'short_end_positions': record['end_positions'],

                'short_answer_types': record['answer_types']

            }



            return (x, y)

        

        return x



    dataset = tf.data.TFRecordDataset(tf_record_file)

    

    dataset = dataset.map(lambda record: decode_record(record, features))

    dataset = dataset.map(select_data_from_record)

    

    if shuffle_buffer_size > 0:

        dataset = dataset.shuffle(shuffle_buffer_size)

    

    dataset = dataset.batch(batch_size, drop_remainder=True)

    

    return dataset





# ----------------------------------------------------------------------------------------





cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER, zone=ZONE, project=PROJECT)

tf.config.experimental_connect_to_cluster(cluster_resolver)

tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)





# ----------------------------------------------------------------------------------------





if FLAGS.num_train_examples is None:

    FLAGS.num_train_examples = 494670





# ----------------------------------------------------------------------------------------





class TFNQModel:

    

    def __init__(self, config, *inputs, **kwargs):

        """

        

        Subclasses of this class are different in self.backend,

        which should be a model that outputs a tensor of shape (batch_size, hidden_dim), and the

        `backend_call()` method.

        

        We will use Hugging Face Bert/DistilBert as backend in this notebook.

        """



        self.backend = None

        

        self.seq_output_dropout = tf.keras.layers.Dropout(kwargs.get('seq_output_dropout_prob', 0.05))

        self.pooled_output_dropout = tf.keras.layers.Dropout(kwargs.get('pooled_output_dropout_prob', 0.05))

    

        self.short_pos_classifier = tf.keras.layers.Dense(2,

                                        kernel_initializer=get_initializer(config.initializer_range),

                                        name='pos_classifier')       



        self.short_answer_type_classifier = tf.keras.layers.Dense(NB_SHORT_ANSWER_TYPES,

                                        kernel_initializer=get_initializer(config.initializer_range),

                                        name='answer_type_classifier')        

                

    def backend_call(self, inputs, **kwargs):

        """This method should be implemented by subclasses.

           

           The implementation should take into account the (somehow) different input formats of Hugging Face's

           models.

           

           For example, the `TFDistilBert` model, unlike `Bert` model, doesn't have segment_id as input.

           

           Then it calls `self.backend_call()` to get the outputs from Bert's model, which is used in self.call().

        """

        

        raise NotImplementedError



    

    def call(self, inputs, **kwargs):

        

        # sequence / [CLS] outputs from original bert

        sequence_output, pooled_output = self.backend_call(inputs, **kwargs)  # shape = (batch_size, seq_len, hidden_dim) / (batch_size, hidden_dim)

        

        # dropout

        sequence_output = self.seq_output_dropout(sequence_output, training=kwargs.get('training', False))

        pooled_output = self.pooled_output_dropout(pooled_output, training=kwargs.get('training', False))

    

        short_pos_logits = self.short_pos_classifier(sequence_output)  # shape = (batch_size, seq_len, 2)

        

        short_start_pos_logits = short_pos_logits[:, :, 0]  # shape = (batch_size, seq_len)

        short_end_pos_logits = short_pos_logits[:, :, 1]  # shape = (batch_size, seq_len)

        

        short_answer_type_logits = self.short_answer_type_classifier(pooled_output)  # shape = (batch_size, NB_SHORT_ANSWER_TYPES)



        outputs = (short_start_pos_logits, short_end_pos_logits, short_answer_type_logits)



        return outputs  # logits

    

    

class TFBertForNQ(TFNQModel, TFBertPreTrainedModel):

    

    def __init__(self, config, *inputs, **kwargs):

        

        TFBertPreTrainedModel.__init__(self, config, *inputs, **kwargs)  # explicit calls without super

        TFNQModel.__init__(self, config)



        self.bert = TFBertMainLayer(config, name='bert')

    

    def backend_call(self, inputs, **kwargs):

        

        outputs = self.bert(inputs, **kwargs)

        sequence_output, pooled_output = outputs[0], outputs[1]  # shape = (batch_size, seq_len, hidden_dim) / (batch_size, hidden_dim)

        

        return sequence_output, pooled_output





class TFDistilBertForNQ(TFNQModel, TFDistilBertPreTrainedModel):

    

    def __init__(self, config, *inputs, **kwargs):

        

        TFDistilBertPreTrainedModel.__init__(self, config, *inputs, **kwargs)  # explicit calls without super

        TFNQModel.__init__(self, config)



        self.backend = TFDistilBertMainLayer(config, name="distilbert")

        

    def backend_call(self, inputs, **kwargs):

        

        if isinstance(inputs, tuple):

            # Distil bert has no segment_id (i.e. `token_type_ids`)

            inputs = inputs[:2]

        else:

            inputs = inputs

        

        outputs = self.backend(inputs, **kwargs)

        

        # TFDistilBertModel's output[0] is of shape (batch_size, sequence_length, hidden_size)

        # We take only for the [CLS].

        

        sequence_output = outputs[0]  # shape = (batch_size, seq_len, hidden_dim)

        pooled_output = sequence_output[:, 0, :]  # shape = (batch_size, hidden_dim)

        

        return sequence_output, pooled_output





model_mapping = {

    "bert": TFBertForNQ,

    "distilbert": TFDistilBertForNQ

}





def get_pretrained_model(model_name):

    

    pretrained_path = os.path.join(FLAGS.model_dir, model_name)

    

    tokenizer = BertTokenizer.from_pretrained(pretrained_path)

    

    model_type = model_name.split("-")[0]

    if model_type not in model_mapping:

        raise ValueError("Model definition not found.")

    

    model_class = model_mapping[model_type]

    model = model_class.from_pretrained(pretrained_path)

    

    return tokenizer, model





def get_metrics(name):



    loss = tf.keras.metrics.Mean(name=f'{name}_loss')



    loss_short_start_pos = tf.keras.metrics.Mean(name=f'{name}_loss_short_start_pos')

    loss_short_end_pos = tf.keras.metrics.Mean(name=f'{name}_loss_short_end_pos')

    loss_short_ans_type = tf.keras.metrics.Mean(name=f'{name}_loss_short_ans_type')

    

    acc = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc')

    

    acc_short_start_pos = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc_short_start_pos')

    acc_short_end_pos = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc_short_end_pos')

    acc_short_ans_type = tf.keras.metrics.SparseCategoricalAccuracy(name=f'{name}_acc_short_ans_type')

    

    return loss, loss_short_start_pos, loss_short_end_pos, loss_short_ans_type, acc, acc_short_start_pos, acc_short_end_pos, acc_short_ans_type





class CustomSchedule(tf.keras.optimizers.schedules.PolynomialDecay):

    

    def __init__(self,

      initial_learning_rate,

      decay_steps,

      end_learning_rate=0.0001,

      power=1.0,

      cycle=False,

      name=None,

      num_warmup_steps=1000):

        

        # Since we have a custom __call__() method, we pass cycle=False when calling `super().__init__()` and

        # in self.__call__(), we simply do `step = step % self.decay_steps` to have cyclic behavior.

        super(CustomSchedule, self).__init__(initial_learning_rate, decay_steps, end_learning_rate, power, cycle=False, name=name)

        

        self.num_warmup_steps = num_warmup_steps

        

        self.cycle = tf.constant(cycle, dtype=tf.bool)

        

    def __call__(self, step):

        """ `step` is actually the step index, starting at 0.

        """

        

        # For cyclic behavior

        step = tf.cond(self.cycle and step >= self.decay_steps, lambda: step % self.decay_steps, lambda: step)

        

        learning_rate = super(CustomSchedule, self).__call__(step)



        # Copy (including the comments) from original bert optimizer with minor change.

        # Ref: https://github.com/google-research/bert/blob/master/optimization.py#L25

        

        # Implements linear warmup: if global_step < num_warmup_steps, the

        # learning rate will be `global_step / num_warmup_steps * init_lr`.

        if self.num_warmup_steps > 0:

            

            steps_int = tf.cast(step, tf.int32)

            warmup_steps_int = tf.constant(self.num_warmup_steps, dtype=tf.int32)



            steps_float = tf.cast(steps_int, tf.float32)

            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)



            # The first training step has index (`step`) 0.

            # The original code use `steps_float / warmup_steps_float`, which gives `warmup_percent_done` being 0,

            # and causing `learning_rate` = 0, which is undesired.

            # For this reason, we use `(steps_float + 1) / warmup_steps_float`.

            # At `step = warmup_steps_float - 1`, i.e , at the `warmup_steps_float`-th step, 

            #`learning_rate` is `self.initial_learning_rate`.

            warmup_percent_done = (steps_float + 1) / warmup_steps_float

            

            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done



            is_warmup = tf.cast(steps_int < warmup_steps_int, tf.float32)

            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

                        

        return learning_rate





num_train_steps = int(FLAGS.epochs * FLAGS.num_train_examples / FLAGS.train_batch_size)

print(f'num_train_steps: {num_train_steps}')





# ----------------------------------------------------------------------------------------



with tpu_strategy.scope():



    # Model

    bert_tokenizer, bert_nq = get_pretrained_model(FLAGS.model_name)



    # Metric

    train_loss, train_loss_short_start_pos, train_loss_short_end_pos, train_loss_short_ans_type, train_acc, train_acc_short_start_pos, train_acc_short_end_pos, train_acc_short_ans_type = get_metrics("train")



    # Loss

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)





    def loss_function(nq_labels, nq_logits):

        (short_start_pos_labels, short_end_pos_labels, short_answer_type_labels) = nq_labels

        (short_start_pos_logits, short_end_pos_logits, short_answer_type_logits) = nq_logits



        loss_short_start_pos = loss_object(short_start_pos_labels, short_start_pos_logits)

        loss_short_end_pos = loss_object(short_end_pos_labels, short_end_pos_logits)

        loss_short_ans_type = loss_object(short_answer_type_labels, short_answer_type_logits)



        loss_short_start_pos = tf.nn.compute_average_loss(loss_short_start_pos, global_batch_size=FLAGS.train_batch_size)

        loss_short_end_pos = tf.nn.compute_average_loss(loss_short_end_pos, global_batch_size=FLAGS.train_batch_size)

        loss_short_ans_type = tf.nn.compute_average_loss(loss_short_ans_type, global_batch_size=FLAGS.train_batch_size)



        loss = (loss_short_start_pos + loss_short_end_pos + loss_short_ans_type) / 3.0



        return loss, loss_short_start_pos, loss_short_end_pos, loss_short_ans_type





    learning_rate = CustomSchedule(

        initial_learning_rate=FLAGS.init_learning_rate,

        decay_steps=num_train_steps,

        end_learning_rate=FLAGS.init_learning_rate,

        power=1.0,

        cycle=FLAGS.cyclic_learning_rate,

        num_warmup_steps=FLAGS.num_warmup_steps

    )



    decay_var_list = []

    for i in range(len(bert_nq.trainable_variables)):

        name = bert_nq.trainable_variables[i].name

        if any(x in name for x in ["LayerNorm", "layer_norm", "bias"]):

            decay_var_list.append(name)



    optimizer = AdamW(weight_decay=FLAGS.init_weight_decay_rate, learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay_var_list=decay_var_list)



    input_signature = [

        tf.TensorSpec(shape=(None, None), dtype=tf.int32),

        tf.TensorSpec(shape=(None, None), dtype=tf.int32),

        tf.TensorSpec(shape=(None, None), dtype=tf.int32),

        tf.TensorSpec(shape=(None,), dtype=tf.int32),

        tf.TensorSpec(shape=(None,), dtype=tf.int32),

        tf.TensorSpec(shape=(None,), dtype=tf.int32)

    ]





    @tf.function(input_signature=input_signature)

    def train_step(input_ids, input_masks, segment_ids, short_start_pos_labels, short_end_pos_labels, short_answer_type_labels):



        nq_inputs = (input_ids, input_masks, segment_ids)

        nq_labels = (short_start_pos_labels, short_end_pos_labels, short_answer_type_labels)



        with tf.GradientTape() as tape:



            nq_logits = bert_nq(nq_inputs, training=True)

            loss, loss_short_start_pos, loss_short_end_pos, loss_short_ans_type = loss_function(nq_labels, nq_logits)



        gradients = tape.gradient(loss, bert_nq.trainable_variables)



        (short_start_pos_logits, short_end_pos_logits, short_answer_type_logits) = nq_logits



        train_acc.update_state(short_start_pos_labels, short_start_pos_logits)

        train_acc.update_state(short_end_pos_labels, short_end_pos_logits)

        train_acc.update_state(short_answer_type_labels, short_answer_type_logits)



        train_acc_short_start_pos.update_state(short_start_pos_labels, short_start_pos_logits)

        train_acc_short_end_pos.update_state(short_end_pos_labels, short_end_pos_logits)

        train_acc_short_ans_type.update_state(short_answer_type_labels, short_answer_type_logits)



        optimizer.apply_gradients(zip(gradients, bert_nq.trainable_variables))



        train_loss(loss)



        train_loss_short_start_pos(loss_short_start_pos)

        train_loss_short_end_pos(loss_short_end_pos)

        train_loss_short_ans_type(loss_short_ans_type)





    # `experimental_run_v2` replicates the provided computation and runs it with the distributed input.

    @tf.function

    def distributed_train_step(dataset_inputs):



        features, targets = dataset_inputs

        (input_ids, input_masks, segment_ids) = (features['input_ids'], features['input_mask'], features['segment_ids'])

        (short_start_pos_labels, short_end_pos_labels, short_answer_type_labels) = (targets['short_start_positions'], targets['short_end_positions'], targets['short_answer_types'])



        tpu_strategy.experimental_run_v2(train_step, args=(input_ids, input_masks, segment_ids, short_start_pos_labels, short_end_pos_labels, short_answer_type_labels))



checkpoint_path = FLAGS.input_checkpoint_dir + FLAGS.model_name + "/"

ckpt = tf.train.Checkpoint(model=bert_nq)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10000)



# if a checkpoint exists, restore the latest checkpoint.

if ckpt_manager.latest_checkpoint:

    ckpt.restore(ckpt_manager.latest_checkpoint)

    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])

    print (f'Latest BertNQ checkpoint restored -- Model trained for {last_epoch} epochs')

else:

    print('Checkpoint not found. Train BertNQ from scratch')

    last_epoch = 0





train_start_time = datetime.datetime.now()



epochs = FLAGS.epochs

for epoch in range(epochs):



    print("Epoch = {}".format(epoch))



    train_dataset = get_dataset(

        FLAGS.train_tf_record,

        FLAGS.max_seq_length_for_training,

        FLAGS.train_batch_size,

        FLAGS.shuffle_buffer_size,

        is_training=True

    )



    print("train_dataset is OK.")



    train_dist_dataset = tpu_strategy.experimental_distribute_dataset(train_dataset)



    print("train_dist_dataset is OK.")



    train_loss.reset_states()

    

    train_loss_short_start_pos.reset_states()

    train_loss_short_end_pos.reset_states()

    train_loss_short_ans_type.reset_states()           

    

    train_acc.reset_states()

    

    train_acc_short_start_pos.reset_states()

    train_acc_short_end_pos.reset_states()

    train_acc_short_ans_type.reset_states()

    

    epoch_start_time = datetime.datetime.now()



    print("start iterating over train_dist_dataset ...")



    for (batch_idx, dataset_inputs) in enumerate(train_dist_dataset):



        batch_start_time = datetime.datetime.now()



        distributed_train_step(dataset_inputs)



        batch_end_time = datetime.datetime.now()

        batch_elapsed_time = (batch_end_time - batch_start_time).total_seconds()

        

        if (batch_idx + 1) % 100 == 0:

            print('Epoch {} | Batch {} | Elapsed Time {}'.format(

                epoch + 1,

                batch_idx + 1,

                batch_elapsed_time

            ))

            print('Loss {:.6f} | Loss_SS {:.6f} | Loss_SE {:.6f} | Loss_ST {:.6f}'.format(

                train_loss.result(),

                train_loss_short_start_pos.result(),

                train_loss_short_end_pos.result(),

                train_loss_short_ans_type.result()

            ))

            print(' Acc {:.6f} |  Acc_SS {:.6f} |  Acc_SE {:.6f} |  Acc_ST {:.6f}'.format(

                train_acc.result(),

                train_acc_short_start_pos.result(),

                train_acc_short_end_pos.result(),

                train_acc_short_ans_type.result()                

            ))

            print("-" * 100)



    epoch_end_time = datetime.datetime.now()

    epoch_elapsed_time = (epoch_end_time - epoch_start_time).total_seconds()

            

    if (epoch + 1) % 1 == 0:

        

        ckpt_save_path = ckpt_manager.save()

        print ('\nSaving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

        

        print('\nEpoch {}'.format(epoch + 1))

        print('Loss {:.6f} | Loss_SS {:.6f} | Loss_SE {:.6f} | Loss_ST {:.6f}'.format(

                train_loss.result(),

                train_loss_short_start_pos.result(),

                train_loss_short_end_pos.result(),

                train_loss_short_ans_type.result()

        ))

        print(' Acc {:.6f} |  Acc_SS {:.6f} |  Acc_SE {:.6f} |  Acc_ST {:.6f}'.format(

                train_acc.result(),

                train_acc_short_start_pos.result(),

                train_acc_short_end_pos.result(),

                train_acc_short_ans_type.result() 

        ))



    print('\nTime taken for 1 epoch: {} secs\n'.format(epoch_elapsed_time))

    print("-" * 80 + "\n")
