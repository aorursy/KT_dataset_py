import tensorflow as tf
import pandas as pd
import random
import time
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.framework import arg_scope
import sys
tf.reset_default_graph()

DATASET_PATH      = "../input/diabetes.csv"

LOGS_PATH	= "logs/"

NUM_DIM=8
NUM_LABELS		= 2

TRAIN_BATCH_SIZE	= 0
VALID_BATCH_SIZE	= 0
TEST_BATCH_SIZE		= 0
#epoch 
MAX_EPOCH=70

def read_label_file(file='./diabetes.csv',train_split_ratio=0.6,valid_split_ratio=0.2,test_split_ratio=0.2):
	df = pd.read_csv(file, encoding='latin-1')
	label=df.Outcome.tolist()
	contents=[]
	for row in df.iterrows():
		index, data = row
		contents.append(data[0:8].tolist())

	size=len(label)

	sum_ratio=train_split_ratio+valid_split_ratio+test_split_ratio
	
    
	train_size=int(size*train_split_ratio)
	train_split_range=train_size
	train_contents=contents[0:train_split_range]
	train_label=label[0:train_split_range]
	
	valid_size=int(size*valid_split_ratio)
	valid_split_range=valid_size+train_split_range
	valid_contents=contents[train_split_range:valid_split_range]
	valid_label=label[train_split_range:valid_split_range]
	
	test_size=int(size*test_split_ratio)
	test_split_range=test_size+valid_split_range
	test_contents=contents[valid_split_range:test_split_range]
	test_label=label[valid_split_range:test_split_range]
	
	
	return train_size,valid_size,test_size,train_contents,train_label,valid_contents,valid_label,test_contents,test_label

def preprocess(data,label,num_class):


	contents=data
	

	label = tf.one_hot(label,depth=num_class,on_value=1,off_value=0,axis=-1)
	return contents,label
		
def create_queue(contents,labels):


	input_queue = tf.train.slice_input_producer(
												[contents, labels],
												shuffle=True)

	content = input_queue[0]
	label = input_queue[1]
	
	return content,label



total_train_size,total_valid_size,total_test_size,train_contents,train_labels,valid_contents,valid_labels,test_contents,test_labels = read_label_file(DATASET_PATH,0.6,0.2,0.2)

if(TRAIN_BATCH_SIZE==0):
    TRAIN_BATCH_SIZE=(int)(total_train_size*0.1)
if(VALID_BATCH_SIZE==0):
    VALID_BATCH_SIZE=(int)(total_valid_size*0.1) 
if(TEST_BATCH_SIZE==0):
    TEST_BATCH_SIZE=(int)(total_test_size*0.1) 


train_contents,train_labels=preprocess(train_contents,train_labels,NUM_LABELS)
valid_contents,valid_labels=preprocess(valid_contents,valid_labels,NUM_LABELS)
test_contents,test_labels=preprocess(test_contents,test_labels,NUM_LABELS)




train_content, train_label=create_queue(train_contents,train_labels)
valid_content, valid_label=create_queue(valid_contents,valid_labels)
test_content, test_label=create_queue(test_contents,test_labels)





train_batch = tf.train.batch(
                        [train_content, train_label],
                        batch_size=TRAIN_BATCH_SIZE
                    )

valid_batch = tf.train.batch(
                        [valid_content, valid_label],
                        batch_size=VALID_BATCH_SIZE
                    )		

test_batch = tf.train.batch(
                        [test_content,test_label],
                        batch_size=TEST_BATCH_SIZE
                    )			
print ("input pipeline ready")

def Batch_Norm(x, training, scope="bn"):
    with arg_scope([batch_norm],
                    scope=scope,
                    updates_collections=None,
                    decay=0.9,
                    center=True,
                    scale=True,
                    zero_debias_moving_mean=True):
        return tf.cond(training,
                        lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                        lambda : batch_norm(inputs=x, is_training=training, reuse=True))
                       
#BR layer(= ReLU + Batch Norm)
def BR_Layer(x,name,output_num,training):
    xavier_initializer = tf.contrib.layers.xavier_initializer()
    shape = x.get_shape().as_list()
    input_num= shape[1]
    W = tf.Variable(xavier_initializer([input_num, output_num]))
    b = tf.Variable(xavier_initializer([output_num]))
    _x = tf.matmul(x, W) + b
    _x = Batch_Norm(_x,training, scope="bn_"+name)
    _x = tf.nn.relu(_x)
    return _x



# placeholder is used for feeding data.
x = tf.placeholder(tf.float32, shape=[None, NUM_DIM], name = 'x') # none represents variable length of dimension. 784 is the dimension of MNIST data.
y_target = tf.placeholder(tf.float32, shape=[None, NUM_LABELS], name = 'y_target') # shape argument is optional, but this is useful to debug.
training=tf.placeholder(tf.bool)

xavier_initializer = tf.contrib.layers.xavier_initializer()

# reshape input data
_x = tf.reshape(x,[-1,NUM_DIM],name="x_data")

#create a model
_x=BR_Layer(_x,"layer1",32,training)
_x=BR_Layer(_x,"layer2",16,training)
_x=BR_Layer(_x,"layer3",32,training)

shape = _x.get_shape().as_list()
input_num= shape[1]                       
W = tf.Variable(xavier_initializer([input_num, NUM_LABELS]))
b = tf.Variable(xavier_initializer([NUM_LABELS]))
_x = tf.matmul(_x, W) + b
                       
pred=_x
prob_y=tf.nn.softmax(pred, name="prob_y")

# define the Loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_target))

# define optimization algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)


# correct_prediction is list of boolean which is the result of comparing(model prediction , data)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_target, 1))

#define accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

#save weight
t_vars = tf.trainable_variables()
saver = tf.train.Saver(max_to_keep=None,var_list=t_vars)
saver_def = saver.as_saver_def()
#gpu config
configure=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True))

sess = tf.Session(config=configure) 

#define a tensorboard
training_loss=tf.summary.scalar('training_loss', cost)
validation_loss=tf.summary.scalar('validation_loss', cost)

training_accuracy = tf.summary.scalar("training_accuracy", accuracy)
validation_accuracy = tf.summary.scalar("validation_accuracy", accuracy)


# Merge all summaries into a single op
merged_op = tf.summary.merge_all()
writer=tf.summary.FileWriter(LOGS_PATH, sess.graph)

# initialization
init_op = tf.global_variables_initializer()
print("Session started!")
start_session_time = time.time()
sess.run(init_op)


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
max_epoch=MAX_EPOCH
display_step=1
max_accuracy=0
max_index=0
#train 
for epoch in range(0,max_epoch+1):
    avg_cost = 0
    total_batch=(int)(total_train_size/TRAIN_BATCH_SIZE)
    for step in range(total_batch):
        _train_batch=sess.run([tf.cast(train_batch[0], tf.float32),tf.cast(train_batch[1], tf.float32)])
        if epoch!=0:
            _,c=sess.run([train_step,cost] , feed_dict={x: _train_batch[0], y_target: _train_batch[1], training: True})
            avg_cost += c / total_batch
        else:
            c=sess.run(cost , feed_dict={x: _train_batch[0], y_target: _train_batch[1], training: False})
            avg_cost += c / total_batch

    if epoch % display_step == 0:

        print ("Epoch:", '%04d' % (epoch), "cost=","{:.9f}".format(avg_cost))

        _valid_batch=sess.run([tf.cast(valid_batch[0], tf.float32),tf.cast(valid_batch[1], tf.float32)])

        # traininig accuracy
        train_cos,train_acc, train_summ_acc,train_summ_loss = sess.run(
        [cost, accuracy, training_accuracy,training_loss], 
        feed_dict={x : _train_batch[0],  y_target : _train_batch[1],training: False})
        writer.add_summary(train_summ_acc, epoch) 
        writer.add_summary(train_summ_loss, epoch) 


        # validation accuracy
        valid_cos, valid_acc, valid_summ_acc,valid_summ_loss  = sess.run(
        [cost,accuracy, validation_accuracy,validation_loss],
        feed_dict={x: _valid_batch[0], y_target: _valid_batch[1],training: False})
        writer.add_summary(valid_summ_acc, epoch)
        writer.add_summary(valid_summ_loss, epoch)
        
        #save weight
        saver.save(sess,"./weight/w",epoch)
        print("Train-Accuracy:", train_acc,"Train-Loss:", train_cos, "Validation-Accuracy:", valid_acc,"Val-Loss:", valid_cos,"\n")
        #model selection = vaildation accuracy
        if valid_acc > max_accuracy:
            max_accuracy=valid_acc
            max_index=epoch


#test accuracy
saver.restore(sess,"./weight/w-%d"%(max_index))
_test_batch=sess.run([tf.cast(test_batch[0],tf.float32),tf.cast(test_batch[1], tf.float32)])
_accuracy=sess.run(accuracy, feed_dict={x: _test_batch[0], y_target:  _test_batch[1],training: False})
print("(result)test accuracy: %g / weight-%d"%(_accuracy,max_index))    

# close thread qeueue, writer, session
coord.request_stop()
coord.join(threads)
writer.close()
sess.close()

print("close")



