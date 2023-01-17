# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import tensorflow as tf

# Any results you write to the current directory are saved as output.
#find landmarks, so determine unique number of classes
train_data= pd.read_csv('../input/train.csv')
test_data= pd.read_csv('../input/test.csv')
submission = pd.read_csv("../input/sample_submission.csv")

unique_landmark = pd.DataFrame(train_data.landmark_id.value_counts())
# print(len(unique_landmark))
#print(unique_landmark)


unique_landmark.reset_index(inplace=True)
unique_landmark.columns = ['landmark_id','count']

clsses_number= len(unique_landmark) #gives number of clases
classes=[unique_landmark.landmark_id]

print(classes)
# land_count_data= pd.DataFrame()

image_size=128
channels= 3 # color image

import os, errno
import shutil
dir_path = "/images"
train_files="/train"
full_image_path="/images/train/dummy"
direct_path=dir_path+train_files
directory = os.path.dirname(os.getcwd()+full_image_path) # /kaggle/working/images/train
#print(os.listdir(directory))

#print(os.listdir(os.getcwd()+direct_path))
# try:
#     shutil.rmtree("image")
# except OSError as e:
#     print ("Error: %s - %s." % (e.filename,e.strerror))
    
try:
   if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created!")
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Some unwanted error!")

# f = open(os.path.join(directory, 'file.txt'), 'w')
# f.write('This is the new file.')
# f.close()

# print(os.listdir('/'))
print(directory)
print(os.listdir(directory))
# with open(os.path.join(directory, 'file.txt'), 'r') as ins:
#     array = []
#     for line in ins:
#         print(line)
import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
print(directory)
print(os.listdir(directory))

def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    print(key_url_list[1:])
    return key_url_list[1:]  # Chop off header


def download_image(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1
    
    return 0


def loader(file_path, ouput_dir):
#     if len(sys.argv) != 3:
#         print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
#         sys.exit(0)
    (data_file, out_dir) = (file_path,ouput_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print("Cominggggggggggggggg")
    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=20)  # Num of CPUs
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list), total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()

loader('../input/train.csv',directory)
# # import wget
# url = 'https://i1.wp.com/python3.codes/wp-content/uploads/2015/06/Python3-powered.png?fit=650%2C350'  
# # wget.download(url, 'images/train/image_1.jpg')
# post_payload = { "event": { "Title": "Something, sometime, something, Python"} }
# post_headers = {'Content-Type': 'application/xml'}
# urlretrieve(url=url,filename= "images/train/image_1.jpg",data=post_payload)
# #urlretrieve(url,, headers)
# from urllib.request import urlretrieve

# import csv

# id_tr=[]
# url_tr=[]
# land_tr=[]

# with open('../input/train.csv') as fileOpen:
#     data = csv.reader(fileOpen)
#     next(data) #bypassinh header in csv
#     img_count = 0  # start at 1
    
# #     with open(os.path.join(dir_path, 'image_train.csv'), 'a+b') as csvfile:
# #         filewriter = csv.writer(csvfile, delimiter=',',
# #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for row in data:
#     #         print(row[1])
#             try:
#                 img_count += 1
#                 print(os.path.join(directory, 'image_{0}.jpg').format(img_count))
#                 urlretrieve(row[1],
#                         "images/train/image_1.jpg")
#             except Exception as e :    
#                 print(e)
#                 print(row[1])
#                 continue  # continue to next row
                
# #             filewriter.writerow(row[0], row[1], row[2])
#             id_tr.append(row[0]) 
#             url_tr.append(row[1]) 
#             land_tr.append(row[2]) 
#         #         print(row[1])
        
# data_to_submit = pd.DataFrame({
#     'id':id_tr,
#     'url':url_tr,
#     'landmark_id':land_tr
# })
# data_to_submit.to_csv(os.path.join(dir_path, 'image_train.csv'),header=None,index=False)

# filter_size_conv1 = 3 
# num_filters_conv1 = 32

# filter_size_conv2 = 3
# num_filters_conv2 = 32

# filter_size_conv3 = 3
# num_filters_conv3 = 64
    
# fc_layer_size = 128

# x= tf.placeholder(tf.float32, shape=[None, img_size,image_size,channels], name='x')

# y= tf.placeholder(tf.float32, shape=[None, clsses_number], name='y')



# weight1=tf.Variable(tf.truncated_normal([filter_size_conv1, filter_size_conv1, channels,  num_filters_conv1],stddev=0.02 ))

# weight2=tf.Variable(tf.truncated_normal([filter_size_conv2, filter_size_conv2, num_filters_conv1,  num_filters_conv2],stddev=0.02 ))

# weight3=tf.Variable(tf.truncated_normal([filter_size_conv3, filter_size_conv3, num_filters_conv2,  num_filters_conv3],stddev=0.02 ))


# bias1=tf.Variable(tf.constant(0.05, shape=[num_filters_conv1]))

# bias2=tf.Variable(tf.constant(0.05, shape=[num_filters_conv2]))

# bias3=tf.Variable(tf.constant(0.05, shape=[num_filters_conv3]))


# #FC layer weghts and biases



# def convolutional_operations(inputs,weight, bias):
    
#     layer= tf.nn.conv2d(input= inputs, filter=weight, strides=[1,1,1,1], padding='SAME')
#     layer = tf.add(layer, bias)
    
#     layer= tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
#     activation_layer= tf.nn.relu(layer)
    
#     return activation_layer

# weight_fc1=tf.Variable(tf.truncated_normal([],stddev=0.02 ))
#     weight_fc2=
    

# def fc_operations(inputs, weights, bias):
#     layer = tf.add(tf.matmul(input, weights) , biases)
#     activation_layer= tf.nn.relu(layer)
    
#     return activation_layer


# conv_layer1=convolutional_operations(inputs=x ,weight=weight1 , bias=bias1)

# conv_layer2=convolutional_operations(inputs=conv_layer1 ,weight=weight2 , bias=bias2)

# conv_layer3=convolutional_operations(inputs=conv_layer2 ,weight=weight3 , bias=bias3)

# #calcuate fc layer shape, same but get it from conv_layer3

# layer_resize= conv_layer3,[-1,].get_shape()
# features_size= layer_resize[1:4].num_elements()
# fc_layer=tf.reshape(conv_layer3,[-1,features_size]) # or tf.reshape(conv_layer3,[-1,3,3,filter_size_conv3])
    
# weghts_fc1=tf.Variable(tf.truncated_normal([ fc_layer.get_shape()[1:4].num_elements(),  fc_layer_size],stddev=0.02 ))
# bias_fc1=tf.Variable(tf.constant(0.05, shape=[fc_layer_size]))
# fc_layer1= fc_operations(inputs=fc_layer, weights=weghts_fc1, bias=bias_fc1 )

# weghts_fc2=tf.Variable(tf.truncated_normal([  fc_layer_size, clsses_number],stddev=0.02 ))
# bias_fc2=tf.Variable(tf.constant(0.05, shape=[clsses_number]))
# fc_layer2= fc_operations(inputs=fc_layer1, weights=weghts_fc2, bias=bias_fc2 )


# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer2, labels=y)
# cost= tf.reduce_mean(cross_entropy)

# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# y_predict = tf.argmax(fc_layer2, dimension=1) #predict class like: 0 0 0 1 0 0 0
# y_class = tf.argmax(y, dimension=1) #y class like: 0 0 0 1 0 0 0
# correctly_predicted = tf.equal(y_predict, y_class)

# accuracy= tf.reduce_mean(tf.cast(correctly_predicted ,float32))

# session= tf.Session()
# session.run(tf.global_variables_initializer())
# epoch=100

# with session as sess:
    
#     for i in range(epoch):
#         sess.run([optimizer,cost],feed_dcit={x:, y: } )
        
#         if i % 10 = 0:
            
#             accuracy = sess.run([accuracy], feed_dcit={x:test, y:test })
#             print("accuracy {}"%(accuracy))
            

            
            
        
    

