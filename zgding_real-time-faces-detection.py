import numpy as np

import os

print(os.listdir("../input"))

print(os.listdir("../input/train-script/facedct/facedct/"))





#添加python搜索目录

import sys

run_path = "../input/train-script/facedct/facedct/"

sys.path.append(run_path)

#import face_generator



#dataset

print(os.listdir("../input/training/facedct/facedct/dataset/"))#"/images/"



#预训练模型

print(os.listdir("../input/training/facedct/facedct/base_models/" ))

#import warnings

#warnings.filterwarnings("ignore")

#加载依赖库

from keras.optimizers import Adam, SGD, Nadam

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler

from keras.callbacks import Callback

from keras import backend as K 

from keras.models import load_model

from math import ceil 

import numpy as np 

from termcolor import colored

#当前执行目录下面的库

from mn_model import mn_model

from face_generator import BatchGenerator

from keras_ssd_loss import SSDLoss

from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2



import scipy.misc as sm

print("ok")

#定义超参

img_height = 512

img_width = 512

img_channels = 3



n_classes = 2 

class_names = ["background","face"]



scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # anchorboxes for coco dataset

aspect_ratios = [[0.5, 1.0, 2.0],

                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],

                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],

                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],

                 [0.5, 1.0, 2.0],

                 [0.5, 1.0, 2.0]] # The anchor box aspect ratios used in the original SSD300

two_boxes_for_ar1 = True

limit_boxes = True # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries

variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation

coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation

normalize_coords = True

print("ok")
### 模型定义和加载预训练模型  ###



K.clear_session()

#模型定义

model, model_layer, img_input, predictor_sizes = mn_model(image_size=(img_height, img_width, img_channels), 

                                                                      n_classes = n_classes,

                                                                      min_scale = None, 

                                                                      max_scale = None, 

                                                                      scales = scales, 

                                                                      aspect_ratios_global = None, 

                                                                      aspect_ratios_per_layer = aspect_ratios, 

                                                                      two_boxes_for_ar1= two_boxes_for_ar1, 

                                                                      limit_boxes=limit_boxes, 

                                                                      variances= variances, 

                                                                      coords=coords, 

                                                                      normalize_coords=normalize_coords)

#Freeze layers冻结分类的层，不训练，只用来提取特征

print ("Freezing classification layers")

for layer_key in model_layer:

  if('detection'  not in layer_key): #prefix detection to freeze layers which does not have detection

    model_layer[layer_key].trainable = False

print (colored("classification layers freezed", 'green'))



#加载预训练模型

print ("loading classification weights")

classification_model = run_path + './base_models/mobilenet_1_0_224_tf.h5'#预训练模型路径

model.load_weights(classification_model,  by_name= True)

print (colored( ('classification weights %s loaded' % classification_model), 'green'))

print ("ok")

#########  喂数据  #######



#小批量训练数据，用来调试

train_data = run_path + 'wider_train_small.npy'

test_data = run_path + 'wider_val_small.npy'

data_path = '../input/training/facedct/facedct/dataset/'



'''

#全量数据

train_data = run_path + 'wider_train_full.npy'

test_data = run_path + 'wider_val_full.npy'

data_path = '../input/training/facedct/facedct/'

'''



batch_size = 64

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,

                                img_width=img_width,

                                n_classes=n_classes, 

                                predictor_sizes=predictor_sizes,

                                min_scale=None,

                                max_scale=None,

                                scales=scales,

                                aspect_ratios_global=None,

                                aspect_ratios_per_layer=aspect_ratios,

                                two_boxes_for_ar1=two_boxes_for_ar1,

                                limit_boxes=limit_boxes,

                                variances=variances,

                                pos_iou_threshold=0.5,

                                neg_iou_threshold=0.2,

                                coords=coords,

                                normalize_coords=normalize_coords)



train_dataset = BatchGenerator(images_path=train_data, 

                               #image_set_path=data_path,

                               include_classes='all', 

                               box_output_format = ['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])



print ("TRAINING DATA")



train_dataset.parse_xml(

                  annotations_path=train_data,

                  #image_set_path='None',

                  image_set_path=data_path,

                  image_set='None',

                  classes = class_names, 

                  exclude_truncated=False,

                  exclude_difficult=False,

                  ret=False, 

                  debug = False)



train_generator = train_dataset.generate(

                 batch_size=batch_size,

                 train=True,

                 ssd_box_encoder=ssd_box_encoder,

                 equalize=True,

                 brightness=(0.5,2,0.5),

                 flip=0.5,

                 translate=((0, 20), (0, 30), 0.5),

                 scale=(0.75, 1.2, 0.5),

                 crop=False,

                 #random_crop = (img_height,img_width,1,3), 

                 random_crop=False,

                 resize=(img_height, img_width),

                 #resize=False,

                 gray=False,

                 limit_boxes=True,

                 include_thresh=0.4,

                 diagnostics=False)



n_train_samples = train_dataset.get_n_samples()



print ("Total number of training samples = {}".format(n_train_samples))





print ("VALIDATION DATA")

val_dataset = BatchGenerator(images_path=test_data, include_classes='all', 

                box_output_format = ['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])





val_dataset.parse_xml(

                  annotations_path=test_data,

                  image_set_path=data_path,

                  image_set='None',

                  classes = class_names, 

                  exclude_truncated=False,

                  exclude_difficult=False,

                  ret=False, 

                  debug = False)





val_generator = val_dataset.generate(

                 batch_size=batch_size,

                 train=True,

                 ssd_box_encoder=ssd_box_encoder,

                 equalize=False,

                 brightness=False,

                 flip=False,

                 translate=False,

                 scale=False,

                 crop=False,

                 #random_crop = (img_height,img_width,1,3), 

                 random_crop=False, 

                 resize=(img_height, img_width), 

                 #resize=False, 

                 gray=False,

                 limit_boxes=True,

                 include_thresh=0.4,

                 diagnostics=False)



n_val_samples = val_dataset.get_n_samples()



print ("Total number of validation samples = {}".format(n_val_samples))



print ("ok")
########### 启动训练 ############

det_model_path = "./"

print("输出文件夹：",os.listdir(det_model_path))



#Adam 优化器

base_lr = 0.002

adam = Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay = 0.0)

ssd_loss = SSDLoss(neg_pos_ratio=2, n_neg_min=0, alpha=1.0, beta = 1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)



def scheduler(epoch):

  if epoch%10==0 and epoch!=0:

    lr = K.get_value(model.optimizer.lr)

    K.set_value(model.optimizer.lr, lr*.95)

    print("lr changed to {}".format(lr*.95))

  else: 

    print("lr remains {}".format(K.get_value(model.optimizer.lr)))



  return K.get_value(model.optimizer.lr)

lr_schedule = LearningRateScheduler(scheduler)



#训练

num_epochs = 5 #20

plateau = ReduceLROnPlateau(monitor='val_loss', factor = 0.3, patience =4, epsilon=0.001, cooldown=0)

tensorboard = TensorBoard(log_dir='./logs/trial1/', histogram_freq=1, batch_size=16, write_graph=True, write_grads=True, 

                          write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100)

model_checkpoint =  ModelCheckpoint(det_model_path + 'ssd_mobilenet_face_epoch_{epoch:02d}_loss{val_loss:.4f}.h5',

                                                           monitor='val_loss',

                                                           verbose=1,

                                                           save_best_only=True,

                                                           save_weights_only=True,

                                                           mode='auto',

                                                           period=1)





history = model.fit_generator(generator = train_generator,

                              steps_per_epoch = ceil(n_train_samples/batch_size)*2,

                              epochs = num_epochs,

                              callbacks = [model_checkpoint, lr_schedule, early_stopping],                      

                              validation_data = val_generator,

                              validation_steps = ceil(n_val_samples/batch_size))



#模型保存

model.save_weights(det_model_path + 'ssd_mobilenet_weights_epoch_{}.h5'.format(num_epochs))



print ("model and weight files saved at : " + det_model_path)