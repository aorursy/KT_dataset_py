#Aruchomu's original notebook is written in tensorflow v1

!pip install gast==0.2.2

!pip uninstall -y tensorflow

!pip install tensorflow-gpu==1.14
import numpy as np, pandas as pd, tensorflow as tf #For dataframe processing and Deep Learning

from PIL import Image, ImageDraw, ImageFont        #'Pillow' library for image processing

from IPython.display import display                #To display images in notebook

from seaborn import color_palette                  #For bounding boxes colors

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass

print(tf.__version__)
''' DEFAULT YOLO HYPERPARAMS '''

_BATCH_NORM_DECAY=0.9           #Momentum (Beta) value to compute weighted moving average

_BATCH_NORM_EPSILON=1e-5        #Added in denominator to handle case of divide by 0

#moving_avg= beta*moving_avg + (1-beta)*current_avg



_LEAKY_RELU=0.1                 #Leakage (Alpha) value to calculate minimum instead of 0 as in regular RELU function

#a=max(alpha*z,z)



_ANCHORS=[(10,13),(16,30),(33,23),

          (30,61),(62,45),(59,119),

          (116,90),(156,198),(373,326)]

                                #3 pairs of (anchor1, anchor2, anchor3)

                                #total 9 anchors each of (anchorW, anchorH) values that'll be multiplied with exponent of bh, bw predicted values

    

_MODEL_SIZE=(416,416)           #Input size of model
def batch_norm(inputs,trainable,data_format):

    return tf.layers.batch_normalization(

        inputs=inputs, axis=1 if data_format=='channels_first' else 3,

        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,

        scale=True, training=trainable

    )



def fixed_padding(inputs, kernel_size, data_format):

    ''' 

        RESNET IMPLEMENTATION ON TENSORFLOW ALSO MENTIONS THIS

        RETURNS A TENSOR WITH SAME FORMAT AS INPUT, BUT WITH PADDING

    '''

    pad_total = kernel_size - 1

    pad_beg = pad_total // 2

    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':

        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],

                                        [pad_beg, pad_end],

                                        [pad_beg, pad_end]])

    else:

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],

                                        [pad_beg, pad_end], [0, 0]])

    return padded_inputs





def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides=1):

    if strides > 1:

        inputs = fixed_padding(inputs, kernel_size, data_format)



    return tf.layers.conv2d(

        inputs=inputs, filters=filters, kernel_size=kernel_size,

        strides=strides, padding=('SAME' if strides == 1 else 'VALID'),

        use_bias=False, data_format=data_format)







def darknet53_residual_block(inputs, filters, trainable, data_format, strides=1):

    ''' 

        input => [ CONV2D --> BN->LRELU --> CONV2D --> BN->LRELU ] => y+input 

    '''

    residualConnection_input=inputs

    

    inputs=conv2d_fixed_padding(inputs,filters,1,data_format,strides)

    inputs=batch_norm(inputs,trainable,data_format)    

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    

    inputs=conv2d_fixed_padding(inputs,filters*2,3,data_format,strides)

    inputs=batch_norm(inputs,trainable,data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    return tf.add(inputs,residualConnection_input)
''' TESTING INDIVIDUAL COMPONENT FUNCTIONS 

tf.reset_default_graph()

x=tf.placeholder(shape=(1,64,64,3),dtype=tf.float32)



res=batch_norm(inputs=tf.to_float(x),trainable=False,data_format='channels_last')

print(res,"\n") #1,7,7,3



res=conv2d_fixed_padding(inputs=res,filters=10,kernel_size=3,data_format='channels_last')

print(res.shape) #1,7,7,10





res=darknet53_residual_block(inputs=res,filters=5,trainable=False,data_format='channels_last')

print(res.shape)



r1,r2,res2=darknet53(inputs=res,trainable=False,data_format='channels_last')

print("Without Eval res2=",res2)





ip=np.random.randn(1,64,64,3)

print('ip.shape=',ip.shape)



with tf.Session() as s:

    s.run(tf.global_variables_initializer())

    s.run(res, feed_dict={x:ip})

    print("After Eval res2=",res)

    '''
def darknet53(inputs,trainable,data_format):

    '''

    C=Normal Conv block, R=Residual Block, layer xNum=Num occurences of layer

    

    C -> C -> R -> C -> R x2 -> C -> R x8 -> C -> R x8 -> C -> R x4

    

    '''

    

    inputs=conv2d_fixed_padding(inputs,filters=32, kernel_size=3,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    #C

    

    inputs=conv2d_fixed_padding(inputs,filters=32*2, kernel_size=3,strides=2, data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    #C

    

    inputs=darknet53_residual_block(inputs,filters=32,trainable=trainable,data_format=data_format)

    #R

    

    inputs=conv2d_fixed_padding(inputs,filters=32*2*2,kernel_size=3,strides=2,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    #C

    

    for _ in range(2):

        inputs=darknet53_residual_block(inputs,filters=32*2, trainable=trainable,data_format=data_format)

        #R x2

        

    inputs=conv2d_fixed_padding(inputs,filters=32*2*2*2,kernel_size=3,strides=2,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    #C

    

    for _ in range(8):

        inputs=darknet53_residual_block(inputs,filters=32*2*2, trainable=trainable, data_format=data_format)

        #R x8

        

    route1=inputs

    

    inputs=conv2d_fixed_padding(inputs,filters=32*2*2*2*2,kernel_size=3,strides=2,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    #C

    

    

    

    for _ in range(8):

        inputs=darknet53_residual_block(inputs,filters=32*2*2*2, trainable=trainable, data_format=data_format)

        #R x8

    route2=inputs

    

    inputs=conv2d_fixed_padding(inputs,filters=32*2*2*2*2*2,kernel_size=3,strides=2,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    #C

    

    for _ in range(4):

        inputs=darknet53_residual_block(inputs,filters=32*2*2*2*2,trainable=trainable,data_format=data_format)

        #R x4

        

    return route1,route2,inputs
def yolo_conv_block(inputs, filters, trainable, data_format):

    inputs=conv2d_fixed_padding(inputs,filters=filters, kernel_size=1,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)



    inputs=conv2d_fixed_padding(inputs,filters=2*filters,kernel_size=3,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    

    inputs=conv2d_fixed_padding(inputs,filters=filters, kernel_size=1,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)



    inputs=conv2d_fixed_padding(inputs,filters=2*filters,kernel_size=3,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    

    inputs=conv2d_fixed_padding(inputs,filters=filters, kernel_size=1,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)



    route=inputs

    

    inputs=conv2d_fixed_padding(inputs,filters=2*filters,kernel_size=3,data_format=data_format)

    inputs=batch_norm(inputs,trainable=trainable,data_format=data_format)

    inputs=tf.nn.leaky_relu(features=inputs, alpha=_LEAKY_RELU)

    

    return route,inputs
def yolo_layer(inputs, n_classes, anchors, image_size, data_format):

    '''

        1)   EACH ANCHOR-BOX PREDICTION NEEDS TO BE OF LENGTH = (5 + numClasses)

        1.1) HENCE SEMI-FINAL FEATURE MAP IS GENERATED OF DIMENSIONS = <m, W, H, numAnchors*(5 + numClasses)>

        2)   SET grid_shape = <W,H>

        3)   inputs NEEDS TO BE OF SHAPE = <m, W, H, numAnchors*(5 + numClasses)>

        4)   inputs ARE RESHAPED TO < -1, (numAnchors*W*H), (5+numClasses) >

        5)   SET strides = int(imgW/W , imgH/H)

    '''

    num_anchors=len(anchors)

    inputs=tf.layers.conv2d(inputs=inputs, filters=num_anchors*(5+n_classes) ,

                            kernel_size=1, strides=1, use_bias=True ,data_format=data_format)

    

    shape=inputs.get_shape().as_list() #get_shape() is used to return dynamic shape that is computed at runtime

    

    grid_shape=shape[2:4] if data_format=='channels_first' else shape[1:3]

    

    if data_format=='channels_first':

        inputs=tf.transpose(inputs,perm=[0,2,3,1])

    

    inputs=tf.reshape(inputs,[ -1, num_anchors*grid_shape[0]*grid_shape[1] , 5+n_classes ])

    

    strides=(image_size[0]//grid_shape[0] , image_size[1]//grid_shape[1])

    

    

    '''

       6)   SPLIT UP inputs TO GIVE (x,y) box_centers COORDINATES, (xmax,ymax) box_shapes COORDINATES, OBJECTNESS confidence SCORE

            AND classes ARRAY OF OBJECT IDENTIFIERS

            box_centers will have shape = <m, numAnchors*H*W, 2>

            box_shapes will have shape =  <m, numAnchors*H*W, 2>

            confidence will have shape =  <m, numAnchors*H*W, 1>

            classes will have shape =     <m, numAnchors*H*W, nClasses>

       7)   SET x = [0,1,2,...W]

            AND y = [0,1,2,...H]

       8)   SET x_offset, y_offset = MESHGRID OF (x,y)

       8.1) x_offset WILL HAVE SHAPE <H, W>

            AND y_offset WILL HAVE SHAPE <H, W>

       9)   RESHAPE x_offset TO HAVE SHAPE = <H*W, 1>

            RESHAPE y_offset TO HAVE SHAPE = <H*W, 1>

       10)  CONCAT x_offset, y_offset TO GET xy_offset: A SERIES OF (x,y) COORDINATES OF SHAPE = <H*W, 2>

       11)  USE tf.tile TO REPEAT THE SERIES OF (x,y) COORDINATES numAnchor TIMES ALONG THE SECOND DIMENSION

            xy_offset IS NOW OF SHAPE = <H*W, 2*3>   assuming anchors are 3

       12)  RESHAPE xy_offset TO BE OF SHAPE = <1, H*W*3, 2>

    '''

    

    

    box_centers, box_shapes, confidence, classes = tf.split(inputs, [2,2,1,n_classes], axis=-1)

    

    

    x=tf.range(grid_shape[0],dtype=tf.float32)  #range of width

    y=tf.range(grid_shape[1],dtype=tf.float32)  #range of height

    

    x_offset,y_offset=tf.meshgrid(x,y)

    x_offset=tf.reshape(x_offset,(-1,1))

    y_offset=tf.reshape(y_offset,(-1,1))

    

    #x_offset, y_offset are flattened tensors

    xy_offset=tf.concat([x_offset,y_offset],axis=-1) #(x,y) coordinate pairs from 0-W and 0-H respectively

    xy_offset=tf.tile(xy_offset,[1,num_anchors])     #used to repeat xy_offset 'n_anchors' times along 2nd dimension

    xy_offset=tf.reshape(xy_offset,[1,-1,2])         #(x,y) coordinate pairs from 0 to W and 0 to H respectively, repeated nAnchor times

    

    

    '''

       13)  SET (x,y) box_centers = sigmoid([x,y])

       14)  SET box_centers = (box_centers + xy_offset) * strides

       15)  USE tf.tile TO REPEAT THE SERIES OF anchors (H*W) TIMES ALONG THE FIRST DIMENSION

       16)  SET box_shapes = e^box_shapes * anchors

            ie: box_width = e^box_width * anchorWidth

                box_height = e^box_height * anchorHeight

       17)  SET confidence = sigmoid (confidence)

       18)  SET classes = sigmoid(classes)

       19)  RETURN [((box_centers + xy_offset) * strides)  ,  e^box_shapes * duplicated(anchors)  ,  sigmoid(confidence)  ,  sigmoid(classes)]

    '''

    

    box_centers=tf.nn.sigmoid(box_centers)

    box_centers=(box_centers+xy_offset)*strides

    

    anchors=tf.tile(anchors,[grid_shape[0]*grid_shape[1],1])

    box_shapes=tf.exp(box_shapes) * tf.to_float(anchors)

    

    confidence=tf.nn.sigmoid(confidence)

    

    classes=tf.nn.sigmoid(classes)

    

    inputs=tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

    

    return inputs
def upsample(inputs,out_shape,data_format):

    if data_format=='channels_first':

        inputs=tf.transpose(inputs,[0,2,3,1])

        new_height=out_shape[3]

        new_width=out_shape[2]

    else:

        new_height=out_shape[2]

        new_width=out_shape[1]

        

    inputs=tf.image.resize_nearest_neighbor(images=inputs, size=(new_height,new_width))

    

    if data_format=='channels_first':

        inputs=tf.transpose(inputs,[0,3,1,2])

    

    return inputs
def build_boxes(inputs):

    print("YOU ARE NOW BUILDING BOXES")

    print(inputs)

    center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1,1,1,1,1,-1],axis=-1)

    print('center_x=', center_x)

    print('center_y=', center_y)

    print('width=', width)

    print('height=', height)

    print('confidence=', confidence)

    print('classes=',classes)

    topleft_x = center_x-(width/2)

    topleft_y = center_y-(height/2)

    bottomright_x=center_x+(width/2)

    bottomright_y=center_y+(height/2)

    

    boxes=tf.concat([topleft_x,topleft_y,bottomright_x,bottomright_y,confidence,classes],axis=-1)

    return boxes







def non_max_suppression(inputs, n_classes, max_output_size, iou_threshold, confidence_threshold):

    '''

    PERFORM NONMAX SUPPRESSION FOR EACH CLASS

        max_output_size: Max number of boxes to be selected for each class

    RETURN A LIST OF class-to-boxes DICTIONARIES FOR EACH SAMPLE IN THE BATCH

    '''

    batch=tf.unstack(inputs)   #for inputs of shape (Batchsize,B,C,D) returns list of unstacked tensors of shape (B,C,D)

    boxes_dicts=[]

    for boxes in batch:

        boxes=tf.boolean_mask(boxes,boxes[:,4]>confidence_threshold) #internally creates a mask of condition, applies it to 'boxes' and returns values which satisfy True condition

        classes=tf.argmax(boxes[:,5:],axis=-1)  #select the position of maximum value in array of classes

        classes=tf.expand_dims(tf.to_float(classes),axis=-1) #add a dimension to the above 'classes' tensor at last dimension

        boxes=tf.concat([boxes[:,:5],classes],axis=-1)

        

        boxes_dict=dict()

        for cls in range(n_classes):

            mask=tf.equal(boxes[:,5],cls)

            mask_shape=mask.get_shape()

            if mask_shape.ndims!=0:

                class_boxes=tf.boolean_mask(boxes,mask)

                boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,[4,1,-1],axis=-1)

                boxes_conf_scores=tf.reshape(boxes_conf_scores,[-1]) #flatten conf_scores completely

                indices=tf.image.non_max_suppression(boxes=boxes_coords,scores=boxes_conf_scores,

                                                     max_output_size=max_output_size,iou_threshold=iou_threshold)

                class_boxes=tf.gather(class_boxes,indices)

                boxes_dict[cls]=class_boxes[:,:5]

                

            

        boxes_dicts.append(boxes_dict)

    return boxes_dicts
class Yolo_v3:

    

    def __init__(self, n_classes, model_size, max_output_size, iou_threshold, confidence_threshold, data_format=None):

        '''

           max_output_size IS THE MAX NUM OF BOXES TO BE SELECTED PER CLASS

        '''

        if not data_format:

            if tf.test.is_gpu_available():

                data_format='channels_first'

            else:

                data_format='channels_last'

            

        self.n_classes=n_classes

        self.model_size=model_size

        self.max_output_size=max_output_size

        self.iou_threshold=iou_threshold

        self.confidence_threshold=confidence_threshold

        self.data_format=data_format

        

    

    def __call__(self,inputs,trainable):

        with tf.variable_scope('yolo_v3_model'):

            if self.data_format=='channels_first':

                inputs=tf.transpose(inputs,[0,3,1,2])  #bring the channels-dimension first, right after the batch-dimension

            inputs=inputs/255.

            

            route1,route2,inputs=darknet53(inputs=inputs, trainable=trainable, data_format=self.data_format)

            

            route,inputs=yolo_conv_block(inputs=inputs, filters=512, trainable=trainable, data_format=self.data_format)

            

            detect1=yolo_layer(inputs=inputs, n_classes=self.n_classes, anchors=_ANCHORS[6:9], image_size=self.model_size, data_format=self.data_format)

            #finished 1st detection layer

            

            

            inputs=conv2d_fixed_padding(inputs=route, filters=256, kernel_size=1, data_format=self.data_format)

            inputs=batch_norm(inputs=inputs, trainable=trainable, data_format=self.data_format)

            inputs=tf.nn.leaky_relu(features=inputs,alpha=_LEAKY_RELU)

            

            upsample_size=route2.get_shape().as_list()

            inputs=upsample(inputs=inputs,out_shape=upsample_size,data_format=self.data_format)

            

            axis=1 if self.data_format=='channels_first' else 3  #just set a main variable 'axis' as either 1 or 3 based on 'data_format'

            

            inputs=tf.concat([inputs,route2],axis=axis)

            route,inputs=yolo_conv_block(inputs=inputs, filters=256, trainable=trainable, data_format=self.data_format)

            

            detect2=yolo_layer(inputs=inputs, n_classes=self.n_classes, anchors=_ANCHORS[3:6], image_size=self.model_size, data_format=self.data_format)

            #finished 2nd detection layer

            

            

            inputs=conv2d_fixed_padding(inputs=route, filters=128, kernel_size=1, data_format=self.data_format)

            inputs=batch_norm(inputs=inputs, trainable=trainable, data_format=self.data_format)

            inputs=tf.nn.leaky_relu(features=inputs,alpha=_LEAKY_RELU)

            

            upsample_size=route1.get_shape().as_list()

            inputs=upsample(inputs=inputs, out_shape=upsample_size, data_format=self.data_format)

            

            inputs=tf.concat([inputs,route1],axis=axis)

            

            route,inputs=yolo_conv_block(inputs=inputs, filters=128, trainable=trainable, data_format=self.data_format)

            

            detect3=yolo_layer(inputs=inputs, n_classes=self.n_classes, anchors=_ANCHORS[0:3], image_size=self.model_size, data_format=self.data_format)

            #finished 3rd detection layer

            

            inputs=tf.concat([detect1,detect2,detect3],axis=1)

            print("\n\nYOU ARE NOW GOING TO BUILD BOXES\n\n")

            inputs=build_boxes(inputs)

            

            boxes_dicts=non_max_suppression(inputs=inputs, n_classes=self.n_classes, max_output_size=self.max_output_size, iou_threshold=self.iou_threshold, confidence_threshold=self.confidence_threshold)

            

            return boxes_dicts
def draw_boxes(img_names,boxes_dicts,class_names,model_size):

    colors = ((np.array(color_palette("hls", len(class_names))) * 255)).astype(np.uint8)

    for num, img_name, boxes_dict in zip(range(len(img_names)), img_names, boxes_dicts):

        img = Image.open(img_name)

        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype(font='/kaggle/input/data-for-yolo-v3-kernel/futur.ttf', size=(img.size[0] + img.size[1]) // 100)

        

        resize_factor = (img.size[0] / model_size[0], img.size[1] / model_size[1])

        for cls in range(len(class_names)):

            boxes = boxes_dict[cls]

            if tf.size(boxes) != 0:

                color = colors[cls]

                for box in boxes:

                    xy, confidence = box[:4], box[4]

                    xy = [xy[i] * resize_factor[i % 2] for i in range(4)]

                    x0, y0 = xy[0], xy[1]

                    

                    thickness = (img.size[0] + img.size[1]) // 200

                    for t in np.linspace(0, 1, thickness):

                        xy[0], xy[1] = xy[0] + t, xy[1] + t          #adjusting top left x and y

                        xy[2], xy[3] = xy[2] - t, xy[3] - t          #adjusting bottom right x and y

                        draw.rectangle(xy, outline=tuple(color))

                    

                    text = '{} {:.1f}%'.format(class_names[cls], confidence * 100)

                    

                    text_size = draw.textsize(text, font=font)

                    

                    draw.rectangle( [x0, y0 - text_size[1], x0 + text_size[0], y0],   fill=tuple(color))

                    

                    draw.text((x0, y0 - text_size[1]), text, fill='black', font=font)

        display(img)



        

def load_images(img_names, model_size):

    imgs=[]

    for imgname in img_names:

        img=Image.open(imgname)

        img=img.resize(size=model_size)

        img=np.array(img,dtype=np.float32)

        img=np.expand_dims(img,axis=0)

        imgs.append(img)

    

    imgs=np.concatenate(imgs)

    return imgs





def load_class_names(filename):

    with open(filename,'r') as f:

        class_names=f.read().splitlines()

    return class_names
def load_weights(variables,filename):

    with open(filename,'rb') as f:

        np.fromfile(f,dtype=np.int32,count=5)  #skip the first 5 lines

        weights=np.fromfile(f,dtype=np.float32) #remainder of file is 'weights'

        assign_operations=[]

        pointer=0

        

        #Load weights for Darknet-53

        #Each conv layer has batch normalization

        for i in range(52):

            conv_var=variables[5*i]

            gamma, beta, mean, variance=variables[(5*i + 1) : (5*i + 5)]

            batch_norm_variables=[beta, gamma, mean, variance]

            

            for var in batch_norm_variables:

                shape=var.shape.as_list()

                num_params=np.prod(shape)  #totalParams=shape[0]*shape[1]

                var_weights=weights[pointer : pointer+num_params].reshape(shape)

                pointer+=num_params

                assign_operations.append(tf.assign(var,var_weights))

                

            shape=conv_var.shape.as_list()

            num_params=np.prod(shape)

            var_weights=weights[pointer : pointer+num_params].reshape((shape[3],shape[2],shape[0],shape[1]))

            var_weights=np.transpose(var_weights,(2,3,1,0))

            pointer+=num_params

            assign_operations.append(tf.assign(conv_var,var_weights))

            

        #Load weights for the Yolo-layers

        #7th , 15th and 23rd layers don't have Batch-norm layer, but have use_biases=True since we use the default Conv2D function

        ranges=[range(0,6) , range(6,13) , range(13,20)]

        unnormalized=[6,13,20]

        

        for j in range(3):

            for i in ranges[j]: #for yolo-layers (0 -> 5)

                current=52*5 + 5*i + j*2

                conv_var=variables[current]

                gamma, beta, mean, variance=variables[current+1 : current+5]

                batch_norm_variables=[beta, gamma, mean, variance]

                

                for var in batch_norm_variables:

                    shape=var.shape.as_list()

                    num_params=np.prod(shape)

                    var_weights=weights[pointer : pointer+num_params].reshape(shape)

                    pointer+=num_params

                    assign_operations.append(tf.assign(var,var_weights))

                

                shape=conv_var.shape.as_list()

                #print(conv_var)

                num_params=np.prod(shape)

                var_weights=weights[pointer : pointer+num_params].reshape((shape[3],shape[2],shape[0],shape[1]))

                var_weights=np.transpose(var_weights,(2,3,1,0))

                pointer+=num_params

                assign_operations.append(tf.assign(conv_var,var_weights))

                

            bias =   variables[52*5 + 5*unnormalized[j] + 2*j + 1]

            shape=bias.shape.as_list()

            num_params=np.prod(shape)

            var_weights=weights[pointer : pointer+num_params].reshape(shape)

            pointer+=num_params

            assign_operations.append(tf.assign(bias,var_weights))

            

            conv_var=variables[52*5 + 5*unnormalized[j] + 2*j]

            shape=conv_var.shape.as_list()

            num_params=np.prod(shape)

            var_weights=weights[pointer : pointer+num_params].reshape((shape[3],shape[2],shape[0],shape[1]))

            var_weights=np.transpose(var_weights,(2,3,1,0))

            pointer+=num_params

            assign_operations.append(tf.assign(conv_var,var_weights))

            

    return assign_operations
images=['/kaggle/input/data-for-yolo-v3-kernel/office.jpg','/kaggle/input/yolo-sample-images/Manly_beach.jpg','../input/yolo-sample-images/Madame_Toussades-III.jpg']

for img in images:

    #display(Image.open(img))

    pass

batch_size = len(images)

batch = load_images(images, model_size=_MODEL_SIZE)

class_names = load_class_names('/kaggle/input/data-for-yolo-v3-kernel/coco.names')

n_classes = len(class_names)

max_output_size = 10

iou_threshold = 0.5

confidence_threshold = 0.5

tf.reset_default_graph()

model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,

                max_output_size=max_output_size,

                iou_threshold=iou_threshold,

                confidence_threshold=confidence_threshold)



#tf.compat.v1.disable_eager_execution()



inputs = tf.placeholder(tf.float32, [batch_size, 416, 416, 3])

detections=model(inputs, False)



model_vars=tf.global_variables(scope='yolo_v3_model')

assign_ops=load_weights(model_vars,'/kaggle/input/data-for-yolo-v3-kernel/yolov3.weights')

print("\n\n")



with tf.Session() as s:

    s.run(assign_ops)

    computed_detections_from_tensors=s.run(detections, feed_dict={inputs:batch})

computed_detections_from_tensors
draw_boxes(images,computed_detections_from_tensors,class_names,_MODEL_SIZE)