import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import keras as ks

%matplotlib inline
sample_image_path = "../input/open-images-2019-object-detection/test/6beb79b52308112d.jpg"
import struct

import numpy as np

from keras.layers import Conv2D

from keras.layers import Input

from keras.layers import BatchNormalization

from keras.layers import LeakyReLU

from keras.layers import ZeroPadding2D

from keras.layers import UpSampling2D

from keras.layers.merge import add, concatenate

from keras.models import Model

 

def _conv_block(inp, convs, skip=True):

	x = inp

	count = 0

	for conv in convs:

		if count == (len(convs) - 2) and skip:

			skip_connection = x

		count += 1

		if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top

		x = Conv2D(conv['filter'],

				   conv['kernel'],

				   strides=conv['stride'],

				   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top

				   name='conv_' + str(conv['layer_idx']),

				   use_bias=False if conv['bnorm'] else True)(x)

		if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)

		if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

	return add([skip_connection, x]) if skip else x

 

def make_yolov3_model():

	input_image = Input(shape=(None, None, 3))

	# Layer  0 => 4

	x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},

								  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},

								  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},

								  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

	# Layer  5 => 8

	x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},

						{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},

						{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

	# Layer  9 => 11

	x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},

						{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

	# Layer 12 => 15

	x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},

						{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},

						{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

	# Layer 16 => 36

	for i in range(7):

		x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},

							{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])

	skip_36 = x

	# Layer 37 => 40

	x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},

						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},

						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

	# Layer 41 => 61

	for i in range(7):

		x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},

							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])

	skip_61 = x

	# Layer 62 => 65

	x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},

						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},

						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

	# Layer 66 => 74

	for i in range(3):

		x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},

							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])

	# Layer 75 => 79

	x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},

						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},

						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},

						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},

						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

	# Layer 80 => 82

	yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},

							  {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

	# Layer 83 => 86

	x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)

	x = UpSampling2D(2)(x)

	x = concatenate([x, skip_61])

	# Layer 87 => 91

	x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},

						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},

						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},

						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},

						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

	# Layer 92 => 94

	yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},

							  {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)

	# Layer 95 => 98

	x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)

	x = UpSampling2D(2)(x)

	x = concatenate([x, skip_36])

	# Layer 99 => 106

	yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},

							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},

							   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},

							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},

							   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},

							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},

							   {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

	model = Model(input_image, [yolo_82, yolo_94, yolo_106])

	return model

 

class WeightReader:

	def __init__(self, weight_file):

		with open(weight_file, 'rb') as w_f:

			major,	= struct.unpack('i', w_f.read(4))

			minor,	= struct.unpack('i', w_f.read(4))

			revision, = struct.unpack('i', w_f.read(4))

			if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:

				w_f.read(8)

			else:

				w_f.read(4)

			transpose = (major > 1000) or (minor > 1000)

			binary = w_f.read()

		self.offset = 0

		self.all_weights = np.frombuffer(binary, dtype='float32')

 

	def read_bytes(self, size):

		self.offset = self.offset + size

		return self.all_weights[self.offset-size:self.offset]

 

	def load_weights(self, model):

		for i in range(106):

			try:

				conv_layer = model.get_layer('conv_' + str(i))

				print("loading weights of convolution #" + str(i))

				if i not in [81, 93, 105]:

					norm_layer = model.get_layer('bnorm_' + str(i))

					size = np.prod(norm_layer.get_weights()[0].shape)

					beta  = self.read_bytes(size) # bias

					gamma = self.read_bytes(size) # scale

					mean  = self.read_bytes(size) # mean

					var   = self.read_bytes(size) # variance

					weights = norm_layer.set_weights([gamma, beta, mean, var])

				if len(conv_layer.get_weights()) > 1:

					bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))

					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))

					kernel = kernel.transpose([2,3,1,0])

					conv_layer.set_weights([kernel, bias])

				else:

					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))

					kernel = kernel.transpose([2,3,1,0])

					conv_layer.set_weights([kernel])

			except ValueError:

				print("no convolution #" + str(i))

 

	def reset(self):

		self.offset = 0

 
#define model

model = make_yolov3_model()

# load the model weights

weight_reader = WeightReader('../input/yoloweight/yolov3.weights')

# set the model weights into the model

weight_reader.load_weights(model)

# save the model to file

model.save('model.h5')
# load yolov3 model and perform object detection

# based on https://github.com/experiencor/keras-yolo3

import numpy as np

from numpy import expand_dims

from keras.models import load_model

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from matplotlib import pyplot

from matplotlib.patches import Rectangle



class BoundBox:

	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):

		self.xmin = xmin

		self.ymin = ymin

		self.xmax = xmax

		self.ymax = ymax

		self.objness = objness

		self.classes = classes

		self.label = -1

		self.score = -1



	def get_label(self):

		if self.label == -1:

			self.label = np.argmax(self.classes)



		return self.label



	def get_score(self):

		if self.score == -1:

			self.score = self.classes[self.get_label()]



		return self.score



def _sigmoid(x):

	return 1. / (1. + np.exp(-x))



def decode_netout(netout, anchors, obj_thresh, net_h, net_w):

	grid_h, grid_w = netout.shape[:2]

	nb_box = 3

	netout = netout.reshape((grid_h, grid_w, nb_box, -1))

	nb_class = netout.shape[-1] - 5

	boxes = []

	netout[..., :2]  = _sigmoid(netout[..., :2])

	netout[..., 4:]  = _sigmoid(netout[..., 4:])

	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]

	netout[..., 5:] *= netout[..., 5:] > obj_thresh

	for i in range(grid_h*grid_w):

		row = i / grid_w

		col = i % grid_w

		for b in range(nb_box):

			# 4th element is objectness score

			objectness = netout[int(row)][int(col)][b][4]

			if(objectness.all() <= obj_thresh): continue

			# first 4 elements are x, y, w, and h

			x, y, w, h = netout[int(row)][int(col)][b][:4]

			x = (col + x) / grid_w # center position, unit: image width

			y = (row + y) / grid_h # center position, unit: image height

			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width

			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height

			# last elements are class probabilities

			classes = netout[int(row)][col][b][5:]

			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

			boxes.append(box)

	return boxes



def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):

	new_w, new_h = net_w, net_h

	for i in range(len(boxes)):

		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w

		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)

		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)

		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)

		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)



def _interval_overlap(interval_a, interval_b):

	x1, x2 = interval_a

	x3, x4 = interval_b

	if x3 < x1:

		if x4 < x1:

			return 0

		else:

			return min(x2,x4) - x1

	else:

		if x2 < x3:

			 return 0

		else:

			return min(x2,x4) - x3



def bbox_iou(box1, box2):

	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])

	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

	intersect = intersect_w * intersect_h

	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin

	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

	union = w1*h1 + w2*h2 - intersect

	return float(intersect) / union



def do_nms(boxes, nms_thresh):

	if len(boxes) > 0:

		nb_class = len(boxes[0].classes)

	else:

		return

	for c in range(nb_class):

		sorted_indices = np.argsort([-box.classes[c] for box in boxes])

		for i in range(len(sorted_indices)):

			index_i = sorted_indices[i]

			if boxes[index_i].classes[c] == 0: continue

			for j in range(i+1, len(sorted_indices)):

				index_j = sorted_indices[j]

				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:

					boxes[index_j].classes[c] = 0



# load and prepare an image

def load_image_pixels(filename, shape):

	# load the image to get its shape

	image = load_img(filename)

	width, height = image.size

	# load the image with the required size

	image = load_img(filename, target_size=shape)

	# convert to numpy array

	image = img_to_array(image)

	# scale pixel values to [0, 1]

	image = image.astype('float32')

	image /= 255.0

	# add a dimension so that we have one sample

	image = expand_dims(image, 0)

	return image, width, height



# get all of the results above a threshold

def get_boxes(boxes, labels, thresh):

	v_boxes, v_labels, v_scores = list(), list(), list()

	# enumerate all boxes

	for box in boxes:

		# enumerate all possible labels

		for i in range(len(labels)):

			# check if the threshold for this label is high enough

			if box.classes[i] > thresh:

				v_boxes.append(box)

				v_labels.append(labels[i])

				v_scores.append(box.classes[i]*100)

				# don't break, many labels may trigger for one box

	return v_boxes, v_labels, v_scores



# draw all results

def draw_boxes(filename, v_boxes, v_labels, v_scores):

	# load the image

	data = pyplot.imread(filename)

	# plot the image

	pyplot.imshow(data)

	# get the context for drawing boxes

	ax = pyplot.gca()

	# plot each box

	for i in range(len(v_boxes)):

		box = v_boxes[i]

		# get coordinates

		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax

		# calculate width and height of the box

		width, height = x2 - x1, y2 - y1

		# create the shape

		rect = Rectangle((x1, y1), width, height, fill=False, color='white')

		# draw the box

		ax.add_patch(rect)

		# draw text and score in top left corner

		label = "%s (%.3f)" % (v_labels[i], v_scores[i])

		pyplot.text(x1, y1, label, color='white')

	# show the plot

	pyplot.show()



def workflow(model,image_path=sample_image_path):

    # define the expected input shape for the model

    input_w, input_h = 416, 416

    # define our new photo

    photo_filename = image_path

    # load and prepare image

    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

    # make prediction

    yhat = model.predict(image)

    # summarize the shape of the list of arrays

    #print([a.shape for a in yhat])

    # define the anchors

    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

    # define the probability threshold for detected objects

    class_threshold = 0.6

    boxes = list()

    for i in range(len(yhat)):

        # decode the output of the network

        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

    # correct the sizes of the bounding boxes for the shape of the image

    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    # suppress non-maximal boxes

    do_nms(boxes, 0.5)

    # define the labels

    labels = ["Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck",

        "Boat", "Traffic sign", "Fire hydrant", "Stop sign", "Parking meter", "Bench",

        "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra", "Giraffe",

        "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis", "Snowboard",

        "Ball", "Kite", "Baseball bat", "Baseball glove", "Skateboard", "Surfboard",

        "Tennis racket", "Bottle", "Wine glass", "Cup", "Fork", "Knife", "Spoon", "Bowl", "Banana",

        "Apple", "Sandwich", "Orange", "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut", "Cake",

        "Chair", "Sofa", "Pottedplant", "Bed", "Dining table", "Toilet", "Tvmonitor", "Laptop", "Mouse",

        "Remote", "Keyboard", "Mobile phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator",

        "Book", "Clock", "Vase", "Scissors", "Teddy bear", "Hair drier", "Toothbrush"]

    # get the details of the detected objects

    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    return (v_boxes, v_labels, v_scores)

# load yolov3 model

yolo = load_model('model.h5')

v_boxes, v_labels, v_scores=workflow(yolo,sample_image_path)

# summarize what we found

for i in range(len(v_boxes)):

	print(v_labels[i], v_scores[i])

# draw what we found

draw_boxes(sample_image_path, v_boxes, v_labels, v_scores)
vocab = {"/m/011k07": "Tortoise", "/m/011q46kg": "Container", "/m/012074": "Magpie", "/m/0120dh": "Sea turtle", "/m/01226z": "Football", "/m/012n7d": "Ambulance", "/m/012w5l": "Ladder", "/m/012xff": "Toothbrush", "/m/012ysf": "Syringe", "/m/0130jx": "Sink", "/m/0138tl": "Toy", "/m/013y1f": "Organ", "/m/01432t": "Cassette deck", "/m/014j1m": "Apple", "/m/014sv8": "Human eye", "/m/014trl": "Cosmetics", "/m/014y4n": "Paddle", "/m/0152hh": "Snowman", "/m/01599": "Beer", "/m/01_5g": "Chopsticks", "/m/015h_t": "Human beard", "/m/015p6": "Bird", "/m/015qbp": "Parking meter", "/m/015qff": "Traffic light", "/m/015wgc": "Croissant", "/m/015x4r": "Cucumber", "/m/015x5n": "Radish", "/m/0162_1": "Towel", "/m/0167gd": "Doll", "/m/016m2d": "Skull", "/m/0174k2": "Washing machine", "/m/0174n1": "Glove", "/m/0175cv": "Tick", "/m/0176mf": "Belt", "/m/017ftj": "Sunglasses", "/m/018j2": "Banjo", "/m/018p4k": "Cart", "/m/018xm": "Ball", "/m/01940j": "Backpack", "/m/0199g": "Bicycle", "/m/019dx1": "Home appliance", "/m/019h78": "Centipede", "/m/019jd": "Boat", "/m/019w40": "Surfboard", "/m/01b638": "Boot", "/m/01b7fy": "Headphones", "/m/01b9xk": "Hot dog", "/m/01bfm9": "Shorts", "/m/01_bhs": "Fast food", "/m/01bjv": "Bus", "/m/01bl7v": "Boy", "/m/01bms0": "Screwdriver", "/m/01bqk0": "Bicycle wheel", "/m/01btn": "Barge", "/m/01c648": "Laptop", "/m/01cmb2": "Miniskirt", "/m/01d380": "Drill", "/m/01d40f": "Dress", "/m/01dws": "Bear", "/m/01dwsz": "Waffle", "/m/01dwwc": "Pancake", "/m/01dxs": "Brown bear", "/m/01dy8n": "Woodpecker", "/m/01f8m5": "Blue jay", "/m/01f91_": "Pretzel", "/m/01fb_0": "Bagel", "/m/01fdzj": "Tower", "/m/01fh4r": "Teapot", "/m/01g317": "Person", "/m/01g3x7": "Bow and arrow", "/m/01gkx_": "Swimwear", "/m/01gllr": "Beehive", "/m/01gmv2": "Brassiere", "/m/01h3n": "Bee", "/m/01h44": "Bat", "/m/01h8tj": "Starfish", "/m/01hrv5": "Popcorn", "/m/01j3zr": "Burrito", "/m/01j4z9": "Chainsaw", "/m/01j51": "Balloon", "/m/01j5ks": "Wrench", "/m/01j61q": "Tent", "/m/01jfm_": "Vehicle registration plate", "/m/01jfsr": "Lantern", "/m/01k6s3": "Toaster", "/m/01kb5b": "Flashlight", "/m/01knjb": "Billboard", "/m/01krhy": "Tiara", "/m/01lcw4": "Limousine", "/m/01llwg": "Necklace", "/m/01lrl": "Carnivore", "/m/01lsmm": "Scissors", "/m/01lynh": "Stairs", "/m/01m2v": "Computer keyboard", "/m/01m4t": "Printer", "/m/01mqdt": "Traffic sign", "/m/01mzpv": "Chair", "/m/01n4qj": "Shirt", "/m/01n5jq": "Poster", "/m/01nkt": "Cheese", "/m/01nq26": "Sock", "/m/01pns0": "Fire hydrant", "/m/01prls": "Land vehicle", "/m/01r546": "Earrings", "/m/01rkbr": "Tie", "/m/01rzcn": "Watercraft", "/m/01s105": "Cabinetry", "/m/01s55n": "Suitcase", "/m/01tcjp": "Muffin", "/m/01vbnl": "Bidet", "/m/01ww8y": "Snack", "/m/01x3jk": "Snowmobile", "/m/01x3z": "Clock", "/m/01xgg_": "Medical equipment", "/m/01xq0k1": "Cattle", "/m/01xqw": "Cello", "/m/01xs3r": "Jet ski", "/m/01x_v": "Camel", "/m/01xygc": "Coat", "/m/01xyhv": "Suit", "/m/01y9k5": "Desk", "/m/01yrx": "Cat", "/m/01yx86": "Bronze sculpture", "/m/01z1kdw": "Juice", "/m/02068x": "Gondola", "/m/020jm": "Beetle", "/m/020kz": "Cannon", "/m/020lf": "Computer mouse", "/m/021mn": "Cookie", "/m/021sj1": "Office building", "/m/0220r2": "Fountain", "/m/0242l": "Coin", "/m/024d2": "Calculator", "/m/024g6": "Cocktail", "/m/02522": "Computer monitor", "/m/025dyy": "Box", "/m/025fsf": "Stapler", "/m/025nd": "Christmas tree", "/m/025rp__": "Cowboy hat", "/m/0268lbt": "Hiking equipment", "/m/026qbn5": "Studio couch", "/m/026t6": "Drum", "/m/0270h": "Dessert", "/m/0271qf7": "Wine rack", "/m/0271t": "Drink", "/m/027pcv": "Zucchini", "/m/027rl48": "Ladle", "/m/0283dt1": "Human mouth", "/m/0284d": "Dairy", "/m/029b3": "Dice", "/m/029bxz": "Oven", "/m/029tx": "Dinosaur", "/m/02bm9n": "Ratchet", "/m/02crq1": "Couch", "/m/02ctlc": "Cricket ball", "/m/02cvgx": "Winter melon", "/m/02d1br": "Spatula", "/m/02d9qx": "Whiteboard", "/m/02ddwp": "Pencil sharpener", "/m/02dgv": "Door", "/m/02dl1y": "Hat", "/m/02f9f_": "Shower", "/m/02fh7f": "Eraser", "/m/02fq_6": "Fedora", "/m/02g30s": "Guacamole", "/m/02gzp": "Dagger", "/m/02h19r": "Scarf", "/m/02hj4": "Dolphin", "/m/02jfl0": "Sombrero", "/m/02jnhm": "Tin can", "/m/02jvh9": "Mug", "/m/02jz0l": "Tap", "/m/02l8p9": "Harbor seal", "/m/02lbcq": "Stretcher", "/m/02mqfb": "Can opener", "/m/02_n6y": "Goggles", "/m/02p0tk3": "Human body", "/m/02p3w7d": "Roller skates", "/m/02p5f1q": "Cup", "/m/02pdsw": "Cutting board", "/m/02pjr4": "Blender", "/m/02pkr5": "Plumbing fixture", "/m/02pv19": "Stop sign", "/m/02rdsp": "Office supplies", "/m/02rgn06": "Volleyball", "/m/02s195": "Vase", "/m/02tsc9": "Slow cooker", "/m/02vkqh8": "Wardrobe", "/m/02vqfm": "Coffee", "/m/02vwcm": "Whisk", "/m/02w3r3": "Paper towel", "/m/02w3_ws": "Personal care", "/m/02wbm": "Food", "/m/02wbtzl": "Sun hat", "/m/02wg_p": "Tree house", "/m/02wmf": "Flying disc", "/m/02wv6h6": "Skirt", "/m/02wv84t": "Gas stove", "/m/02x8cch": "Salt and pepper shakers", "/m/02x984l": "Mechanical fan", "/m/02xb7qb": "Face powder", "/m/02xqq": "Fax", "/m/02xwb": "Fruit", "/m/02y6n": "French fries", "/m/02z51p": "Nightstand", "/m/02zn6n": "Barrel", "/m/02zt3": "Kite", "/m/02zvsm": "Tart", "/m/030610": "Treadmill", "/m/0306r": "Fox", "/m/03120": "Flag", "/m/0319l": "Horn", "/m/031b6r": "Window blind", "/m/031n1": "Human foot", "/m/0323sq": "Golf cart", "/m/032b3c": "Jacket", "/m/033cnk": "Egg", "/m/033rq4": "Street light", "/m/0342h": "Guitar", "/m/034c16": "Pillow", "/m/035r7c": "Human leg", "/m/035vxb": "Isopod", "/m/0388q": "Grape", "/m/039xj_": "Human ear", "/m/03bbps": "Power plugs and sockets", "/m/03bj1": "Panda", "/m/03bk1": "Giraffe", "/m/03bt1vf": "Woman", "/m/03c7gz": "Door handle", "/m/03d443": "Rhinoceros", "/m/03dnzn": "Bathtub", "/m/03fj2": "Goldfish", "/m/03fp41": "Houseplant", "/m/03fwl": "Goat", "/m/03g8mr": "Baseball bat", "/m/03grzl": "Baseball glove", "/m/03hj559": "Mixing bowl", "/m/03hl4l9": "Marine invertebrates", "/m/03hlz0c": "Kitchen utensil", "/m/03jbxj": "Light switch", "/m/03jm5": "House", "/m/03k3r": "Horse", "/m/03kt2w": "Stationary bicycle", "/m/03l9g": "Hammer", "/m/03ldnb": "Ceiling fan", "/m/03m3pdh": "Sofa bed", "/m/03m3vtv": "Adhesive tape", "/m/03m5k": "Harp", "/m/03nfch": "Sandal", "/m/03p3bw": "Bicycle helmet", "/m/03q5c7": "Saucer", "/m/03q5t": "Harpsichord", "/m/03q69": "Human hair", "/m/03qhv5": "Heater", "/m/03qjg": "Harmonica", "/m/03qrc": "Hamster", "/m/03rszm": "Curtain", "/m/03ssj5": "Bed", "/m/03s_tn": "Kettle", "/m/03tw93": "Fireplace", "/m/03txqz": "Scale", "/m/03v5tg": "Drinking straw", "/m/03vt0": "Insect", "/m/03wvsk": "Hair dryer", "/m/03_wxk": "Kitchenware", "/m/03wym": "Indoor rower", "/m/03xxp": "Invertebrate", "/m/03y6mg": "Food processor", "/m/03__z0": "Bookcase", "/m/040b_t": "Refrigerator", "/m/04169hn": "Wood-burning stove", "/m/0420v5": "Punching bag", "/m/043nyj": "Common fig", "/m/0440zs": "Cocktail shaker", "/m/0449p": "Jaguar", "/m/044r5d": "Golf ball", "/m/0463sg": "Fashion accessory", "/m/046dlr": "Alarm clock", "/m/047j0r": "Filing cabinet", "/m/047v4b": "Artichoke", "/m/04bcr3": "Table", "/m/04brg2": "Tableware", "/m/04c0y": "Kangaroo", "/m/04cp_": "Koala", "/m/04ctx": "Knife", "/m/04dr76w": "Bottle", "/m/04f5ws": "Bottle opener", "/m/04g2r": "Lynx", "/m/04gth": "Lavender", "/m/04h7h": "Lighthouse", "/m/04h8sr": "Dumbbell", "/m/04hgtk": "Human head", "/m/04kkgm": "Bowl", "/m/04lvq_": "Humidifier", "/m/04m6gz": "Porch", "/m/04m9y": "Lizard", "/m/04p0qw": "Billiard table", "/m/04rky": "Mammal", "/m/04rmv": "Mouse", "/m/04_sv": "Motorcycle", "/m/04szw": "Musical instrument", "/m/04tn4x": "Swim cap", "/m/04v6l4": "Frying pan", "/m/04vv5k": "Snowplow", "/m/04y4h8h": "Bathroom cabinet", "/m/04ylt": "Missile", "/m/04yqq2": "Bust", "/m/04yx4": "Man", "/m/04z4wx": "Waffle iron", "/m/04zpv": "Milk", "/m/04zwwv": "Ring binder", "/m/050gv4": "Plate", "/m/050k8": "Mobile phone", "/m/052lwg6": "Baked goods", "/m/052sf": "Mushroom", "/m/05441v": "Crutch", "/m/054fyh": "Pitcher", "/m/054_l": "Mirror", "/m/054xkw": "Lifejacket", "/m/05_5p_0": "Table tennis racket", "/m/05676x": "Pencil case", "/m/057cc": "Musical keyboard", "/m/057p5t": "Scoreboard", "/m/0584n8": "Briefcase", "/m/058qzx": "Kitchen knife", "/m/05bm6": "Nail", "/m/05ctyq": "Tennis ball", "/m/05gqfk": "Plastic bag", "/m/05kms": "Oboe", "/m/05kyg_": "Chest of drawers", "/m/05n4y": "Ostrich", "/m/05r5c": "Piano", "/m/05r655": "Girl", "/m/05s2s": "Plant", "/m/05vtc": "Potato", "/m/05w9t9": "Hair spray", "/m/05y5lj": "Sports equipment", "/m/05z55": "Pasta", "/m/05z6w": "Penguin", "/m/05zsy": "Pumpkin", "/m/061_f": "Pear", "/m/061hd_": "Infant bed", "/m/0633h": "Polar bear", "/m/063rgb": "Mixer", "/m/0642b4": "Cupboard", "/m/065h6l": "Jacuzzi", "/m/0663v": "Pizza", "/m/06_72j": "Digital clock", "/m/068zj": "Pig", "/m/06bt6": "Reptile", "/m/06c54": "Rifle", "/m/06c7f7": "Lipstick", "/m/06_fw": "Skateboard", "/m/06j2d": "Raven", "/m/06k2mb": "High heels", "/m/06l9r": "Red panda", "/m/06m11": "Rose", "/m/06mf6": "Rabbit", "/m/06msq": "Sculpture", "/m/06ncr": "Saxophone", "/m/06nrc": "Shotgun", "/m/06nwz": "Seafood", "/m/06pcq": "Submarine sandwich", "/m/06__v": "Snowboard", "/m/06y5r": "Sword", "/m/06z37_": "Picture frame", "/m/07030": "Sushi", "/m/0703r8": "Loveseat", "/m/071p9": "Ski", "/m/071qp": "Squirrel", "/m/073bxn": "Tripod", "/m/073g6": "Stethoscope", "/m/074d1": "Submarine", "/m/0755b": "Scorpion", "/m/076bq": "Segway", "/m/076lb9": "Training bench", "/m/078jl": "Snake", "/m/078n6m": "Coffee table", "/m/079cl": "Skyscraper", "/m/07bgp": "Sheep", "/m/07c52": "Television", "/m/07c6l": "Trombone", "/m/07clx": "Tea", "/m/07cmd": "Tank", "/m/07crc": "Taco", "/m/07cx4": "Telephone", "/m/07dd4": "Torch", "/m/07dm6": "Tiger", "/m/07fbm7": "Strawberry", "/m/07gql": "Trumpet", "/m/07j7r": "Tree", "/m/07j87": "Tomato", "/m/07jdr": "Train", "/m/07k1x": "Tool", "/m/07kng9": "Picnic basket", "/m/07mcwg": "Cooking spray", "/m/07mhn": "Trousers", "/m/07pj7bq": "Bowling equipment", "/m/07qxg_": "Football helmet", "/m/07r04": "Truck", 

         "/m/07v9_z": "Measuring cup", "/m/07xyvk": "Coffeemaker", "/m/07y_7": "Violin", "/m/07yv9": "Vehicle", "/m/080hkjn": "Handbag", "/m/080n7g": "Paper cutter", "/m/081qc": "Wine", "/m/083kb": "Weapon", "/m/083wq": "Wheel", "/m/084hf": "Worm", "/m/084rd": "Wok", "/m/084zz": "Whale", "/m/0898b": "Zebra", "/m/08dz3q": "Auto part", "/m/08hvt4": "Jug", "/m/08ks85": "Pizza cutter", "/m/08p92x": "Cream", "/m/08pbxl": "Monkey", "/m/096mb": "Lion", "/m/09728": "Bread", "/m/099ssp": "Platter", "/m/09b5t": "Chicken", "/m/09csl": "Eagle", "/m/09ct_": "Helicopter", "/m/09d5_": "Owl", "/m/09ddx": "Duck", "/m/09dzg": "Turtle", "/m/09f20": "Hippopotamus", "/m/09f_2": "Crocodile", "/m/09g1w": "Toilet", "/m/09gtd": "Toilet paper", "/m/09gys": "Squid", "/m/09j2d": "Clothing", "/m/09j5n": "Footwear", "/m/09k_b": "Lemon", "/m/09kmb": "Spider", "/m/09kx5": "Deer", "/m/09ld4": "Frog", "/m/09qck": "Banana", "/m/09rvcxw": "Rocket", "/m/09tvcd": "Wine glass", "/m/0b3fp9": "Countertop", "/m/0bh9flk": "Tablet computer", "/m/0bjyj5": "Waste container", "/m/0b_rs": "Swimming pool", "/m/0bt9lr": "Dog", "/m/0bt_c3": "Book", "/m/0bwd_0j": "Elephant", "/m/0by6g": "Shark", "/m/0c06p": "Candle", "/m/0c29q": "Leopard", "/m/0c2jj": "Axe", "/m/0c3m8g": "Hand dryer", "/m/0c3mkw": "Soap dispenser", "/m/0c568": "Porcupine", "/m/0c9ph5": "Flower", "/m/0ccs93": "Canary", "/m/0cd4d": "Cheetah", "/m/0cdl1": "Palm tree", "/m/0cdn1": "Hamburger", "/m/0cffdh": "Maple", "/m/0cgh4": "Building", "/m/0ch_cf": "Fish", "/m/0cjq5": "Lobster", "/m/0cjs7": "Asparagus", "/m/0c_jw": "Furniture", "/m/0cl4p": "Hedgehog", "/m/0cmf2": "Aeroplane", "/m/0cmx8": "Spoon", "/m/0cn6p": "Otter", "/m/0cnyhnx": "Bull", "/m/0_cp5": "Oyster", "/m/0cqn2": "Horizontal bar", "/m/0crjs": "Convenience store", "/m/0ct4f": "Bomb", "/m/0cvnqh": "Bench", "/m/0cxn2": "Ice cream", "/m/0cydv": "Caterpillar", "/m/0cyf8": "Butterfly", "/m/0cyfs": "Parachute", "/m/0cyhj_": "Orange", "/m/0czz2": "Antelope", "/m/0d20w4": "Beaker", "/m/0d_2m": "Moths and butterflies", "/m/0d4v4": "Window", "/m/0d4w1": "Closet", "/m/0d5gx": "Castle", "/m/0d8zb": "Jellyfish", "/m/0dbvp": "Goose", "/m/0dbzx": "Mule", "/m/0dftk": "Swan", "/m/0dj6p": "Peach", "/m/0djtd": "Coconut", "/m/0dkzw": "Seat belt", "/m/0dq75": "Raccoon", "/m/0_dqb": "Chisel", "/m/0dt3t": "Fork", "/m/0dtln": "Lamp", "/m/0dv5r": "Camera", "/m/0dv77": "Squash", "/m/0dv9c": "Racket", "/m/0dzct": "Human face", "/m/0dzf4": "Human arm", "/m/0f4s2w": "Vegetable", "/m/0f571": "Diaper", "/m/0f6nr": "Unicycle", "/m/0f6wt": "Falcon", "/m/0f8s22": "Chime", "/m/0f9_l": "Snail", "/m/0fbdv": "Shellfish", "/m/0fbw6": "Cabbage", "/m/0fj52s": "Carrot", "/m/0fldg": "Mango", "/m/0fly7": "Jeans", "/m/0fm3zh": "Flowerpot", "/m/0fp6w": "Pineapple", "/m/0fqfqc": "Drawer", "/m/0fqt361": "Stool", "/m/0frqm": "Envelope", "/m/0fszt": "Cake", "/m/0ft9s": "Dragonfly", "/m/0ftb8": "Sunflower", "/m/0fx9l": "Microwave oven", "/m/0fz0h": "Honeycomb", "/m/0gd2v": "Marine mammal", "/m/0gd36": "Sea lion", "/m/0gj37": "Ladybug", "/m/0gjbg72": "Shelf", "/m/0gjkl": "Watch", "/m/0gm28": "Candy", "/m/0grw1": "Salad", "/m/0gv1x": "Parrot", "/m/0gxl3": "Handgun", "/m/0h23m": "Sparrow", "/m/0h2r6": "Van", "/m/0h8jyh6": "Grinder", "/m/0h8kx63": "Spice rack", "/m/0h8l4fh": "Light bulb", "/m/0h8lkj8": "Corded phone", "/m/0h8mhzd": "Sports uniform", "/m/0h8my_4": "Tennis racket", "/m/0h8mzrc": "Wall clock", "/m/0h8n27j": "Serving tray", "/m/0h8n5zk": "Dining table", "/m/0h8n6f9": "Dog bed", "/m/0h8n6ft": "Cake stand", "/m/0h8nm9j": "Cat furniture", "/m/0h8nr_l": "Bathroom accessory", "/m/0h8nsvg": "Facial tissue holder", "/m/0h8ntjv": "Pressure cooker", "/m/0h99cwc": "Kitchen appliance", "/m/0h9mv": "Tire", "/m/0hdln": "Ruler", "/m/0hf58v5": "Luggage and bags", "/m/0hg7b": "Microphone", "/m/0hkxq": "Broccoli", "/m/0hnnb": "Umbrella", "/m/0hnyx": "Pastry", "/m/0hqkz": "Grapefruit", "/m/0j496": "Band-aid", "/m/0jbk": "Animal", "/m/0jg57": "Bell pepper", "/m/0jly1": "Turkey", "/m/0jqgx": "Lily", "/m/0jwn_": "Pomegranate", "/m/0jy4k": "Doughnut", "/m/0jyfg": "Glasses", "/m/0k0pj": "Human nose", "/m/0k1tl": "Pen", "/m/0_k2": "Ant", "/m/0k4j": "Car", "/m/0k5j": "Aircraft", "/m/0k65p": "Human hand", "/m/0km7z": "Skunk", "/m/0kmg4": "Teddy bear", "/m/0kpqd": "Watermelon", "/m/0kpt_": "Cantaloupe", "/m/0ky7b": "Dishwasher", "/m/0l14j_": "Flute", "/m/0l3ms": "Balance beam", "/m/0l515": "Sandwich", "/m/0ll1f78": "Shrimp", "/m/0llzx": "Sewing machine", "/m/0lt4_": "Binoculars", "/m/0m53l": "Rays and skates", "/m/0mcx2": "Ipod", "/m/0mkg": "Accordion", "/m/0mw_6": "Willow", "/m/0n28_": "Crab", "/m/0nl46": "Crown", "/m/0nybt": "Seahorse", "/m/0p833": "Perfume", "/m/0pcr": "Alpaca", "/m/0pg52": "Taxi", "/m/0ph39": "Canoe", "/m/0qjjc": "Remote control", "/m/0qmmr": "Wheelchair", "/m/0wdt60w": "Rugby ball", "/m/0xfy": "Armadillo", "/m/0xzly": "Maracas", "/m/0zvk5": "Helmet"}

rev = {}

for k,v in vocab.items():

    rev[v] = k 
pred_str=''

for i in range(len(v_labels)):

    pred_str=pred_str + rev[v_labels[i]]+' '+ str(v_scores[i]/100)+ ' '

    pred_str=pred_str+ str(v_boxes[i].xmin/1000)+' '+str(v_boxes[i].ymin/1000)+' '

    pred_str=pred_str+str(v_boxes[i].xmax/1000)+' '+str(v_boxes[i].ymax/1000)+' '

    

print(pred_str)
from tqdm import tqdm



sample_submission_df = pd.read_csv('../input/open-images-2019-object-detection/sample_submission.csv')

image_ids = sample_submission_df['ImageId']

predictions = []



i=0

# load yolov3 model

yolo = load_model('model.h5')

for image_id in tqdm(image_ids):

    # Load the image string

    image_path = f'../input/open-images-2019-object-detection/test/{image_id}.jpg'

    i+=1

    v_boxes, v_labels, v_scores=workflow(yolo,image_path)

    pred_str=''

    fig = plt.figure(figsize=(20, 15))

    #draw_boxes(image_path, v_boxes, v_labels, v_scores)

    for i in range(len(v_labels)):

        pred_str=pred_str + rev[v_labels[i]]+' '+ str(v_scores[i]/100)+ ' '

        pred_str=pred_str+ str(v_boxes[i].xmin/1000)+' '+str(v_boxes[i].ymin/1000)+' '

        pred_str=pred_str+str(v_boxes[i].xmax/1000)+' '+str(v_boxes[i].ymax/1000)+' '

    #print(pred_str)

    predictions.append(pred_str)

    plt.close()

    

    