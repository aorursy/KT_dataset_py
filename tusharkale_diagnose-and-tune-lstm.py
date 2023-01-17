# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from random import random
from numpy import array
from matplotlib import pyplot
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# generate a rectangle with random width and height
def random_rectangle():
	width, height = random(), random()
	points = list()
	# bottom left
	points.append([0.0, 0.0])
	# bottom right
	points.append([width, 0.0])
	# top right
	points.append([width, height])
	# top left
	points.append([0.0, height])
	return points

# plot a rectangle
def plot_rectangle(rect):
	# close the rectangle path
	rect.append(rect[0])
	# define path
	codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
	path = Path(rect, codes)
	axis = pyplot.gca()
	patch = PathPatch(path)
	# add shape to plot
	axis.add_patch(patch)
	axis.set_xlim(-0.1,1.1)
	axis.set_ylim(-0.1,1.1)
	pyplot.show()

# generate input and output sequences for one random rectangle
def get_samples():
	# generate rectangle
	rect = random_rectangle()
	X, y = list(), list()
	# create input output pairs for each coordinate
	for i in range(1, len(rect)):
		X.append(rect[i-1])
		y.append(rect[i])
	# convert input sequence shape to have 1 time step and 2 features
	X, y = array(X), array(y)
	X = X.reshape((X.shape[0], 1, 2))
	return X, y

# use a fit LSTM model to generate a new rectangle from scratch
def generate_rectangle(model):
	rect = list()
	# use [0,0] to seed the generation process
	last = array([0.0,0.0]).reshape((1, 1, 2))
	rect.append([[y for y in x] for x in last[0]][0])
	# generate the remaining 3 coordinates
	for _ in range(3):
		# predict the next coordinate
		yhat = model.predict(last, verbose=0)
		# use this output as input for the next prediction
		last = yhat.reshape((1, 1, 2))
		# store coordinate
		rect.append([[y for y in x] for x in last[0]][0])
	return rect

# define model
model = Sequential()
model.add(LSTM(10, input_shape=(1, 2)))
model.add(Dense(2, activation='linear'))
model.compile(loss='mae', optimizer='adam')
model.summary()

# fit model
for i in range(25000):
	X, y = get_samples()
	model.fit(X, y, epochs=1, verbose=2, shuffle=False)

# generate new shapes from scratch
rect = generate_rectangle(model)
plot_rectangle(rect)