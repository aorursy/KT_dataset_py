import sys
import pathlib
import numpy as np
import shutil

from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.callbacks import TensorBoard, ModelCheckpoint

this = sys.modules[__name__]
this.logDir = None
this.model = None
this.x_train, this.x_test, this.y_train, this.y_test = None, None, None, None

def reset():
    try:
        shutil.rmtree(str(this.logDir))
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

def load(cls, *layers, **compile):
	this.model = cls(layers)
	this.model.summary()
	ckPoint = this.logDir.joinpath('checkpoint.ckpt')
	exists = ckPoint.exists()

	if exists:
		this.model.load_weights(str(ckPoint))

	this.model.compile(**compile)

	return exists

def train(batchSize, epochs=1000):
	modelCheckpoint = ModelCheckpoint(
		str(this.logDir.joinpath('checkpoint.ckpt')), save_weights_only=True, verbose=1
	)

	tensorboard = TensorBoard(
		log_dir=str(this.logDir),
		histogram_freq=1,
		embeddings_freq=1,
		embeddings_layer_names=['features'],
		embeddings_metadata='metadata.tsv',
		embeddings_data=this.x_test,
		batch_size=batchSize,
		write_graph=True,
		write_grads=True
	)

	this.model.fit(
		this.x_train, this.y_train,
		batch_size=batchSize,
		epochs=epochs,
		callbacks=[modelCheckpoint, tensorboard],
		verbose=1,
		validation_data=(this.x_test, this.y_test)
	)

def evaluate():
	loss, accuracy = this.model.evaluate(this.x_test, this.y_test, verbose=0)
	print('\nResults:')
	print(f' > Accuracy: {round(accuracy, 3)*100}%')
	print(f' > Error: {round(loss, 3)}\n')

def init(name, x, y, outputDim, y_categorical=True):

	this.logDir = pathlib.Path('./models', name)
	if not this.logDir.exists():
		this.logDir.mkdir(parents=True)

	scaler = MinMaxScaler(feature_range=(0, 1))
	x = scaler.fit_transform(x)

	this.x_train, this.x_test, this.y_train, this.y_test = train_test_split(
		np.array(x), np.array(y), test_size=0.35
	)

	with open(str(this.logDir.joinpath('metadata.tsv')), 'w') as f:
		np.savetxt(f, this.y_test)

	if y_categorical:
		this.y_train = to_categorical(this.y_train, outputDim)
		this.y_test = to_categorical(this.y_test, outputDim)

	if not this.logDir.exists():
		this.logDir.mkdir(parents=True)

from keras import Sequential
from keras.layers import Dense
from keras import optimizers, losses
import sqlite3

inputDim = 4
outputDim = 3
batchSize = 100

### DATA ##########################
x, y = [], []
flowers = {'Iris-virginica': 0, 'Iris-setosa': 1, 'Iris-versicolor': 2}
conn = sqlite3.connect('../input/database.sqlite')
for row in conn.execute('SELECT * FROM Iris'):
	x.append(list(row[1:-1]))
	y.append(flowers[row[-1]])
### DATA ##########################

init('iris', x, y, outputDim, y_categorical=True)
reset()
exists = load(
	Sequential,

	Dense(10, activation='tanh', input_dim=inputDim, name='features'),
	Dense(20, activation='tanh'),
	Dense(10, activation='tanh'),
	Dense(outputDim, activation='softmax'),

	optimizer=optimizers.Adam(),
	loss=losses.categorical_crossentropy,
	metrics=['accuracy']
)

print(f'\n > Models exists: {exists}')
train(batchSize, epochs=600)
evaluate()