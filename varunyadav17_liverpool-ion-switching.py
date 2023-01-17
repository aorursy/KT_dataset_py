import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Bidirectional

from keras.layers import GRU

from keras import callbacks



from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split



from plotly import tools

import plotly.graph_objs as go

import plotly.express as px



from plotly.subplots import make_subplots

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

submission_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
train_df.head()
train_df.shape
train_df.tail()
def redcolor(val):

    color = 'red' if val < 0 else 'black'

    return 'color: %s' % color
train_df.describe().round(4).style.applymap(redcolor)
test_df.head()
test_df.shape
test_df.tail()
test_df.describe().round(4).style.applymap(redcolor)
submission_df.head()
submission_df.shape
submission_df.tail()
fig = make_subplots(rows = 5, cols = 2)

graphno = 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'], 

                     marker = dict(color = 'plum'), name = 'First Batch'), 

                     row = 1, col = 1)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'purple'), name = 'Second Batch'), 

                     row = 1, col = 2)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'blue'), name = 'Third Batch'), 

                     row = 2, col = 1)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'blueviolet'), name = 'Fourth Batch'), 

                     row = 2, col = 2)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'crimson'), name = 'Fifth Batch'), 

                     row = 3, col = 1)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'cyan'), name = 'Sixth Batch'), 

                     row = 3, col = 2)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'darkmagenta'), name = 'Seventh Batch'), 

                     row = 4, col = 1)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'darkolivegreen'), name = 'Eigth Batch'), 

                     row = 4, col = 2)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'mediumturquoise'), name = 'Ninth Batch'), 

                     row = 5, col = 1)

graphno += 1



fig.add_trace(go.Scatter(x = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = train_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'salmon'), name = 'Tenth Batch'), 

                     row = 5, col = 2)

graphno += 1



fig.update_layout(height = 1000, width = 850, title_text=f'Training Data Signal Graphs')

fig.show()
fig = make_subplots(rows = 2, cols = 2)

graphno = 1



fig.add_trace(go.Scatter(x = test_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = test_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'], 

                     marker = dict(color = 'plum'), name = 'First Batch'), 

                     row = 1, col = 1)

graphno += 1



fig.add_trace(go.Scatter(x = test_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = test_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'purple'), name = 'Second Batch'), 

                     row = 1, col = 2)

graphno += 1



fig.add_trace(go.Scatter(x = test_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = test_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'blue'), name = 'Third Batch'), 

                     row = 2, col = 1)

graphno += 1



fig.add_trace(go.Scatter(x = test_df.iloc[500000 * (graphno - 1):500000 * graphno]['time'], 

                     y = test_df.iloc[500000 * (graphno - 1):500000 * graphno]['signal'],

                     marker = dict(color = 'blueviolet'), name = 'Fourth Batch'), 

                     row = 2, col = 2)



fig.update_layout(height = 400, width = 850, title_text=f'Test Data Signal Graphs')

fig.show()
fig = px.box(train_df.iloc[::100, :], x='open_channels', y='signal', color='open_channels', title='Signal vs Channels')

fig.update_traces(quartilemethod='exclusive')

fig.show()
fig = px.bar(train_df, x = list(range(11)), y = train_df["open_channels"].value_counts(sort = False).values) 

fig.update_layout(title = "Open Channels Distribution", xaxis_title = "Open Channel Number", yaxis_title = "Frequency",

                  font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))

fig.show()
i = 0

fig = make_subplots(rows = 5, cols = 2, subplot_titles = ["Batch {}".format(i) for i in range(1, 11)])

for idxRow in range(5):

    for idxCol in range(2):

        data = train_df[500000 * i: 500000 * (i + 1)]["open_channels"].value_counts(sort = False).values

        fig.add_trace(go.Bar(x = list(range(11)), y = data), row = idxRow + 1, col = idxCol + 1)

        i += 1

fig.update_layout(height = 1400, width = 800, title = "Open Channel Distribution for each batch", showlegend = False)

fig.show()
X = train_df.signal.values.reshape(-1, 500, 1)

Y = train_df.open_channels.values.reshape(-1, 500, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

test_signal_input = test_df.signal.values.reshape(-1, 500, 1) 
model = Sequential()

model.add(Dense(256, input_shape = (500, 1), activation = "linear"))

model.add(Bidirectional(GRU(256, return_sequences = True)))

model.add(Bidirectional(GRU(256, return_sequences = True)))

model.add(Dense(11, activation = 'softmax'))



model.compile('adam', loss = 'sparse_categorical_crossentropy')

model.summary()
class F1ScoreCallbacks(callbacks.Callback):

    def __init__(self, X, Y):

        super().__init__()

        self.X = X

        self.Y = Y.reshape(-1)

    def on_epoch_begin(self, epoch, logs = None):

        if epoch == 0:

            return

        else:

            prediction = (model.predict(self.X, batch_size = 64).argmax(axis = -1).reshape(-1))

            score = f1_score(self.Y, prediction, average = 'macro')

            print(f'F1 Score = {score:.4f}')
model.fit(X_train, Y_train, batch_size = 64, epochs = 20, callbacks = [callbacks.ReduceLROnPlateau(), 

                                                                       F1ScoreCallbacks(X_test, Y_test), 

                                                                       callbacks.ModelCheckpoint('savedModel.h5')], 

                                                                       validation_data = (X_test, Y_test))

model.load_weights('savedModel.h5')

predictions = model.predict(X_test, batch_size = 64).argmax(axis = -1)

f1_score(Y_test.reshape(-1), predictions.reshape(-1), average = 'macro')
final_predictions = model.predict(test_signal_input, batch_size = 64).argmax(axis = -1)

submission_df.open_channels = final_predictions.reshape(-1)

submission_df.to_csv('submission.csv', index = False, float_format='%.4f')