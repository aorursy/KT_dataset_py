import tensorflow as tf
import numpy as np
import tqdm
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


cnn_none = tf.keras.models.load_model("../input/temp22/100dCNN-none.h5")
cnn_fast = tf.keras.models.load_model("../input/temp22/100dCNN-FastText (1).h5")
cnn_glove = tf.keras.models.load_model("../input/temp22/100dCNN-glove (1).h5")


lstm_none = tf.keras.models.load_model("../input/tempdata/100dLSTM-nothing.h5")
lstm_fast = tf.keras.models.load_model("../input/tempdata/100dLSTM-FastText.h5")
lstm_glove = tf.keras.models.load_model("../input/tempdata/100dLSTM-glove.h5")


lstmcnn_none = tf.keras.models.load_model("../input/tempdata/100dLSTMCNN-nothing.h5")
lstmcnn_fast = tf.keras.models.load_model("../input/tempdata/100dCNNLSTM-FastText.h5")
lstmcnn_glove = tf.keras.models.load_model("../input/tempdata/100dLSTMCNN-glove.h5")

models = [cnn_none, cnn_fast, cnn_glove, lstm_none, lstm_fast, lstm_glove, lstmcnn_none, lstmcnn_fast, lstmcnn_glove]
for model in models:
    model.summary()
data = pd.DataFrame()
data["models"] = ["cnn","cnn f", "cnn g", "lstm", "lstm f", "lstm g", "lc", "lc f", "lc g"]
data
x = np.load("../input/tempdata/testx.npy")
y = np.load("../input/tempdata/testy.npy")

def pred(model, x):
    p = []
    for i in tqdm.tqdm(x):
        try:
            p.append(cnn_none.predict(np.array([i])))
        except:
            p.append(0)
        
    return p
c = np.round(cnn_none.predict(x[500:2500]))
c_f = np.round(cnn_fast.predict(x[500:2500]))
c_g = np.round(cnn_glove.predict(x[500:2500]))

l = np.round(lstm_none.predict(x[500:2500]))
l_f = np.round(lstm_fast.predict(x[500:2500]))
l_g = np.round(lstm_glove.predict(x[500:2500]))

lc = np.round(lstmcnn_none.predict(x[500:2500]))
lc_f = np.round(lstmcnn_fast.predict(x[500:2500]))
lc_g = np.round(lstmcnn_glove.predict(x[500:2500]))
i=0
recall = []
precision = []
fscore = []
acc = []
for m in models:
    z =  np.round(m.predict(x[5000:20000]))
    r, p, f, _ = precision_recall_fscore_support(y[5000:20000], z)
    acc.append(accuracy_score(y[5000:20000], z))
    recall.append(r)
    precision.append(p)
    fscore.append(f)
    print(i)
    i+=1
data['recall'] = recall
data['precision'] = precision
data['fscore'] = fscore
data['acc'] = acc
data
