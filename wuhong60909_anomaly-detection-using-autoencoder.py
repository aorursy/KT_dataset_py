# Importing Libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf

from tensorflow.python.client import device_lib

import time

import seaborn as sns

import matplotlib.gridspec as gridspec



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, average_precision_score, roc_curve, auc
tf.__version__
print(device_lib.list_local_devices())
tf.test.is_gpu_available(

    cuda_only = False,

    min_cuda_compute_capability = None)
# Importing dataset

dataset = pd.read_csv('../input/creditcardfraud/creditcard.csv')

dataset_X = dataset.iloc[:, 0:30]

dataset_y = dataset.iloc[:, 30]
dataset.info()
dataset.describe()
dataset.isnull().sum()
dataset.head()
# Select the features.

v_features = dataset.iloc[:, 0:30].columns



plt.figure(figsize = (12, 120))

gs = gridspec.GridSpec(30, 1)

for i, cn in enumerate(dataset[v_features]):

    ax = plt.subplot(gs[i])    

    sns.distplot(dataset[cn][dataset.Class == 0], bins = 50, label = 'Normal')

    sns.distplot(dataset[cn][dataset.Class == 1], bins = 50, label = 'Fraud')

    ax.set_xlabel('')

    ax.set_title('Histogram of feature: ' + str(cn))

    plt.legend()

plt.show()
X = dataset_X.iloc[:, 1:].values # Drop 'Time'

y = dataset_y.values
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)
# Feature Scaling: Only scale 'Amount'

sc = StandardScaler()

X_train[:, -1:] = sc.fit_transform(X_train[:, -1:])

X_test[:, -1:] = sc.transform(X_test[:, -1:])
X_train.shape
class MyModel(tf.keras.Model):

    def __init__(self, latent_dim):

        super(MyModel, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential(

            [

                tf.keras.layers.InputLayer(input_shape = (29, ), name = 'InputLayer'), 

                tf.keras.layers.Dense(100, kernel_initializer = 'uniform', activation = 'tanh', name = 'Encoder_1'), 

                tf.keras.layers.Dense(latent_dim, kernel_initializer = 'uniform', activation = 'tanh', name = 'Laten_Space')          

            ], 

            name = 'Encoder'

        )

        self.decoder = tf.keras.Sequential(

            [

                tf.keras.layers.InputLayer(input_shape = (latent_dim, )), 

                tf.keras.layers.Dense(100, kernel_initializer = 'uniform', activation = 'tanh'), 

                tf.keras.layers.Dense(29, kernel_initializer = 'uniform', activation = 'linear')

            ],

            name = 'Decoder'

        )

        self.AE_model = tf.keras.Model(inputs = self.encoder.input, 

                                       outputs = self.decoder(self.encoder.output), 

                                       name = 'Auto Encoder')

                

    def call(self, input_tensor):

        latent_space = self.encoder.output

        reconstruction = self.decoder(latent_space)

        AE_model = tf.keras.Model(inputs = self.encoder.input, outputs = reconstruction)        

        return AE_model(input_tensor) 

    

    def summary(self):

        return self.AE_model.summary()
model = MyModel(50)

model.summary()
model.compile(optimizer = 'adam', loss = 'mean_squared_error', experimental_run_tf_function = False)

tStart = time.time()

h = model.fit(X_train[y_train == 0, :], X_train[y_train == 0, :], 

              validation_data = (X_test[y_test == 0, :], X_test[y_test == 0, :]), 

              batch_size = 256, epochs = 10, verbose = 1)

tEnd = time.time()

print('It cost %0.2f seconds.' % (tEnd - tStart))
plt.figure()

plt.plot(h.history['loss'], label = 'loss')

plt.plot(h.history['val_loss'], label = 'val_loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
# Training set

train_mse = np.mean(np.power(model(X_train) - X_train, 2), axis = 1)

train_error = pd.DataFrame({'Reconstruction_error': train_mse, 

                            'True_class': y_train})
train_error.describe()
plt.figure(figsize = (12, 5))

plt.scatter(train_error.index[train_error['True_class'] == 0], 

            train_error[train_error['True_class'] == 0]['Reconstruction_error'], 

            s = 5, label = 'Normal')

plt.scatter(train_error.index[train_error['True_class'] == 1], 

            train_error[train_error['True_class'] == 1]['Reconstruction_error'], 

            s = 5, label = 'Fraud')

plt.xlabel('Index')

plt.ylabel('Mean Squared Error (MSE)')

plt.legend()

plt.show()
# Plotting the precision recall curve.

precision, recall, threshold = precision_recall_curve(train_error.True_class, train_error.Reconstruction_error)

f1_score = 2 * precision * recall / (precision + recall)

average_precision = average_precision_score(train_error.True_class, train_error.Reconstruction_error)



# Choosing the threshold to maximize the F1 score

max_f1 = f1_score[f1_score == max(f1_score)]

best_threshold = threshold[f1_score[1: ] == max_f1]
## Precision, Recall curve 

plt.figure(figsize = (12, 6))

plt.plot(threshold, precision[1: ], label = "Precision", linewidth = 3)

plt.plot(threshold, recall[1: ], label ="Recall", linewidth = 3)

plt.axvline(best_threshold, color = 'black', ls = '--', label = 'Threshold = %0.3f' % (best_threshold))

plt.ylim(0, 1.1)

plt.xlabel('Threshold')

plt.ylabel('Precision/ Recall')

plt.title('Precision and recall for different threshold values')

plt.legend(loc = 'upper right')



## F1 score curve

plt.figure(figsize = (12, 6))

plt.plot(threshold, f1_score[1: ], label = "F1_score", linewidth = 3, color = 'green')

plt.scatter(threshold[f1_score[1: ] == max_f1], max_f1, label = 'Max F1 score = %0.3f' % (max_f1), s = 50, color = 'red')

plt.axvline(best_threshold, color = 'black', ls = '--', label = 'Threshold = %0.3f' % (best_threshold))

plt.axhline(max_f1, color = 'black', ls = '-')

plt.ylim(0, 1.1)

plt.xlabel('Threshold')

plt.ylabel('F1 score')

plt.title('F1 score for different threshold values')

plt.legend(loc = 'upper right')



plt.show()

print('Best threshold = %f' % (best_threshold))

print('Max F1 score = %f' % (max_f1))
## Recall - Precision curve

plt.figure(figsize = (12, 6))

f_scores = np.linspace(0.2, 0.8, num = 4)



for f_score in f_scores:

    x = np.linspace(0.001, 1)

    y = f_score * x / (2 * x - f_score)

    plt.plot(x[y >= 0], y[y >= 0], color = 'gray', alpha = 0.2)

    plt.annotate('F1 = {0:0.2f}'.format(f_score), xy = (0.95, y[45] + 0.02))



plt.plot(recall[1: ], precision[1: ], label = 'Area = %0.3f' % (average_precision), linewidth = 3)

plt.scatter(recall[f1_score == max_f1], precision[f1_score == max_f1], label = 'F1 score = %0.3f' % (max_f1), s = 50, color = 'red')

plt.axvline(recall[f1_score == max_f1], color = 'black', ls = '--', label = 'Recall = %0.3f' % (recall[f1_score == max_f1]))

plt.axhline(precision[f1_score == max_f1], color = 'black', ls = '-', label = 'Precision = %0.3f' % (precision[f1_score == max_f1]))

plt.ylim(0, 1.1)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision - Recall curve')

plt.legend(loc = 'upper right')



plt.show()
## Training Set

plt.figure(figsize = (12, 5))

plt.scatter(train_error.index[train_error['True_class'] == 0], 

            train_error[train_error['True_class'] == 0]['Reconstruction_error'], 

            s = 5, label = 'Normal')

plt.scatter(train_error.index[train_error['True_class'] == 1], 

            train_error[train_error['True_class'] == 1]['Reconstruction_error'], 

            s = 5, label = 'Fraud')

plt.axhline(best_threshold, color = 'red', label = 'Threshold = %0.3f' % (best_threshold))

plt.xlabel('Index')

plt.ylabel('Mean Squared Error (MSE)')

plt.title('Training Set')

plt.legend()

plt.show()

print('Best threshold = %f' % (best_threshold))
# Create AE predictor

def AE_predictor(X, model, threshold):

    X_valid = model(X)

    mse = np.mean(np.power(X_valid - X, 2), axis = 1)

    y = np.zeros(shape = mse.shape)

    y[mse > threshold] = 1

    return y
y_pred = AE_predictor(X = X_test, model = model, threshold = best_threshold)

cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))



plt.figure()

sns.heatmap(cm, cmap = "coolwarm", annot = True, linewidths = 0.5)

plt.title("Confusion Matrix")

plt.xlabel("Predicted class")

plt.ylabel("Real class")

plt.show()
fpr, tpr, thresholds = roc_curve(train_error.True_class, train_error.Reconstruction_error)

roc_auc = auc(fpr, tpr)
## ROC curve

plt.figure(figsize = (8, 5))

plt.plot(fpr, tpr, linewidth = 3, label = 'AUC = %0.3f' % (roc_auc))

plt.plot([0, 1], [0, 1], linewidth = 3)

plt.xlim(left = -0.02, right = 1)

plt.ylim(bottom = 0, top = 1.02)

plt.xlabel('False Positive Rate (FPR)')

plt.ylabel('True Positive Rate (TPR)')

plt.title('Receiver operating characteristic curve (ROC)')

plt.legend()

plt.show()