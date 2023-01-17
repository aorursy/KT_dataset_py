!pip install skorch
!pip list
!ls /kaggle/input/isae-ssl-hackathon-2020/
import os

import random

import datetime
import pandas as pd

import matplotlib.cm

import numpy as np

import scipy

import sklearn

import sklearn.metrics

import sklearn.metrics

import sklearn.preprocessing

import skorch.dataset

import skorch.helper

import torch

import torch.nn as nn

import torch.optim as optim

from matplotlib import pyplot as plt

import tqdm

import skimage.exposure
random.seed(2020)

np.random.seed(2020)

torch.manual_seed(2020)
TRAIN_DATA_URL = "/kaggle/input/isae-ssl-hackathon-2020/eurosat_train.npz"



VALID_DATA_URL = "/kaggle/input/isae-ssl-hackathon-2020/eurosat_valid.npz"



CLASSES = [

    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential',

    'River', 'SeaLake'

]



DATASOURCE = np.DataSource(None)
train_dataset = DATASOURCE.open(TRAIN_DATA_URL, "rb")

train_dataset = np.load(train_dataset)



valid_dataset = DATASOURCE.open(VALID_DATA_URL, "rb")

valid_dataset = np.load(valid_dataset)
x_train, y_train = train_dataset['x'], train_dataset['y']

x_valid, y_valid = valid_dataset['x'], valid_dataset['y']
print(x_train.shape, y_train.shape)

print(x_valid.shape, y_valid.shape)
x_train = x_train.astype(np.float32) / 255.

x_train = x_train.transpose((0, 3, 1, 2))



x_valid = x_valid.astype(np.float32) / 255.

x_valid = x_valid.transpose((0, 3, 1, 2))



# Helper in skorch to use a custom validation dataset instead of splitting the train set

# https://skorch.readthedocs.io/en/stable/user/FAQ.html#i-already-split-my-data-into-training-and-validation-sets-how-can-i-use-them



valid_ds = skorch.dataset.Dataset(x_valid, y_valid)

train_split = skorch.helper.predefined_split(valid_ds)
def init_weights(m):

    if type(m) == nn.Conv2d:

        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        m.bias.data.fill_(0.1)

    if type(m) == nn.Linear:

        # apply a uniform distribution to the weights and a bias=0

        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        m.bias.data.fill_(0.1)





model = nn.Sequential(

    nn.Conv2d(3, 32, 3, padding=1),

    nn.BatchNorm2d(32),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1),

    nn.BatchNorm2d(64),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Conv2d(64, 64, 3, padding=1),

    nn.BatchNorm2d(64),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(8 * 8 * 64, 256),

    nn.BatchNorm1d(256),

    nn.ReLU(),

    nn.Linear(256, 128),

    nn.BatchNorm1d(128),

    nn.ReLU(),

    nn.Linear(128, 10),

    nn.Softmax(dim=-1),

)



model.apply(init_weights)
lr_scheduler = skorch.callbacks.LRScheduler(policy=torch.optim.lr_scheduler.ReduceLROnPlateau,

                                            factor=0.1,

                                            patience=3)



early_stopping = skorch.callbacks.EarlyStopping(patience=5)



checkpoint_dir = './tmp'

os.makedirs(checkpoint_dir, exist_ok=True)



cp = skorch.callbacks.Checkpoint(dirname=checkpoint_dir)

train_end_cp = skorch.callbacks.TrainEndCheckpoint(dirname=checkpoint_dir)



lr = 0.01

batch_size = 32

max_epochs = 10



net = skorch.NeuralNetClassifier(

    model,

    batch_size=batch_size,

    max_epochs=max_epochs,

    lr=lr,

    train_split=train_split,

    # Shuffle training data on each epoch

    iterator_train__shuffle=True,

    device="cuda" if torch.cuda.is_available() else "cpu",

    criterion=nn.modules.loss.NLLLoss,

    optimizer=optim.Adam,

    callbacks=[lr_scheduler, early_stopping, cp, train_end_cp])
net.fit(x_train,y_train)
y_pred = net.predict(x_valid)

print(sklearn.metrics.classification_report(y_valid, y_pred, digits=2))

acc = sklearn.metrics.accuracy_score(y_valid, y_pred)
y_pred_proba = net.predict_proba(x_valid)
y_valid_onehot = sklearn.preprocessing.label_binarize(y_valid, range(10))
# Compute ROC curve and ROC area for each class



def plot_roc_curve(y_test, y_pred, classes):

    n_classes = len(classes)



    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    for i in range(10):

        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_pred[:, i])

        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])



    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(), y_pred.ravel())

    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])



    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



    # Then interpolate all ROC curves at this points

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])



    # Finally average it and compute AUC

    mean_tpr /= n_classes



    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves

    plt.figure(figsize=(10, 10))

    plt.plot(fpr["micro"], tpr["micro"],

             label='micro-average ROC curve (area = {0:0.2f})'

                   ''.format(roc_auc["micro"]),

             color='deeppink', linestyle=':', linewidth=4)



    plt.plot(fpr["macro"], tpr["macro"],

             label='macro-average ROC curve (area = {0:0.2f})'

                   ''.format(roc_auc["macro"]),

             color='navy', linestyle=':', linewidth=4)



    #     colors = itertools.cycle(['aqua', 'coral', 'gold', 'ivory', 'green', 'blue', 'cyan'])

    colors = matplotlib.cm.get_cmap('Pastel1')

    colors = colors(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):

        plt.plot(fpr[i], tpr[i], color=color, lw=2,

                 label='ROC curve of class {0} (area = {1:0.2f})'

                       ''.format(classes[i], roc_auc[i]))



    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Some extension of Receiver operating characteristic to multi-class')

    plt.legend(loc="lower right")

    plt.show()
plot_roc_curve(y_valid_onehot, y_pred_proba, CLASSES)
# get and save torch model

model = net.module
xp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M")

torch.save(model, "/kaggle/working/model_{}.pt".format(xp_id))
TEST_URL = "/kaggle/input/isae-ssl-hackathon-2020/test.npz"
test_dataset = DATASOURCE.open(TEST_URL, 'rb')

test_dataset = np.load(test_dataset)

x_test = test_dataset['x']
def plot_imgs(x, y=None, grid_size=4, title="samples"):

    """

    Plot grid_size*grid_size images 

    """

    fig, ax = plt.subplots(grid_size, grid_size, figsize=(20, 20))

    fig.tight_layout()

    idxs = np.random.randint(len(x), size=16)



    for i in range(grid_size ** 2):

        k = idxs[i]

        if y is not None:

            img, lbl = x[k], CLASSES[y[k]]

        else:

            img, lbl = x[k], "unlabelled"

        img = skimage.exposure.adjust_gamma(img, gamma=0.7)

        ax[i % 4][i // 4].imshow(img)

        ax[i % 4][i // 4].set_title(lbl)

        ax[i % 4][i // 4].axis('off')

    fig.suptitle(title, fontsize=14)

    plt.show()
plot_imgs(x_test, y=None, title="test Dataset")
model = torch.load("/kaggle/working/model_{}.pt".format(xp_id))
def predict(image):

    """

    Fonction de prédiction qui renvoie la classe (index) prédite par le modèle

    """



    # preprocessing sur l'image

    x = image.astype(np.float32) / 255.

    x = np.transpose(x, (2, 0, 1))

    x = x[None, :, :]



    # prédiction (index !)

    y_pred = model(torch.tensor(x))

    y_pred = y_pred.detach().numpy()



    y_pred = np.argmax(y_pred)

    return y_pred



def make_submission(predictions):

    """

    Génère un dataframe de soumission pour la compétition

    predictions est une liste de couples (idx, class_index)

    Exemple [(0,1),(1,9),(2,4)]"""

    df = []

    for idx, y_pred in predictions:

        cls_str = CLASSES[y_pred]

        df.append({"Id": idx, "Category": cls_str})

    df = sorted(df, key=lambda x: x["Id"])

    df = pd.DataFrame(df)

    return df

predictions = []



for idx, img in enumerate(tqdm.tqdm(x_test)):

    y_pred = predict(img)

    predictions.append((idx, y_pred))



submission_csv = make_submission(predictions)

sub_id = "submission_{}.csv".format(xp_id)

# Sauvegarde du dataframe en csv. ATTENTION: index=False sinon BUG !

submission_csv.to_csv('/kaggle/working/{}.csv'.format(sub_id), index=False)