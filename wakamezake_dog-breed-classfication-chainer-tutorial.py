!curl https://colab.chainer.org/install | sh -
!python -c 'import chainer; chainer.print_runtime_info()'
!apt -y -qq install graphviz > /dev/null 2> /dev/null

!pip install pydot
%ls ../input
%ls ../input/resnet-50-caffe/
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import chainer

import cv2

import chainer.functions as F

import chainer.links as L

from PIL import Image

from sklearn.preprocessing import LabelEncoder

from pathlib import Path

from chainer import optimizers

from chainer import training
data_path = Path("../input/dog-breed-identification/")
labels = pd.read_csv(data_path.joinpath("labels.csv"))

submit = pd.read_csv(data_path.joinpath("sample_submission.csv"))
print("labels'shape : {}".format(labels.shape))

print("unique label : {}".format(len(labels["breed"].unique())))
labels.head()
sns.countplot(data=labels, x="breed")
def load_img_shapes(path_to_img):

    return cv2.imread(path_to_img).shape
# reference

# https://www.kaggle.com/enerrio/data-exploration-distribution-of-image-sizes

def distribution_img_shapes(path):

    shapes = []

    for img_path in path.glob("*.jpg"):

        shapes.append(load_img_shapes(str(img_path)))

    df = pd.DataFrame({'Shapes': shapes})

    shape_counts = df['Shapes'].value_counts()

    print("Image Shapes:")

    for i in range(len(shape_counts)):

        print("Shape %s counts: %d" % (shape_counts.index[i],

                                       shape_counts.values[i]))
# distribution_img_shapes(data_path.joinpath("train"))
# distribution_img_shapes(data_path.joinpath("test"))
le = LabelEncoder()

labels["breed"] = le.fit_transform(labels["breed"])
labels.head()
# sample

labels.loc[labels["id"] == "000bec180eb18c7604dcecc8fe0dba07", "breed"]

labels.loc[labels["id"] == "000bec180eb18c7604dcecc8fe0dba07", "breed"].values[0]
train_image_files = []

for img_path in data_path.joinpath("train").glob("*.jpg"):

    label = labels.loc[labels["id"] == img_path.stem, "breed"].values[0]

    train_image_files.append((str(img_path), label.astype(np.int32)))
train_image_files[0]
type(train_image_files[0][0]), type(train_image_files[0][1])
seed = 42

epochs = 10

batch_size = 32

loaderjob = 4
train_dataset = chainer.datasets.LabeledImageDataset(train_image_files)
def _transform(data):

    img, label = data

    img = img[:3, ...]

    img = _resize(img.astype(np.uint8))

    img = img / 255.0

    img = img.astype(np.float32)

    return img, label



def _resize(img,width=224, height=224):

    img = Image.fromarray(img.transpose(1, 2, 0))

    img = img.resize((width, height), Image.BICUBIC)

    return np.asarray(img).transpose(2, 0, 1)
transform_train_dataset = chainer.datasets.TransformDataset(train_dataset, _transform)
train, val = chainer.datasets.split_dataset_random(transform_train_dataset, 9000, seed=seed)
len(train), len(val)
train_iter = chainer.iterators.SerialIterator(train, batch_size)

val_iter = chainer.iterators.SerialIterator(val, batch_size, repeat=False)
class_size = len(labels["breed"].unique())

print(class_size)
class PretrainedResNet50(chainer.Chain):

    def __init__(self, class_labels):

        super(PretrainedResNet50, self).__init__()



        with self.init_scope():

            self.base = L.ResNet50Layers()

            self.fc6 = L.Linear(None, class_labels)



    def forward(self, x):

        h = self.base(x, layers=['pool5'])['pool5']

        return self.fc6(h)
import os

to_path = "~/.chainer/dataset/pfnet/chainer/models"

os.makedirs(os.path.dirname(filename), exist_ok=True)
%cp ../input/resnet-50-caffe/ResNet-50-model.caffemodel  $to_path
%ls $filename
model = PretrainedResNet50(class_labels=class_size)

classifier_model = L.Classifier(model)
optimizer = optimizers.SGD()

optimizer.setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (epochs, 'epoch'),

                           out='out')
from chainer.training import extensions



trainer.extend(extensions.LogReport())

trainer.extend(extensions.PrintReport(['epoch',

                                       'main/loss',

                                       'main/accuracy',

                                       'validation/main/loss',

                                       'validation/main/accuracy',

                                       'elapsed_time']))

trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],

                                     x_key='epoch',

                                     file_name='loss.png'))

trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],

                                     x_key='epoch',

                                     file_name='accuracy.png'))

trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))

trainer.extend(extensions.snapshot_object(classifier_model.predictor, filename='model_epoch-{.updater.epoch}'))

# trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=0))

trainer.extend(extensions.dump_graph('main/loss'))
trainer.run()