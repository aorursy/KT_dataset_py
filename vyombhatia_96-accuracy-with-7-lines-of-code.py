# Importing from the Libary:
from fastai.vision import *
from fastai.metrics import *
# Here's the path for our file:
path = "../input/bee-vs-wasp/kaggle_bee_vs_wasp"

# Defining the Data:
data = ImageDataBunch.from_folder(path, ds_tfms = get_transforms(), valid_pct = 0.1, size = 512, bs = 8)
# Visualizing:
data.show_batch(row = 3)
# Building the model through transfer learning:
model = cnn_learner(data, models.densenet121, metrics = accuracy ,model_dir = "/tmp/model/")
# Training the model:
model.fit_one_cycle(5)

# Lets try to predict now since I am confident that this bad boy won't disappoint me:

img = open_image("../input/bee-vs-wasp/kaggle_bee_vs_wasp/bee1/10007154554_026417cfd0_n.jpg")

# Showing you the image:
img.show()
prediction, idx, probability = model.predict(img)
print("The insect is a", prediction, ". The Probabilty being:", max(probability.numpy()))