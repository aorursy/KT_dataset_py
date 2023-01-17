from fastai.vision import *
from fastai.metrics import *
data = ImageDataBunch.from_folder(Path("../input/intel-image-classification/seg_train"),
                                  train = Path("../input/intel-image-classification/seg_train/seg_train/"),
                                  valid = Path("../input/intel-image-classification/seg_test/seg_test/"),
                                 ds_tfms = get_transforms(flip_vert = False),
                                 bs = 8, size = 512, valid_pct = 0.1)
data.show_batch(3)
learner = cnn_learner(data, models.densenet121, model_dir = "/model/tmp/ ", metrics = accuracy)
learner.fit_one_cycle(5)
learner.lr_find()
learner.recorder.plot()
im = open_image("../input/intel-image-classification/seg_train/seg_train/buildings/10165.jpg")
preds, x, probs = learner.predict(im)
im.show()
print("The Class is:",preds)
im = open_image("../input/intel-image-classification/seg_pred/seg_pred/10199.jpg")
preds, x, probs = learner.predict(im)
im.show()
print("The Class is:",preds)