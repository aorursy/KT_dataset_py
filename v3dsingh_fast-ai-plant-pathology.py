from fastai.vision import *
from fastai.metrics import error_rate
torch.cuda.set_device(0)
torch.cuda.is_available()
Image_Data_Path = "../input/plant-pathology-2020-fgvc7/images/"

train_data = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test_df = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
sub_data = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

train_data["image_id"] = train_data["image_id"] + ".jpg"
test_df["image_id"] = test_df["image_id"] + ".jpg"
train_data["label"] = (0*train_data.healthy + 1*train_data.multiple_diseases+
             2*train_data.rust + 3*train_data.scab)
train_data.drop(columns=["healthy","multiple_diseases","rust","scab"],inplace=True)

train_data.head(2)
test_data = ImageList.from_df(test_df, Image_Data_Path)
np.random.seed(42)
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_df(Image_Data_Path, train_data, ds_tfms=tfms, size = 224, bs=32).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(15,15))
data.add_test(test_data)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(2)
learn.model_dir = '/kaggle/working/models'
learn.lr_find()
learn.recorder.plot()
learn.save('stage-1')
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(3e-3,1e-2))
learn.save('stage-1-1')
learn.load('stage-1')
learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(3e-4,1e-2))
learn.save('stage-1-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')
interp.plot_top_losses(9, figsize=(15,11), heatmap=True)
preds, y = learn.get_preds(DatasetType.Test)
preds_np = preds.numpy()
px = pd.DataFrame(preds_np)
sub_data.healthy = px[0]
sub_data.multiple_diseases = px[1]
sub_data.rust = px[2]
sub_data.scab = px[3] 
sub_data.head()
sub_data.to_csv('submission1.csv', index=False)
