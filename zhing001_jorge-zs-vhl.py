#Allows you to save your models somewhere without

# copying all the data over.

# Trust me on this.



!mkdir input

!cp /kaggle/input/train_labels.csv input

!cp /kaggle/input/sample_submission.csv input

!ln -s /kaggle/input/train/ input/train

!ln -s /kaggle/input/test/ input/test
from fastai.vision import *
db = (ImageItemList.from_csv(csv_name='train_labels.csv', path='input', folder='train', suffix='.tif')

        .random_split_by_pct()

        .label_from_df()

        .transform(get_transforms(flip_vert=True), size=64)

        .add_test_folder('test')

        .databunch(bs=32)

        .normalize(imagenet_stats))
db.show_batch(4, figsize=(12,12), ds_type=DatasetType.Test)
learn = create_cnn(db, models.resnet34, metrics=[error_rate])
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(3, 1e-02)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(4)
probs, _ = learn.get_preds(DatasetType.Test)
preds = probs[:,1]
test_df = pd.read_csv('./input/sample_submission.csv')

test_df['id'] = [i.stem for i in db.test_ds.items]

test_df['label'] = preds
test_df.to_csv('submission.csv', index=False)