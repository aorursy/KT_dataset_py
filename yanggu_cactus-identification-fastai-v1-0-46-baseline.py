import fastai

from fastai.vision import *
# Copy pretrained model weights to the default path

!mkdir '/tmp/.torch'

!mkdir '/tmp/.torch/models/'

!cp '../input/resnet50/resnet50.pth' '/tmp/.torch/models/resnet50-19c8e357.pth'
fastai.__version__
data_path = Path('../input/aerial-cactus-identification')

df = pd.read_csv(data_path/'train.csv')

df.head()
sub_csv = pd.read_csv(data_path/'sample_submission.csv')

sub_csv.head()
test = ImageList.from_df(sub_csv, path=data_path/'test', folder='test')

data = (ImageList.from_df(df, path=data_path/'train', folder='train')

        .random_split_by_pct(0.2)

        .label_from_df()

        .add_test(test)

        .transform(get_transforms(flip_vert=True), size=128)

        .databunch(path='.', bs=64)

        .normalize(imagenet_stats)

       )
learn = create_cnn(data, models.resnet50, metrics=[accuracy])
# learn.data.show_batch()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(6, slice(3e-2))
learn.save('stage-1')
learn.unfreeze()
learn.fit_one_cycle(3, slice(3e-2/100, 3e-2/10))
learn.save('stage-2')
valid_preds = learn.get_preds()
preds = learn.TTA(ds_type=DatasetType.Test)
sub_csv['has_cactus'] = preds[0].argmax(1)
sub_csv.to_csv('submission.csv', index=False)