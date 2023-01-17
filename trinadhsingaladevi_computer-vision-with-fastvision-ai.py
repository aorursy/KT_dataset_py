from fastai.vision import *
import pandas as pd
df = pd.read_csv('../input/hp-2020/jh_2020/train.csv')
test_df = pd.read_csv('../input/hp-2020/jh_2020/test.csv')
df.head()

tfms = get_transforms(do_flip = True,flip_vert= True, max_rotate= 50, max_lighting= 0.1, max_warp = 0)
data = ImageDataBunch.from_df('../input/hp-2020/jh_2020/images', df,ds_tfms=tfms,label_delim= None,valid_pct=0.2,fn_col=0, label_col=1 , size=299,bs=64).normalize(imagenet_stats)
data.path = pathlib.Path('.')
#data.normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,7))

print(data.classes)
learn =cnn_learner(data,models.resnet50,pretrained=True,metrics=[accuracy])

learn.fit_one_cycle(4)

#learn.save("fastai_model")
#learn = learn.load("fastai_model")
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(7,7))


interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
learn.unfreeze()

learn.fit_one_cycle(3)
#learn.save("../input/output/fastai_model_unfreeze")
#learn = learn.load("fastai_model_unfreeze")

# #print(os.listdir("../input/fastai-pretrained-models"))
# PATH = "../input/emergency-vehicle-detection"
# !mkdir -p /root/.cache/torch/checkpoints/
# !cp ../input/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth

#learn.model_dir('/root/.cache/torch/checkpoints/resnet50-19c8e357.pth')
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(20,max_lr=slice(1e-4,1e-3))

learn.freeze()
learn.save('car_classification1')
path = '../input/hp-2020/jh_2020/images/'
emergency_or_not = []
for i in test_df['image_names']:
    img = open_image(path + i)
    pred_class,pred_idx, outputs = learn.predict(img)
    emergency_or_not.append(pred_class.obj)
test_df['emergency_or_not'] = emergency_or_not
test_df.to_csv('fastai1.csv', index = False)
