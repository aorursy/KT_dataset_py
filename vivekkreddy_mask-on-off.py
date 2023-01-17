# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#Path Cleaning
path = Path('/kaggle/input')
path_imgs = path/'face-mask-detection-data'
fnames0 = get_image_files(path_imgs/'with_mask')
fnames1 = get_image_files(path_imgs/'without_mask')
fnames = fnames0 + fnames1

#DataBunch creation
data = ImageDataBunch.from_name_func(path_imgs, fnames, ds_tfms=get_transforms(), size=224, label_func = lambda x: 'with_mask' if '/with_mask/' in str(x) else 'without_mask')
data.classes


# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Resnet Training
learn= cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
#Save Model
learn.model_dir=Path('/kaggle/working')
learn.save('stage-1')
# learn.load('/kaggle/input/covid-mask-detectionv1resnet34/stage-1')
# Export Model
# learn.export(file = Path("/kaggle/working/FaceMask-stage-best.pkl"))
# learn.layer_groups
learn.unfreeze()
learn.data = data
learn.fit_one_cycle(1)
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(7e-3,1e-2))
# from pathlib import Path
# importi=Path('../input/covid-mask-detectionv1resnet34')
# learn = load_learner(importi, 'FaceMask-stage-1.pkl')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
learn.freeze()
learn.save('stage-2')
# learn.export(file = Path("/kaggle/working/FaceMask-stage-2.pkl"))
learn.show_results(ds_type=DatasetType.Valid)