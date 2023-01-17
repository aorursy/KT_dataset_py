import numpy as np 

import pandas as pd



from PIL import Image



import torch

import torch.nn as nn

import torch.utils.data as D

from torch.optim.lr_scheduler import ExponentialLR



from torchvision import models, transforms as T



from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer

from ignite.metrics import Loss, Accuracy

from ignite.contrib.handlers.tqdm_logger import ProgressBar

from ignite.handlers import  EarlyStopping, ModelCheckpoint



from tqdm import tqdm_notebook



from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')
path_data = '../input'

device = 'cuda'

batch_size = 32

torch.manual_seed(0)
class ImagesDS(D.Dataset):

    def __init__(self, df, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):

        self.records = df.to_records(index=False)

        self.channels = channels

        self.site = site

        self.mode = mode

        self.img_dir = img_dir

        self.len = df.shape[0]

        

    @staticmethod

    def _load_img_as_tensor(file_name):

        with Image.open(file_name) as img:

            return T.ToTensor()(img)



    def _get_img_path(self, index, channel):

        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate

        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])

        

    def __getitem__(self, index):

        paths = [self._get_img_path(index, ch) for ch in self.channels]

        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])

        if self.mode == 'train':

            return img, int(self.records[index].sirna)

        else:

            return img, self.records[index].id_code



    def __len__(self):

        return self.len
df = pd.read_csv(path_data+'/train.csv')

df_train, df_val = train_test_split(df, test_size = 0.025, random_state=42)

df_test = pd.read_csv(path_data+'/test.csv')
ds = ImagesDS(df_train, path_data, mode='train')

ds_val = ImagesDS(df_val, path_data, mode='train')

ds_test = ImagesDS(df_test, path_data, mode='test')
classes = 1108

model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features

model.fc = torch.nn.Linear(num_ftrs, classes)



# let's make our model work with 6 channels

trained_kernel = model.conv1.weight

new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

with torch.no_grad():

    new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

model.conv1 = new_conv
loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

val_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=4)

tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
metrics = {

    'loss': Loss(criterion),

    'accuracy': Accuracy(),

}





trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
@trainer.on(Events.EPOCH_COMPLETED)

def compute_and_display_val_metrics(engine):

    epoch = engine.state.epoch

    metrics = val_evaluator.run(val_loader).metrics

    print("Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} "

          .format(engine.state.epoch, 

                      metrics['loss'], 

                      metrics['accuracy']))
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)



@trainer.on(Events.EPOCH_COMPLETED)

def update_lr_scheduler(engine):

    lr_scheduler.step()

    lr = float(optimizer.param_groups[0]['lr'])

    print("Learning rate: {}".format(lr))
handler = EarlyStopping(patience=6, score_function=lambda engine: engine.state.metrics['accuracy'], trainer=trainer)

val_evaluator.add_event_handler(Events.COMPLETED, handler)
@trainer.on(Events.EPOCH_STARTED)

def turn_on_layers(engine):

    epoch = engine.state.epoch

    if epoch == 1:

        for name, child in model.named_children():

            if name == 'fc':

                pbar.log_message(name + ' is unfrozen')

                for param in child.parameters():

                    param.requires_grad = True

            else:

                pbar.log_message(name + ' is frozen')

                for param in child.parameters():

                    param.requires_grad = False

    if epoch == 3:

        pbar.log_message("Turn on all the layers")

        for name, child in model.named_children():

            for param in child.parameters():

                param.requires_grad = True
checkpoints = ModelCheckpoint('models', 'Model', save_interval=3, n_saved=3, create_dir=True)

trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoints, {'ResNet18': model})
pbar = ProgressBar(bar_format='')

pbar.attach(trainer, output_transform=lambda x: {'loss': x})
import os

if not 'KAGGLE_WORKING_DIR' in os.environ:  #  If we are not on kaggle server

    from ignite.contrib.handlers.tensorboard_logger import *

    tb_logger = TensorboardLogger("board/ResNet18")

    tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", output_transform=lambda loss: {'loss': loss}),

                     event_name=Events.ITERATION_COMPLETED)



    tb_logger.attach(val_evaluator, log_handler=OutputHandler(tag="validation", metric_names=["accuracy", "loss"],

                     another_engine=trainer),event_name=Events.EPOCH_COMPLETED)



    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)



    tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)

    tb_logger.close()
trainer.run(loader, max_epochs=50)
model.eval()

with torch.no_grad():

    preds = np.empty(0)

    for x, _ in tqdm_notebook(tloader): 

        x = x.to(device)

        output = model(x)

        idx = output.max(dim=-1)[1].cpu().numpy()

        preds = np.append(preds, idx, axis=0)
submission = pd.read_csv(path_data + '/test.csv')

submission['sirna'] = preds.astype(int)

submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])