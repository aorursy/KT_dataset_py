# Let's install ignite as a custom package:
!pip install git+https://github.com/pytorch/ignite.git --prefix=/kaggle/working
    
import sys
sys.path.insert(0, "/kaggle/working/lib/python3.6/site-packages")
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.transforms import ColorJitter, ToTensor, Normalize


FRUIT360_PATH = Path(".").resolve().parent / "input" / "fruits-360_dataset" / "fruits-360"

img_size = 64

device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

train_transform = Compose([
    RandomHorizontalFlip(),    
    RandomResizedCrop(size=img_size),
    ColorJitter(brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = Compose([
    RandomResizedCrop(size=img_size),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

batch_size = 128
num_workers = 8

train_dataset = ImageFolder((FRUIT360_PATH /"Training").as_posix(), transform=train_transform, target_transform=None)
val_dataset = ImageFolder((FRUIT360_PATH /"Test").as_posix(), transform=val_transform, target_transform=None)

pin_memory = "cuda" in device
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                          drop_last=True, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                        drop_last=False, pin_memory=pin_memory)

print("PyTorch version: {} | Device: {}".format(torch.__version__, device))
print("Train loader: num_batches={} | num_samples={}".format(len(train_loader), len(train_loader.sampler)))
print("Validation loader: num_batches={} | num_samples={}".format(len(val_loader), len(val_loader.sampler)))
import torch.nn as nn
from torchvision.models.squeezenet import squeezenet1_1
from torch.optim import SGD
model = squeezenet1_1(pretrained=False, num_classes=81)
model.classifier[-1] = nn.AdaptiveAvgPool2d(1)
model = model.to(device)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()
from ignite.engine import Engine, _prepare_batch, create_supervised_trainer

def model_update(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = _prepare_batch(batch, device=device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(model_update)
from ignite.engine import Events

log_interval = 50 
if 'cpu' in device:
    log_interval = 5 

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iteration = (engine.state.iteration - 1) % len(train_loader) + 1
    if iteration % log_interval == 0:
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(engine.state.epoch, iteration, len(train_loader), engine.state.output))

output = trainer.run(train_loader, max_epochs=1)
from ignite.metrics import Loss, CategoricalAccuracy, Precision, Recall


metrics = {
    'avg_loss': Loss(criterion),
    'avg_accuracy': CategoricalAccuracy(),
    'avg_precision': Precision(average=True), 
    'avg_recall': Recall(average=True)
}
from ignite.engine import create_supervised_evaluator

train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
import numpy as np
from torch.utils.data.dataset import Subset

random_indices = np.random.permutation(np.arange(len(train_dataset)))[:len(val_dataset)]
train_subset = Subset(train_dataset, indices=random_indices)

train_eval_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                               drop_last=True, pin_memory="cuda" in device)
@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_offline_train_metrics(engine):
    epoch = engine.state.epoch
    print("Compute train metrics...")
    metrics = train_evaluator.run(train_eval_loader).metrics
    print("Training Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f}"
          .format(engine.state.epoch, metrics['avg_loss'], metrics['avg_accuracy'], metrics['avg_precision'], metrics['avg_recall']))
    
    
@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    print("Compute validation metrics...")
    metrics = val_evaluator.run(val_loader).metrics
    print("Validation Results - Epoch: {}  Average Loss: {:.4f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f}"
          .format(engine.state.epoch, metrics['avg_loss'], metrics['avg_accuracy'], metrics['avg_precision'], metrics['avg_recall']))    
output = trainer.run(train_loader, max_epochs=1)
from torch.optim.lr_scheduler import ExponentialLR


lr_scheduler = ExponentialLR(optimizer, gamma=0.8)


@trainer.on(Events.EPOCH_STARTED)
def update_lr_scheduler(engine):
    lr_scheduler.step()
    # Display learning rate:
    if len(optimizer.param_groups) == 1:
        lr = float(optimizer.param_groups[0]['lr'])
        print("Learning rate: {}".format(lr))
    else:
        for i, param_group in enumerate(optimizer.param_groups):
            lr = float(param_group['lr'])
            print("Learning rate (group {}): {}".format(i, lr))    
from ignite.handlers import ModelCheckpoint


def score_function(engine):
    val_avg_accuracy = engine.state.metrics['avg_accuracy']
    # Objects with highest scores will be retained.
    return val_avg_accuracy


best_model_saver = ModelCheckpoint("best_models",  # folder where to save the best model(s)
                                   filename_prefix="model",  # filename prefix -> {filename_prefix}_{name}_{step_number}_{score_name}={abs(score_function_result)}.pth
                                   score_name="val_accuracy",  
                                   score_function=score_function,
                                   n_saved=3,
                                   atomic=True,  # objects are saved to a temporary file and then moved to final destination, so that files are guaranteed to not be damaged
                                   save_as_state_dict=True,  # Save object as state_dict
                                   create_dir=True)

val_evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {"best_model": model})
training_saver = ModelCheckpoint("checkpoint",
                                 filename_prefix="checkpoint",
                                 save_interval=1000,
                                 n_saved=1,
                                 atomic=True,
                                 save_as_state_dict=True,
                                 create_dir=True)

to_save = {"model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler} 
trainer.add_event_handler(Events.ITERATION_COMPLETED, training_saver, to_save)
from ignite.handlers import EarlyStopping

early_stopping = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)

val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)
max_epochs = 10

output = trainer.run(train_loader, max_epochs=max_epochs)
!ls best_models/
!ls checkpoint/
class TestDataset(Dataset):
    
    def __init__(self, ds):
        self.ds = ds
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index][0], index

    
test_dataset = TestDataset(val_dataset)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                         drop_last=False, pin_memory="cuda" in device)
import torch.nn.functional as F
from ignite._utils import convert_tensor


def _prepare_batch(batch):
    x, index = batch
    x = convert_tensor(x, device=device)
    return x, index


def inference_update(engine, batch):
    x, indices = _prepare_batch(batch)
    y_pred = model(x)
    y_pred = F.softmax(y_pred, dim=1)
    return {"y_pred": convert_tensor(y_pred, device='cpu'), "indices": indices}

    
model.eval()
inferencer = Engine(inference_update)    
@inferencer.on(Events.EPOCH_COMPLETED)
def log_tta(engine):
    print("TTA {} / {}".format(engine.state.epoch, n_tta))

    
n_tta = 3
num_classes = 81
n_samples = len(val_dataset)

# Array to store prediction probabilities
y_probas_tta = np.zeros((n_samples, num_classes, n_tta), dtype=np.float32)

# Array to store sample indices
indices = np.zeros((n_samples, ), dtype=np.int)
    

@inferencer.on(Events.ITERATION_COMPLETED)
def save_results(engine):
    output = engine.state.output
    tta_index = engine.state.epoch - 1
    start_index = ((engine.state.iteration - 1) % len(test_loader)) * batch_size
    end_index = min(start_index + batch_size, n_samples)
    batch_y_probas = output['y_pred'].detach().numpy()
    y_probas_tta[start_index:end_index, :, tta_index] = batch_y_probas
    if tta_index == 0:
        indices[start_index:end_index] = output['indices']
inferencer.run(test_loader, max_epochs=n_tta)
y_probas = np.mean(y_probas_tta, axis=-1)
y_preds = np.argmax(y_probas, axis=-1)
from sklearn.metrics import accuracy_score

y_test_true = [y for _, y in val_dataset]
accuracy_score(y_test_true, y_preds)
# Remove output to be able to commit
!rm -R best_models/ checkpoint/ lib/