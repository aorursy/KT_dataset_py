import os

import gc

import time

import math

import random

import numpy as np 

import pandas as pd 

from pathlib import Path

from collections import defaultdict, Counter

import tqdm

from PIL import Image, ImageOps, ImageEnhance



import torch

from torch import nn, cuda

import torch.nn.functional as F

from torch.autograd import Variable

import torchvision.transforms as transforms

import torchvision.models as M

from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, SGD, Optimizer

from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau



import warnings

warnings.filterwarnings('ignore')
TEST_CV = False
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 42

seed_everything(SEED)
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):

    """

    Multi-class Lovasz-Softmax loss

      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).

              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].

      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)

      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.

      per_image: compute the loss per image instead of per batch

      ignore: void class labels

    """

    if per_image:

        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)

                          for prob, lab in zip(probas, labels))

    else:

        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)

    return loss





def lovasz_softmax_flat(probas, labels, classes='present'):

    """

    Multi-class Lovasz-Softmax loss

      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)

      labels: [P] Tensor, ground truth labels (between 0 and C - 1)

      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.

    """

    if probas.numel() == 0:

        # only void pixels, the gradients should be 0

        return probas * 0.

    C = probas.size(1)

    losses = []

    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes

    for c in class_to_sum:

        fg = (labels == c).float() # foreground for class c

        if (classes is 'present' and fg.sum() == 0):

            continue

        if C == 1:

            if len(classes) > 1:

                raise ValueError('Sigmoid output possible only with 1 class')

            class_pred = probas[:, 0]

        else:

            class_pred = probas[:, c]

        errors = (Variable(fg) - class_pred).abs()

        errors_sorted, perm = torch.sort(errors, 0, descending=True)

        perm = perm.data

        fg_sorted = fg[perm]

        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))

    return mean(losses)





def flatten_probas(probas, labels, ignore=None):

    """

    Flattens predictions in the batch

    """

    if probas.dim() == 3:

        # assumes output of a sigmoid layer

        B, H, W = probas.size()

        probas = probas.view(B, 1, H, W)

    B, C, H, W = probas.size()

    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C

    labels = labels.view(-1)

    if ignore is None:

        return probas, labels

    valid = (labels != ignore)

    vprobas = probas[valid.nonzero().squeeze()]

    vlabels = labels[valid]

    return vprobas, vlabels



def xloss(logits, labels, ignore=None):

    """

    Cross entropy loss

    """

    return F.cross_entropy(logits, Variable(labels), ignore_index=255)
from torchvision.models.resnet import ResNet, Bottleneck



def _resnext(arch, block, layers, pretrained, progress, **kwargs):

    model = ResNet(block, layers, **kwargs)

#     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

#     model.load_state_dict(state_dict)

    return model





def resnext101_32x8d_wsl(progress=True, **kwargs):

    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data

    and finetuned on ImageNet from Figure 5 in

    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_

    Args:

        progress (bool): If True, displays a progress bar of the download to stderr.

    """

    kwargs['groups'] = 32

    kwargs['width_per_group'] = 8

    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], False, progress, **kwargs)
class CosineAnnealingWithRestartsLR(_LRScheduler):

    '''

    SGDR\: Stochastic Gradient Descent with Warm Restarts: https://arxiv.org/abs/1608.03983

    code: https://github.com/gurucharanmk/PyTorch_CosineAnnealingWithRestartsLR/blob/master/CosineAnnealingWithRestartsLR.py

    added restart_decay value to decrease lr for every restarts

    '''

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_mult=1, restart_decay=0.95):

        self.T_max = T_max

        self.T_mult = T_mult

        self.next_restart = T_max

        self.eta_min = eta_min

        self.restarts = 0

        self.last_restart = 0

        self.T_num = 0

        self.restart_decay = restart_decay

        super(CosineAnnealingWithRestartsLR,self).__init__(optimizer, last_epoch)



    def get_lr(self):

        self.Tcur = self.last_epoch - self.last_restart

        if self.Tcur >= self.next_restart:

            self.next_restart *= self.T_mult

            self.last_restart = self.last_epoch

            self.T_num += 1

        learning_rate = [(self.eta_min + ((base_lr)*self.restart_decay**self.T_num - self.eta_min) * (1 + math.cos(math.pi * self.Tcur / self.next_restart)) / 2) for base_lr in self.base_lrs]

        return learning_rate
class AdamW(Optimizer):

    """Implements AdamW algorithm.



    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.



    Arguments:

        params (iterable): iterable of parameters to optimize or dicts defining

            parameter groups

        lr (float, optional): learning rate (default: 1e-3)

        betas (Tuple[float, float], optional): coefficients used for computing

            running averages of gradient and its square (default: (0.9, 0.999))

        eps (float, optional): term added to the denominator to improve

            numerical stability (default: 1e-8)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)



    .. Fixing Weight Decay Regularization in Adam:

    https://arxiv.org/abs/1711.05101

    """



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps,

                        weight_decay=weight_decay)

        super(AdamW, self).__init__(params, defaults)



    def step(self, closure=None):

        """Performs a single optimization step.



        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')



                state = self.state[p]



                # State initialization

                if len(state) == 0:

                    state['step'] = 0

                    # Exponential moving average of gradient values

                    state['exp_avg'] = torch.zeros_like(p.data)

                    # Exponential moving average of squared gradient values

                    state['exp_avg_sq'] = torch.zeros_like(p.data)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                state['step'] += 1



                # according to the paper, this penalty should come after the bias correction

                # if group['weight_decay'] != 0:

                #     grad = grad.add(group['weight_decay'], p.data)



                # Decay the first and second moment running average coefficient

                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)



                denom = exp_avg_sq.sqrt().add_(group['eps'])



                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1



                p.data.addcdiv_(-step_size, exp_avg, denom)



                if group['weight_decay'] != 0:

                    p.data.add_(-group['weight_decay'], p.data)



        return loss
class LabelSmoothingLoss(nn.Module):

    """

        based on https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L38

    """

    def __init__(self, smooth_eps=0.1, ignore_index=-100, reduction='mean'):

        assert 0.0 < smooth_eps <= 1.0

        super(LabelSmoothingLoss, self).__init__()



        self.eps = smooth_eps

        self.ignore_index = ignore_index

        self.reduction = reduction



    def forward(self, pred, target):

        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)

        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)

        log_prb = F.log_softmax(pred, dim=1)



        non_pad_mask = target.ne(self.ignore_index) # ignore mask

        loss = -(one_hot * log_prb).sum(dim=1) # negative log likelihood



        if self.reduction == 'mean':

            loss = loss.masked_select(non_pad_mask).mean()

        elif self.reduction == 'sum':

            loss = loss.masked_select(non_pad_mask).mean()

        else:

            raise

        return loss

    

'''

mixup: Beyond Empirical Risk Minimization

https://arxiv.org/abs/1710.09412

'''

def mixup_criterion(criterion, pred, y_a, y_b, lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Returns mixed inputs, pairs of targets, and lambda'''

    if alpha > 0:

        lam = np.random.beta(alpha, alpha)

    else:

        lam = 1



    batch_size = x.size()[0]

    if use_cuda:

        index = torch.randperm(batch_size).cuda()

    else:

        index = torch.randperm(batch_size)



    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
use_cuda = cuda.is_available()

use_cuda
os.listdir('../input/face-images-cls')
IMAGES_PATH = '../input/face-images-cls/faces_images'



train_df = pd.read_csv('../input/face-images-cls/train_vision.csv')

test_df = pd.read_csv('../input/face-images-cls/test_vision.csv')

submission = pd.read_csv('../input/face-images-cls/sample_output_vision.csv')

df = pd.read_csv('../input/face-images-cls/folds.csv')



num_classes = train_df['label'].nunique()
df.head()
print("total train+valid images: {}".format(len(df)))

print("total test images: {}".format(len(test_df)))

print("classes to predict: {}".format(df['label'].nunique()))

print("total images: {}".format(len(df)))

class TrainDataset(Dataset):

    def __init__(self, df, mode='train', transforms=None):

        self.df = df

        self.mode = mode

        self.transform = transforms[self.mode]

        

    def __len__(self):            

        return len(self.df)

            

    def __getitem__(self, idx):

        

        image = Image.open(IMAGES_PATH + '/' + self.df['filename'][idx]).convert("RGB")



        if self.transform:

            image = self.transform(image)

        label = self.df['label'][idx]

        

        return image, label

    

class TestDataset(Dataset):

    def __init__(self, df, transforms=None):

        self.df = df

        self.transform = transforms

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        image = Image.open(IMAGES_PATH + '/' + self.df['filename'][idx]).convert("RGB")

            

        if self.transform:

            image = self.transform(image)

            

        return image     
target_size = (128, 128)



data_transforms = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(target_size, scale=(0.7, 1.0)),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(

                brightness=0.4,

                contrast=0.4,

                saturation=0.4,

            ),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.5161, 0.4225, 0.3763], 

            [0.2442, 0.2184, 0.2164]

        )]),

    'valid': transforms.Compose([

        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.5161, 0.4225, 0.3763], 

            [0.2442, 0.2184, 0.2164]

        )])

}



test_transforms = transforms.Compose([

        transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(

            [0.5161, 0.4225, 0.3763], 

            [0.2442, 0.2184, 0.2164])

])

def train_one_epoch(model, criterion, train_loader, optimizer, mixup_loss=False, label_smoothing=False, accumulation_step=2):

    

    model.train()

    train_loss = 0.

    

    optimizer.zero_grad()



    for i, (inputs, targets) in enumerate(train_loader):

        

        inputs, targets = inputs.cuda(), targets.cuda()    

            

        if mixup_loss:

            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, use_cuda = use_cuda)

            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            preds = model(inputs)

            if label_smoothing:

                loss = mixup_criterion(label_smoothed_loss, preds.cuda(), targets_a.cuda(), targets_b.cuda(), lam)

            else:

                loss = mixup_criterion(criterion, preds.cuda(), targets_a.cuda(), targets_b.cuda(), lam)

        else:

            preds = model(inputs)

            if label_smoothing:

                loss = label_smoothed_loss(preds, targets)

            else:

                loss = criterion(preds, targets)

   

        loss.backward()

        

        if accumulation_step:

            if (i+1) % accumulation_step == 0:  

                optimizer.step()

                optimizer.zero_grad()

        else:

            optimizer.step()

            optimizer.zero_grad()



        train_loss += loss.item() / len(train_loader)

        

    return train_loss





def validation(model, criterion, valid_loader):

    

    model.eval()

    valid_preds = np.zeros((len(valid_dataset), num_classes))

    val_loss = 0.

    valid_total_correct = 0

    

    with torch.no_grad():

        for i, (inputs, targets) in enumerate(valid_loader):

            

            inputs, targets = inputs.cuda(), targets.cuda()

            

            outputs = model(inputs).detach()

            loss = criterion(outputs, targets)

            

            output_prob = F.softmax(outputs, dim=1)

            predict_vector = np.argmax(output_prob.cpu().detach().numpy(), axis=1)

            label_vector = targets.cpu().detach().numpy()

            bool_vector = predict_vector == label_vector

            

            valid_preds[i * batch_size: (i+1) * batch_size] = outputs.cpu().numpy()

            

            val_loss += loss.item() / len(valid_loader)

            valid_total_correct += bool_vector.sum()

            

    val_acc = valid_total_correct / len(valid_loader.dataset)

        

    return val_loss, val_acc   
def train_model(n_epochs=100, accumulation_step=1, fine_tune=False, cosine_schedule=False,

                mixup_loss=False, label_smoothing=False, **kwargs):

    

    if fine_tune:

        optimizer = AdamW(model.parameters(), lr=0.000025, weight_decay=0.000025)   

        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7, factor=0.2)



    else:

        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0025)

        eta_min = 1e-6

        T_max = 10

        T_mult = 1

        restart_decay = 0.97

        scheduler = CosineAnnealingWithRestartsLR(optimizer,T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)



    best_epoch = -1

    best_valid_score = 0.

    best_valid_loss = 1.

    all_train_loss = []

    all_valid_loss = []

    train_step = 0

    

    for epoch in range(n_epochs):

        

        start_time = time.time()



        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, mixup_loss, label_smoothing, accumulation_step)

        val_loss, val_score = validation(model, criterion, valid_loader)

    

        if val_score > best_valid_score:

            best_valid_score = val_score

            torch.save(model.state_dict(), 'best_score.pt')

    

        elapsed = time.time() - start_time

        

        lr = [_['lr'] for _ in optimizer.param_groups]

        print("Epoch {} - train_loss: {:.4f}  val_loss: {:.4f}  val_acc: {:.4f}  lr: {:.7f}  time: {:.0f}s".format(

                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))



        # scheduler update

        if fine_tune:

            scheduler.step(val_score)

        else:

            scheduler.step()

            

#         for param_group in optimizer.param_groups:

#             lrs.append(param_group['lr'])

            

            # paper:  Snapshot Ensembles: Train 1, get M for free

            # https://arxiv.org/abs/1704.00109 

#             if param_group['lr'] == 1e-6 and epoch >= 40:

#                 save_model('{}model.pt'.format(epoch+1), save_all_states=False, model=model.state_dict())

                

    print("\n Fold training done: best score: {}".format(np.round(best_valid_score, 4)))
criterion = nn.CrossEntropyLoss()

label_smoothed_loss = LabelSmoothingLoss()
rough_annealing_epochs = 61

fine_tune_epochs = 50

# rough_annealing_epochs = 1

### all_prediction = []



for fold_num in [0, 1, 2, 3, 4]:

    train_df = df.loc[df['fold'] != fold_num][['filename', 'label']].reset_index(drop=True)

    valid_df = df.loc[df['fold'] == fold_num][['filename', 'label']].reset_index(drop=True)



    batch_size = 32

    train_dataset = TrainDataset(train_df, mode='train', transforms=data_transforms)

    valid_dataset = TrainDataset(valid_df, mode='valid', transforms=data_transforms)



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    

#     model = M.resnet50(pretrained=False, num_classes=num_classes)

    model = resnext101_32x8d_wsl(num_classes=num_classes)

    model.cuda()

    

    train_kwargs = dict(

    train_loader=train_loader,

    valid_loader=valid_loader,

    model=model,

    criterion=criterion,

    )

    

    print("start training fold {}".format(fold_num))

    # rough annealing

    n_epochs = rough_annealing_epochs

    train_model(n_epochs=n_epochs, accumulation_step=1, fine_tune=False, mixup_loss=False, label_smoothing=False, **train_kwargs)

    

#     # fine_tuning

#     if not TEST_CV:

# #         model = M.resnet50(pretrained=False, num_classes=num_classes)

#         model = resnext101_32x8d_wsl(num_classes=num_classes)

#         model.load_state_dict(torch.load('../working/best_score.pt'))

#         model.cuda()



#         print("\nstart fine_tune fold {}".format(fold_num))

#         n_epochs = fine_tune_epochs

#         train_model(n_epochs=n_epochs, accumulation_step=1, fine_tune=True, mixup_loss=False, label_smoothing=False, **train_kwargs)

    

    # inference test set

    test_dataset = TestDataset(test_df, transforms=test_transforms)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)



    model = M.resnet50(pretrained=False, num_classes=num_classes)

#     model = resnext101_32x8d_wsl(num_classes=num_classes)

    model.load_state_dict(torch.load('../working/best_score.pt'))

    model.cuda()



    model.eval()

    

    tta = 4

    tta_predictions = []

    for _ in range(tta):

        prediction = np.zeros((len(test_dataset), num_classes))

        

        with torch.no_grad():

            for i, images in enumerate(test_loader):



                images = images.cuda()



                preds = model(images).detach()

                prediction[i * batch_size: (i+1) * batch_size] = preds.cpu().numpy()   

        

        tta_predictions.append(prediction)

         

    mean_tta_predictions = np.mean(tta_predictions, axis=0)

    

    all_prediction.append(mean_tta_predictions)

    del train_dataset, valid_dataset, train_loader, valid_loader, model, prediction

    gc.collect()

    print('\n')

    

    if TEST_CV:

        break
if TEST_CV:

    single_model_prediction = all_prediction[0]

    result = np.argmax(single_model_prediction, axis=1)

    result = result + 1

else:

    folds_averaged = np.mean(all_prediction, axis=0)



    result = np.argmax(folds_averaged, axis=1)

    result = result + 1
submission = pd.DataFrame(result, columns=['prediction'])

submission.to_csv("submission.csv", index=False)
train_df['label'].value_counts()
valid_df['label'].value_counts()
submission['prediction'].value_counts()