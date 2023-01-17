import os

import time

import math

import random

import numpy as np

import pandas as pd

from pathlib import Path

import glob

import matplotlib.pyplot as plt

from PIL import Image, ImageEnhance, ImageOps



from tqdm import tqdm, tqdm_notebook



import torch

from torch import nn, cuda

from torch.autograd import Variable 

import torch.nn.functional as F

import torchvision as vision

import torchvision.models as models

from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, SGD, Optimizer

from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau



from sklearn.metrics import f1_score
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
'''

유명한 mixup 논문. 타대회에서도 많이 씁니다.

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



'''

Homin님 kernel 참조

AutoAugment: Learning Augmentation Policies from Data

https://arxiv.org/abs/1905.

Code: https://github.com/DeepVoltaire/AutoAugment

'''

    

class CIFAR10Policy(object):

    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:

        >>> policy = CIFAR10Policy()

        >>> transformed = policy(image)

        Example as a PyTorch Transform:

        >>> transform=transforms.Compose([

        >>>     transforms.Resize(256),

        >>>     CIFAR10Policy(),

        >>>     transforms.ToTensor()])

    """

    def __init__(self, fillcolor=(128, 128, 128)):

        self.policies = [

            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),

            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),

            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),

            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),

            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),



            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),

            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),

            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),

            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),

            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),



            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),

            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),

            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),

            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),

            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),



            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),

            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),

            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),

            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),

            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),



            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),

            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),

            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),

            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),

            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)

        ]





    def __call__(self, img):

        policy_idx = random.randint(0, len(self.policies) - 1)

        return self.policies[policy_idx](img)



    def __repr__(self):

        return "AutoAugment CIFAR10 Policy"



    

class SubPolicy(object):

    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):

        ranges = {

            "shearX": np.linspace(0, 0.3, 10),

            "shearY": np.linspace(0, 0.3, 10),

            "translateX": np.linspace(0, 150 / 331, 10),

            "translateY": np.linspace(0, 150 / 331, 10),

            "rotate": np.linspace(0, 30, 10),

            "color": np.linspace(0.0, 0.9, 10),

            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),

            "solarize": np.linspace(256, 0, 10),

            "contrast": np.linspace(0.0, 0.9, 10),

            "sharpness": np.linspace(0.0, 0.9, 10),

            "brightness": np.linspace(0.0, 0.9, 10),

            "autocontrast": [0] * 10,

            "equalize": [0] * 10,

            "invert": [0] * 10

        }



        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand

        def rotate_with_fill(img, magnitude):

            rot = img.convert("RGBA").rotate(magnitude)

            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)



        func = {

            "shearX": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),

                Image.BICUBIC, fillcolor=fillcolor),

            "shearY": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),

                Image.BICUBIC, fillcolor=fillcolor),

            "translateX": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),

                fillcolor=fillcolor),

            "translateY": lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),

                fillcolor=fillcolor),

            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),

            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),

            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),

            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),

            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),

            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(

                1 + magnitude * random.choice([-1, 1])),

            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),

            "equalize": lambda img, magnitude: ImageOps.equalize(img),

            "invert": lambda img, magnitude: ImageOps.invert(img)

        }



        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(

        #     operation1, ranges[operation1][magnitude_idx1],

        #     operation2, ranges[operation2][magnitude_idx2])

        self.p1 = p1

        self.operation1 = func[operation1]

        self.magnitude1 = ranges[operation1][magnitude_idx1]

        self.p2 = p2

        self.operation2 = func[operation2]

        self.magnitude2 = ranges[operation2][magnitude_idx2]





    def __call__(self, img):

        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)

        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)

        return img
# seed value fix

# seed 값을 고정해야 hyper parameter 바꿀 때마다 결과를 비교할 수 있습니다.

def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 2019

seed_everything(SEED)
use_cuda = cuda.is_available()

use_cuda
class TrainDataset(Dataset):

    def __init__(self, df, mode='train', transforms=None):

        self.df = df

        self.mode = mode

        self.transform = transforms[self.mode]

        

    def __len__(self):

        return len(self.df)

            

    def __getitem__(self, idx):

        

        image = Image.open(TRAIN_IMAGE_PATH / self.df['img_file'][idx]).convert("RGB")



        if self.transform:

            image = self.transform(image)



        label = self.df['class'][idx]



        return image, label



    

class TestDataset(Dataset):

    def __init__(self, df, mode='test', transforms=None):

        self.df = df

        self.mode = mode

        self.transform = transforms[self.mode]

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        

        image = Image.open(TEST_IMAGE_PATH / self.df[idx]).convert("RGB")

        

        if self.transform:

            image = self.transform(image)

            

        return image        
target_size = (224, 224)



data_transforms = {

    'train': vision.transforms.Compose([

        vision.transforms.Resize(target_size),

        vision.transforms.RandomHorizontalFlip(),

        vision.transforms.RandomRotation(20),

        CIFAR10Policy(),

        vision.transforms.ToTensor(),

        vision.transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

    'valid': vision.transforms.Compose([

        vision.transforms.Resize(target_size),

        vision.transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),

        vision.transforms.RandomHorizontalFlip(),

        vision.transforms.ToTensor(),

        vision.transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

    'test': vision.transforms.Compose([

        vision.transforms.Resize((224,224)),

        vision.transforms.RandomResizedCrop(target_size, scale=(0.8,1.0)),

        vision.transforms.ToTensor(),

        vision.transforms.Normalize(

            [0.485, 0.456, 0.406], 

            [0.229, 0.224, 0.225])

    ]),

}
'''

crop된 이미지 사용

reference 허태명님 커널: https://www.kaggle.com/tmheo74/3rd-ml-month-car-image-cropping

'''



TRAIN_IMAGE_PATH = Path('../input/kakl-3rd-cropped-dataset/train_crop/')

TEST_IMAGE_PATH = Path('../input/kakl-3rd-cropped-dataset/test_crop/')

# train_image_path = Path('../input/2019-3rd-ml-month-with-kakr/train/')

# test_image_path = Path('../input/2019-3rd-ml-month-with-kakr/test/')
# 미리 5 fold로 나누어 csv로 저장한 후 불러왔습니다.

# 80프로를 train set으로, 나머지 20프로를 validation set으로 사용합니다. => 수정: 실수로 4 kfold를 해버렸네요 (3/4 train set, 1/4 valid set입니다)

df = pd.read_csv("../input/car-folds/car_4folds.csv")

test_csv = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/test.csv')

df.head()
# class 분포 고려하여 사전에 split 해놨습니다. fold별 개수 확인 가능

len(df[df['fold'] == 0]), len(df[df['fold'] == 1]), len(df[df['fold'] == 2]), len(df[df['fold'] == 3])
train_df = df.loc[df['fold'] != 0]

valid_df = df.loc[df['fold'] == 0]
# for debugging

# train_df = train_df[:900]

# valid_df = valid_df[:900]
train_df = train_df[['img_file', 'class']].reset_index(drop=True)

valid_df = valid_df[['img_file', 'class']].reset_index(drop=True)

x_test = test_csv['img_file']

train_df.replace(196, 0, inplace=True) # 대회 데이터 클래스에 0이 없기에 일부러 바꿔줬습니다. model train시 클래스에 0이 없으면 오류 나기 때문에



num_classes = train_df['class'].nunique()

y_true = valid_df['class'].values # for cv score
print("number of train dataset: {}".format(len(train_df)))

print("number of valid dataset: {}".format(len(valid_df)))

print("number of classes to predict: {}".format(num_classes))
def train_one_epoch(model, criterion, train_loader, optimizer, mixup_loss, accumulation_step=2):

    

    model.train()

    train_loss = 0.

    optimizer.zero_grad()



    for i, (inputs, targets) in enumerate(train_loader):

            

        inputs, targets = inputs.cuda(), targets.cuda()



        if mixup_loss:

            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, use_cuda = use_cuda) # alpha in [0.4, 1.0] 선택 가능

            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            outputs = model(inputs)

            loss = mixup_criterion(criterion, outputs.cuda(), targets_a.cuda(), targets_b.cuda(), lam)

            

        else:

            outputs = model(inputs)

            loss = criterion(outputs, targets)



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

    

    with torch.no_grad():

        for i, (inputs, targets) in enumerate(valid_loader):



            inputs, targets = inputs.cuda(), targets.cuda()

            

            outputs = model(inputs).detach()

            loss = criterion(outputs, targets)

            valid_preds[i * batch_size: (i+1) * batch_size] = outputs.cpu().numpy()

            

            val_loss += loss.item() / len(valid_loader)

            

        y_pred = np.argmax(valid_preds, axis=1)

        val_score = f1_score(y_true, y_pred, average='micro')  

        

    return val_loss, val_score    
# 스코어 기준과 loss 기준. lb 점수가 cv score와 비교했을 때 굉장히

# consistent해서 cv score를 기준으로 합니다.

def pick_best_score(result1, result2):

    if result1['best_score'] < result2['best_score']:

        return result2

    else:

        return result1

    

def pick_best_loss(result1, result2):

    if result1['best_loss'] < result2['best_loss']:

        return result1

    else:

        return result2
def train_model(num_epochs=60, accumulation_step=4, mixup_loss=False, cv_checkpoint=False, fine_tune=False, weight_file_name='weight_best.pt', **train_kwargs):

    

    # choose scheduler

    if fine_tune:

        lr = 0.00001

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.000025)   

        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    else:    

        lr = 0.01

        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.025)

        eta_min = 1e-6

        T_max = 10

        T_mult = 1

        restart_decay = 0.97

        scheduler = CosineAnnealingWithRestartsLR(optimizer,T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)



    train_result = {}

    train_result['weight_file_name'] = weight_file_name

    best_epoch = -1

    best_score = 0.

    lrs = []

    score = []

    

    for epoch in range(num_epochs):

        

        start_time = time.time()



        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, mixup_loss, accumulation_step)

        val_loss, val_score = validation(model, criterion, valid_loader)

        score.append(val_score)

    

        # model save (score or loss?)

        if cv_checkpoint:

            if val_score > best_score:

                best_score = val_score

                train_result['best_epoch'] = epoch + 1

                train_result['best_score'] = round(best_score, 5)

                torch.save(model.state_dict(), weight_file_name)

        else:

            if val_loss < best_loss:

                best_loss = val_loss

                train_result['best_epoch'] = epoch + 1

                train_result['best_loss'] = round(best_loss, 5)

                torch.save(model.state_dict(), weight_file_name)

        

        elapsed = time.time() - start_time

        

        lr = [_['lr'] for _ in optimizer.param_groups]

        print("Epoch {} - train_loss: {:.4f}  val_loss: {:.4f}  cv_score: {:.4f}  lr: {:.6f}  time: {:.0f}s".format(

                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))

        

        for param_group in optimizer.param_groups:

            lrs.append(param_group['lr'])

        

        # scheduler update

        if fine_tune:

            if cv_checkpoint:

                scheduler.step(val_score)

            else:

                scheduler.step(val_loss)

        else:

            scheduler.step()

     

    return train_result, lrs, score
batch_size = 128



train_dataset = TrainDataset(train_df, mode='train', transforms=data_transforms)

valid_dataset = TrainDataset(valid_df, mode='valid', transforms=data_transforms)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# baseline이기 때문에 resnet50 사용합니다. 바꿔보세요!

model = models.resnet50(pretrained=True)

model.fc = nn.Linear(2048, num_classes)

model.cuda()
criterion = nn.CrossEntropyLoss()



train_kwargs = dict(

    train_loader=train_loader,

    valid_loader=valid_loader,

    model=model,

    criterion=criterion,

    )





print("training starts")

num_epochs = 120

result, lrs, score = train_model(num_epochs=num_epochs, accumulation_step=2, mixup_loss=False, cv_checkpoint=True, fine_tune=False, weight_file_name='weight_best.pt', **train_kwargs)

print(result)





# finetuning 부분은 전 버전 참고하시면 좋을것 같습니다.
# learning rate plot

plt.figure(figsize=(18,4))

plt.subplot(1,2,1)

plt.plot(lrs, 'b')

plt.xlabel('Epochs', fontsize=12, fontweight='bold')

plt.ylabel('Learning rate', fontsize=14, fontweight='bold')

plt.title('Learning rate schedule', fontsize=15, fontweight='bold')



x = [x for x in range(0, num_epochs, 10)]

y = [0.01, 0.005, 0.000001]

ylabel = ['1e-2', '1e-4', '1e-6']

plt.xticks(x)

plt.yticks(y, ylabel)



plt.subplot(1,2,2)

plt.plot(score, 'r')

plt.xlabel('Epochs', fontsize=12, fontweight='bold')

plt.ylabel('Valid score', fontsize=14, fontweight='bold')

plt.title('F1 Score', fontsize=15, fontweight='bold')



x = [x for x in range(0, num_epochs, 10)]



plt.show()
# 저장한 weight 불러와서 predict  

# 최근에 열린 imet 대회 같은 경우는 학습 시간이 9시간 이상 해야하기 때문에 저장하고 불러오기가 중요합니다

# 보통 kaggle에서 딥러닝 대회는 training과 inference는 따로 커널을 만들어서 진행합니다 (저처럼 local gpu 없을 경우 필수)



model = models.resnet50() 

model.fc = nn.Linear(2048, num_classes)

model.cuda()

model.load_state_dict(torch.load(result['weight_file_name']))



batch_size = 1 # 배치 1로 주면 순서대로 나온다

test_dataset = TestDataset(x_test, mode='test', transforms=data_transforms)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



model.eval()

test_preds = []



with torch.no_grad():

    for i, images in enumerate(tqdm_notebook(test_loader)):

        images = images.cuda()

    

        preds = model(images).detach()

        test_preds.append(preds.cpu().numpy())
outputs = []

for _ in test_preds:

    # argmax를 사용해서 가장 높은 확률로 예측한 class 반환

    predicted_class_indices=np.argmax(_, axis=1).tolist()

    outputs.append(predicted_class_indices)



result = np.concatenate(outputs)
submission = pd.read_csv('../input/2019-3rd-ml-month-with-kakr/sample_submission.csv')

submission["class"] = result

submission["class"].replace(0, 196, inplace=True) # 196에서 0으로 수정했던걸 다시 되돌려준다 

submission.to_csv("submission.csv", index=False)

submission.head()
'''

참고하면 좋을 논문들. 타 대회에서도 다 적용 가능합니다.



Snapshot Ensembles: Train 1, get M for free - 심심해서 해봤는데 조금 오르긴 합니다.

https://arxiv.org/abs/1704.00109 



SGDR: Stochastic Gradient Descent with Warm Restarts

https://arxiv.org/abs/1608.03983



Unsupervised Data Augmentation - 김일두님 깃헙 참고해서 구현해보면 좋을것 같습니다

https://arxiv.org/abs/1904.12848



AutoAugment: Learning Augmentation Policies from Data

https://arxiv.org/abs/1805.09501

'''