import numpy as np
import pandas as pd
import math
import torch
from torch import nn, optim
import time
import os
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import builtins as __builtin__
def print(*args, **kwargs):
    __builtin__.print(*args, **kwargs)
    __builtin__.print('\n---------- 华丽的分割线 ----------\n')
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)
# vgg16 最终输出的是1000个类别，我们要调整成本次数据集所用的11个类别
# https://discuss.pytorch.org/t/transfer-learning-usage-with-different-input-size/20744/5
vgg16 = torchvision.models.vgg16()
vgg16.classifier.add_module('7', nn.ReLU(inplace=False))
vgg16.classifier.add_module('8', nn.Dropout(p=0.5, inplace=False))
vgg16.classifier.add_module('9', nn.Linear(1000, 11))
!ls /kaggle/input/food11/food-11/
# %matplotlib inline
# from PIL import Image
# import matplotlib.pyplot as plt 
# import os
# import numpy as np
# from torchvision import transforms
# import torch
# from torch import nn

# def show_images(imgs, num_rows, num_cols, scale=2):
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
#     for i in range(num_rows):
#         for j in range(num_cols):
#             axes[i][j].imshow(imgs[i * num_cols + j])
#             axes[i][j].axes.get_xaxis().set_visible(False)
#             axes[i][j].axes.get_yaxis().set_visible(False)
#     return axes
# files = ['0_0.jpg','0_1.jpg']
# x_train, y_train, y_vld = [], [], []
         
# for  idx,file in enumerate(files):
#     im = Image.open('/kaggle/input/food11/food-11/training/'+ file)
#     im = im.resize((2, 2), Image.BILINEAR)

#     # 将图片转化成RGB数据格式
#     im = im.convert("RGB")
#     im = np.array(im)

#     #图像通道转换， hwc->chw
#     # step 2: 轴与索引之间的对应关系
#     #  意义  轴  索引  -->  索引  轴  意义
#     #  高    3    0         2    3   通道
#     #  宽    2    1         0    3   高
#     #  通道  3    2         1    2   宽
    
#     im_chw = np.transpose(im, (2,0,1)) 

# #     plt.subplot(2,3,1), plt.title('R')
# #     plt.imshow(im_chw[0]), plt.axis('off')
# #     plt.subplot(2,3,2), plt.title('G')
# #     plt.imshow(im_chw[1]), plt.axis('off')
# #     plt.subplot(2,3,3), plt.title('B')
# #     plt.imshow(im_chw[2]), plt.axis('off')
#     x_train.append(im_chw)
    
#     # 获取每张图片的类别
#     _class = file.split('_')[0]
#     y_train.append(int(_class))
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# print(x_train)

# # 标准化操作，可用的方法有很多，这里是用Batch Normalization
# # 更多参考  https://zhuanlan.zhihu.com/p/69659844
# bn = nn.BatchNorm2d(6, eps=0, affine=False, track_running_stats=False)
# x_train = bn(torch.tensor(x_train, dtype=torch.float))

# print(x_train[0])
# plt.show()



%matplotlib inline
from PIL import Image
import matplotlib.pyplot as plt 
import os
import time
import numpy as np
from torch import nn

def load_img(read_local = True):
    print("开始读取图片。。。")
    start = time.time()
    data = os.path.exists('./data.npz')
    if (read_local == True & data == True):
        
        data = np.load('./data.npz')
        print("从保存结果中读取完成， 耗时 %.1f", time.time() - start)
        return data["x_train"], data["y_train"], data["x_vld"], data["y_vld"], data["x_test"]
        
    if data == True:
        os.remove('./data.npz')  
 
    dir = '/kaggle/input/food11/food-11/'
    root = os.listdir(dir)

    x_train, y_train, x_vld, y_vld, x_test = [], [],[], [],[]
    for baseFolder in root:
        for file in sorted(os.listdir(r'/kaggle/input/food11/food-11/'+ baseFolder)):
            im = Image.open('/kaggle/input/food11/food-11/'+ baseFolder +'/' + file)

            im = im.resize((224, 224), Image.BILINEAR) # 不同的模型输入的大小不同

            # 将图片转化成RGB数据格式
            im = im.convert("RGB")
            im = np.array(im) # 默认转换成np.array是hwc格式的

             #图像通道转换， hwc->chw
            # step 2: 轴与索引之间的对应关系
            #  意义  轴  索引  -->  索引  轴  意义
            #  高    3    0         2    3   通道
            #  宽    2    1         0    3   高
            #  通道  3    2         1    2   宽

#             im_chw = np.transpose(im, (2,0,1)) 
            if baseFolder == "training":

                x_train.append(im)
                # 获取每张图片的类别
                _class = file.split('_')[0]
                y_train.append(int(_class))

            elif baseFolder == "validation":

                x_vld.append(im)
                # 获取每张图片的类别
                _class = file.split('_')[0]
                y_vld.append(int(_class))
            elif baseFolder == "testing":

                x_test.append(im)


    # 最终处理完的数据格式应该是N*C*H*W N是样本数量，C是通道数，H是高，W是宽
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_vld = np.array(x_vld)
    y_vld = np.array(y_vld)

    x_test = np.array(x_test)
    
    np.savez("./data", x_train = x_train, y_train=y_train, x_vld= x_vld, y_vld=y_vld, x_test=x_test)
    print("从文件夹中读取完成， 耗时 %.1f", time.time() - start)

    return x_train, y_train, x_vld, y_vld, x_test
            
class Food11Dataset(torch.utils.data.Dataset):
    
    def __init__(self, x, y = None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        X = self.x[idx]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[idx]
            return X, Y
        else:
            return X
def train(net, x_train, y_train, x_vld, y_vld, optimizer, device, num_epochs, batch_size):
    print('开始训练。。。')   
    net = net.to(device)
    print("training on ", device)
    
    # 损失函数
    loss = torch.nn.CrossEntropyLoss()
    
    transform = transforms.Compose([         
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
        transforms.RandomRotation(15), # 隨機旋轉圖片
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ]
)
    
    y_train_fmt = torch.tensor(y_train, dtype=torch.long).view(-1, 1)

    train_dataset = Food11Dataset(x= x_train,y= y_train_fmt,transform=transform)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    y_vld_fmt = torch.tensor(y_vld, dtype=torch.long).view(-1, 1)
    vld_dataset = Food11Dataset(x_vld, y_vld_fmt,transform=transform)
    vld_iter = torch.utils.data.DataLoader(vld_dataset, batch_size=batch_size, shuffle=True)
#     scheduler_1 = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10,min_lr=0.000005)
    
#     min_loss =0.0
    tl = []
    vl = []
    for epoch in range(num_epochs):
        
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        
        net.train()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device).squeeze()
            y_hat = net(X)

            l = loss(y_hat, y)
            optimizer.zero_grad() # 梯度清零
            l.backward()
            optimizer.step()
#             scheduler_1.step(l)
            
            train_l_sum += l.cpu().item() #.cpu()转化成cpu张量 .item()转换成数值
            
            # torch.argmax(input, dim=None, keepdim=False)返回指定维度最大值的序号  dim=0表示列 dim=1表示行
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        print('测试集 epoch %d, loss %.4f, train acc %.3f, time %.1f sec' % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start)) 
        
        vld_acc_sum, vld_l_sum, vld_n, vld_batch_count = 0.0, 0.0, 0, 0
        net.eval()
        with torch.no_grad():
            for X_vld, y_vld in vld_iter: 
                X_vld = X_vld.to(device)
                y_vld = y_vld.to(device).squeeze()
                y_vld_hat = net(X_vld)

                l = loss(y_vld_hat, y_vld)
                vld_l_sum += l.cpu().item() #.cpu()转化成cpu张量 .item()转换成数值

                vld_acc_sum += (y_vld_hat.argmax(dim=1) == y_vld.to(device).squeeze()).sum().cpu().item()
                vld_n += y_vld.shape[0]
                vld_batch_count += 1

        print('验证集 epoch %d , loss %.4f, vld acc %.3f' % (epoch + 1, vld_l_sum / vld_batch_count, vld_acc_sum / vld_n))
        tl.append(train_l_sum / batch_count)
        vl.append(vld_l_sum / vld_batch_count)
        
        #         new_loss = format(vld_l_sum / vld_batch_count, '.4f')
#         if (epoch > 1.0 &  float(new_loss)< float(min_loss) | min_loss== 0.0):
#             min_loss = new_loss
#             data = os.path.exists('*.pt')
#             if data == True:
#                 os.remove('*.pt') 
#             name = "vgg16_{}.pt"
#             torch.save(net.state_dict(), name.format(epoch))


        
lr, num_epochs, batch_size = 0.00001, 20, 8
x_train, y_train, x_vld, y_vld, x_test = load_img(read_local=True)

# # x_train = normalization_train(x_train, device)
# # x_vld = normalization_train(x_vld, device)
# x_test = normalization_train(x_test, device)
# x_train = torch.tensor(x_train, dtype=torch.float)
# x_vld = torch.tensor(x_vld, dtype=torch.float)
# x_test = torch.tensor(x_test, dtype=torch.float)

net = vgg16
adam = torch.optim.Adam(net.parameters(), lr=lr)

tl, vl = train(net, x_train, y_train, x_vld, y_vld, adam, device, num_epochs, batch_size)
# torch.save(net.state_dict(), 'vgg16_2.pt')
# net.load_state_dict(torch.load(PATH))
# net.eval()
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])
ymock = torch.ones(3347, 1)

test_dataset = Food11Dataset(x = x_test, y = ymock,transform=transform_test)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
prediction = []
# net = net.to(device)
with torch.no_grad():
    for X_test,_ in test_iter: 
        y_test_hat = net(X_test.to(device))
        test_label = y_test_hat.argmax(dim=1).cpu().detach().numpy()
    for i in test_label:
        prediction.append(i)
    print('完成')
        
output_fpath = './output_{}.csv'

with open(output_fpath.format('food11'), 'w') as f:
    f.write('Id,Category\n')
    for i, label in  enumerate(prediction):
        f.write('{},{}\n'.format(i, label))