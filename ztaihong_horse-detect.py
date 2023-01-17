root = '/kaggle/input/weizmann-horse-database/weizmann_horse_db'
# 定义weizmann horse database数据集类，主要实现__getitem__和__len__方法

import os

import numpy as np

import torch

from PIL import Image



class HorseDataset(object):

    def __init__(self, root, transforms):

        self.root = root

        self.transforms = transforms

        # 获取所有图像的文件名称，排序是为了保证样本图片和标记图片在列表中的位置一一对应

        self.imgs = list(sorted(os.listdir(os.path.join(root, "horse"))))

        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))



    def __getitem__(self, idx):

        # 加载一个样本图片

        img_path = os.path.join(self.root, "horse", self.imgs[idx])

        img = Image.open(img_path).convert("RGB")

        

        # 加载对应的标记图片，我们没有将标记图片转换为RGB，因为标记图片为二值图片，0表示背景，1表示马匹实例

        mask_path = os.path.join(self.root, "mask", self.masks[idx])

        mask = Image.open(mask_path)

        

        # 将PIL图像转换为numpy array

        mask = np.array(mask)

        # 马匹实例用不同编码值（第一个实例编码为1，第二个实例编码为2...）

        obj_ids = np.unique(mask)

        # 第一个id为背景，所以将其剔除

        obj_ids = obj_ids[1:]



        # 将不同编码值的标记分割为二值化的标记集合

        masks = mask == obj_ids[:, None, None]



        # 获取每个标记的矩形边界坐标

        num_objs = len(obj_ids)

        boxes = []

        for i in range(num_objs):

            pos = np.where(masks[i])

            xmin = np.min(pos[1])

            xmax = np.max(pos[1])

            ymin = np.min(pos[0])

            ymax = np.max(pos[0])

            boxes.append([xmin, ymin, xmax, ymax])



        # 转换为torch.Tensor

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 我们只有一个马匹分类

        labels = torch.ones((num_objs,), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)



        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # 没有重叠的实例

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["masks"] = masks

        target["image_id"] = image_id

        target["area"] = area

        target["iscrowd"] = iscrowd



        if self.transforms is not None:

            img, target = self.transforms(img, target)



        return img, target



    def __len__(self):

        return len(self.imgs)
# 使用Mask R-CNN网络可以实现马匹检测和实例分割

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor





def get_model_instance_segmentation(num_classes):

    # 加载早COCO数据集预训练的Mask R-CNN实例分割网络模型

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)



    # 获取分类器输入特征的数量

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换预训练模型的头部

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    # 获取掩蔽标记分类器输入特征的数量

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256

    # 替换掩蔽标记预测器

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,

                                                       hidden_layer,

                                                       num_classes)



    return model
# 安装必要的工具

!pip install pycocotools

!git clone https://github.com/pytorch/vision.git

!cp vision/references/detection/utils.py ./

!cp vision/references/detection/transforms.py ./

!cp vision/references/detection/coco_eval.py ./

!cp vision/references/detection/engine.py ./

!cp vision/references/detection/coco_utils.py ./
# 数据增强

import transforms as T



def get_transform(train):

    transforms = []

    transforms.append(T.ToTensor())

    if train:

        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)
# 迁移学习训练模型

from engine import train_one_epoch, evaluate

import utils



# 使用我们的数据集并指定是否进行数据增强

dataset = HorseDataset(root, get_transform(train=True))

dataset_test = HorseDataset(root, get_transform(train=False))



# 将数据集分割为训练集和测试集

torch.manual_seed(1)

indices = torch.randperm(len(dataset)).tolist()

# 总共327个样本，取218作为训练集，109作为测试集

dataset = torch.utils.data.Subset(dataset, indices[:-109])

dataset_test = torch.utils.data.Subset(dataset_test, indices[-109:])



# 定义训练、校验数据加载器

data_loader = torch.utils.data.DataLoader(

    dataset, batch_size=2, shuffle=True, num_workers=4,

    collate_fn=utils.collate_fn)



data_loader_test = torch.utils.data.DataLoader(

    dataset_test, batch_size=1, shuffle=False, num_workers=4,

    collate_fn=utils.collate_fn)



# 选择gpu或cpu进行训练

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# 我们的数据只用两个类别：背景、马匹

num_classes = 2



# 模型实例化

model = get_model_instance_segmentation(num_classes)



# 指定训练设备

model.to(device)



# 构造优化器

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005,

                            momentum=0.9, weight_decay=0.0005)



# 添加学习速率调度器，该调度器没训练3趟降低学习速率10倍

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,

                                               step_size=3,

                                               gamma=0.1)



# 总共训练10趟

num_epochs = 10



for epoch in range(num_epochs):

    # 一趟训练，每10此迭代输出一次训练信息

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

    # 更新学习速率

    lr_scheduler.step()

    # 用测试集对模型进行评估

    evaluate(model, data_loader_test, device=device)



print("训练结束!")
# 进行预测



# 从测试集取出一张图片

img, _ = dataset_test[0]



# 将模型置为评估模式

model.eval()



# 推理预测

with torch.no_grad():

    prediction = model([img.to(device)])



# 输出预测信息

prediction



# 输出测试图片

Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())



# 输出预测结果（掩蔽图片）

Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())