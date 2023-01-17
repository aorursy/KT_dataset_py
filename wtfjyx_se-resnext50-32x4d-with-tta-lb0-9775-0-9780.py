import os
import glob
import copy
import cv2
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
from pretrainedmodels.models import se_resnext50_32x4d, se_resnext101_32x4d
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
SEED = 233
CROP_SIZE = 96
RESIZE_SIZE = 224
BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
MAX_EPOCH = 15

TRAIN_DIR = "./train/"
TEST_DTR = "./test/"

TRAIN_CSV = pd.read_csv("train_labels.csv")
TEST_CSV = pd.read_csv("sample_submission.csv")
WSI_CSV = pd.read_csv("patch_id_wsi.csv")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything()

def ImageLoader(path):
    """
    load image as PIL.Image (RGB)
    """
    return Image.open(path).convert('RGB')


def read_image_list(csv_data, data_dir):
    """
    get image path list, return a list of path
    """
    image_list = [os.path.join(data_dir, '{}.tif'.format(e_i)) for e_i in csv_data['id'].values]
    return image_list


def read_label_list(csv_data):
    """
    get image label list, return a list of label(0,1)
    """
    return csv_data['label'].values.reshape(-1, 1)


def get_mean_std():
    """
    calculate (avr std) of training image
    """
    if not os.path.exists("train_mean_std.npy"):
        print("Start computing statistics of 220025 images")
        dark_th = 10 / 255
        bright_th = 245 / 255
        too_dark_idx = []
        too_bright_idx = []

        x_tot = np.zeros(3)
        x2_tot = np.zeros(3)
        counted_ones = 0

        for f_path in read_image_list(TRAIN_CSV, TRAIN_DIR):
            # norm image
            imagearray = np.array(ImageLoader(f_path)).reshape(-1, 3) / 255
            if imagearray.max() < dark_th:  # image too dark?
                too_dark_idx.append(f_path)
                continue
            if imagearray.min() > bright_th:  # image too light?
                too_bright_idx.append(f_path)
                continue

            x_tot += imagearray.mean(axis=0)
            x2_tot += (imagearray ** 2).mean(axis=0)
            counted_ones += 1

        channel_avr = x_tot / counted_ones
        channel_std = np.sqrt(x2_tot / counted_ones - channel_avr ** 2)
        np.save("train_mean_std.npy", np.append(channel_avr, channel_std))

        print("Computing finished: {} images\n".format(counted_ones), "-" * 20)
        return channel_avr, channel_std

    else:
        print("-" * 30, "\nReading existed file")
        avr_std = np.load("train_mean_std.npy")
        channel_avr = avr_std[:3]
        channel_std = avr_std[3:]
        return channel_avr, channel_std


def get_x_trans_change():

    # mean=[0.485, 0.456, 0.406]
    # std=[0.229, 0.224, 0.225]

    all_mean, all_std = get_mean_std()

    x_trans = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.5),
            transforms.ColorJitter(contrast=0.5),
            transforms.ColorJitter(saturation=0.5),
            transforms.ColorJitter(hue=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]),
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((90, 90)),
            ]),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomRotation((270, 270)),
            ])
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean=all_mean, std=all_std)
    ])
    return x_trans


def tta_test():

    all_mean, all_std = get_mean_std()

    change_01 = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE))
    ])

    change_02 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=all_mean, std=all_std)
    ])

    change_list = [
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.5),

        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),

        transforms.RandomRotation((0, 0)),
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
    ]

    x_all = transforms.Lambda(
        lambda image:
        torch.stack([change_02(each(change_01(image))) for each in change_list])
    )

    return x_all


def wsi_kfold_split():
    not_in_wsi = TRAIN_CSV.set_index('id').drop(WSI_CSV.id)

    grouped_l = list(WSI_CSV.groupby(by='wsi'))

    grouped_normal = grouped_l[:128]  # normal wsi
    grouped_tumor = grouped_l[128:]  # tumor wsi

    random.shuffle(grouped_normal)
    random.shuffle(grouped_tumor)

    k_f_data = {}

    kf_5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
    not_in_wsi_kf = list(kf_5.split(not_in_wsi))

    for k in range(5):
        v_normal = grouped_normal[int(k / 5 * len(grouped_normal)):
                                  int((k + 1) / 5 * len(grouped_normal))]
        v_tumor = grouped_tumor[int(k / 5 * len(grouped_tumor)):
                                int((k + 1) / 5 * len(grouped_tumor))]  # validation

        t_normal = [_ for _ in grouped_normal if _ not in v_normal]
        t_tumor = [_ for _ in grouped_tumor if _ not in v_tumor]  # train

        temp_v = [_v[1] for _v in v_normal] + [_v[1] for _v in v_tumor]
        temp_t = [_t[1] for _t in t_normal] + [_t[1] for _t in t_tumor]

        random.shuffle(temp_t)
        random.shuffle(temp_v)

        wsi_k_t = pd.concat(temp_t)  # wsi train
        wsi_k_v = pd.concat(temp_v)  # wsi valid

        img_k_train = pd.merge(wsi_k_t, TRAIN_CSV, how="inner",
                               left_on="id", right_on="id").drop(["wsi"], axis=1)

        img_k_valid = pd.merge(wsi_k_v, TRAIN_CSV, how="inner",
                               left_on="id", right_on="id").drop(["wsi"], axis=1)

        # not in WSI
        img_k_train = pd.concat([not_in_wsi.iloc[not_in_wsi_kf[k][0]].reset_index(),
                                 img_k_train])
        img_k_valid = pd.concat([not_in_wsi.iloc[not_in_wsi_kf[k][1]].reset_index(),
                                 img_k_valid])

        k_f_data[k] = {"train": img_k_train,
                       "val": img_k_valid}

    return k_f_data


class MyDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, loader=ImageLoader):
        self.image_data = read_image_list(csv_file, data_dir)
        self.label_data = read_label_list(csv_file)

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        x = self.image_data[index]
        label = self.label_data[index]

        img = self.loader(x)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_data)


TRANS = get_x_trans_change()
TRANS_test = tta_test()

KF_Dataset = {}
KF_length = {}
k_wsi_csv = wsi_kfold_split()

for cnt in range(5):
    print(cnt + 1)

    train_part = k_wsi_csv[cnt]["train"]
    val_part = k_wsi_csv[cnt]["val"]

    train_data = MyDataset(csv_file=train_part, data_dir=TRAIN_DIR,
                           transform=TRANS, loader=ImageLoader)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    valid_data = MyDataset(csv_file=val_part, data_dir=TRAIN_DIR,
                           transform=TRANS, loader=ImageLoader)
    valid_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=4)

    KF_Dataset[cnt] = {"train": train_data_loader,
                       "val": valid_data_loader}
    KF_length[cnt] = {"train": len(train_data),
                      "val": len(valid_data)}

    print("-" * 30, "\nTrain data size: {}\nValid data size: {}\n".format(
        len(train_part), len(val_part)))
    print("Train data batch: {}\nValid data batch: {}\n".format(
        len(train_data_loader), len(valid_data_loader)))

test_data = MyDataset(
    csv_file=TEST_CSV,
    data_dir=TEST_DTR,
    transform=TRANS_test,
    loader=ImageLoader
)

test_data_loader = DataLoader(
    test_data,
    batch_size=TEST_BATCH_SIZE,
    num_workers=4
)


class SE_ResNext_50(nn.Module):

    def __init__(self, ):
        super(SE_ResNext_50, self).__init__()

        model = se_resnext50_32x4d()
        self.model_layer = nn.Sequential(*list(model.children())[:-1])
        self.linear_layer = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.model_layer(x)  # [-1, 2048, 1, 1]

        batch = x.shape[0]
        conc = x.view(batch, -1)
        out = self.linear_layer(conc)

        return out


def train(max_epoch=MAX_EPOCH):
    for idx in range(5):
        print("CV {}/5 starts!!!".format(idx + 1))
        data_loader = KF_Dataset[idx]

        net = SE_DenseNet169_plus()
        net.to(DEVICE)

        optimizer = optim.Adam(net.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

        best_model_wts = copy.deepcopy(net.state_dict())
        best_acc = 0.0
        best_epoch = 0

        for epoch in range(max_epoch):
            print('Epoch {}/{}\n'.format(epoch, max_epoch - 1), '-' * 30)

            for phase in ["train", "val"]:

                y_true = []
                y_pred = []

                if phase == "train":
                    net.train()
                else:
                    net.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate data
                for iteration, (x, y) in tqdm(enumerate(data_loader[phase])):

                    x = x.to(DEVICE)  # FloatType
                    y = y.to(DEVICE).float()  # LongType → FloatType
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = net(x)  # outputs shape & y shape [64, 1]
                        loss = criterion(outputs, y)

                        proba = nn.Sigmoid()(outputs)
                        preds = torch.round(proba)

                        y_true.append(y.data.detach().cpu().numpy())
                        y_pred.append(proba.detach().cpu().numpy())

                        if phase == 'train':
                            # backward + optimizer
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()  # get loss number
                    running_corrects += torch.sum(preds == y.data)

                y_true = np.vstack(y_true).reshape(-1)
                y_pred = np.vstack(y_pred).reshape(-1)

                if phase == "train":
                    epoch_auc = roc_auc_score(y_true, y_pred)
                    epoch_loss = running_loss / KF_length[idx]["train"]
                    epoch_acc = running_corrects.double() / KF_length[idx]["train"]

                    scheduler.step(epoch_acc)
                else:
                    epoch_auc = roc_auc_score(y_true, y_pred)
                    epoch_loss = running_loss / KF_length[idx]["val"]
                    epoch_acc = running_corrects.double() / KF_length[idx]["val"]

                print('{} Loss: {:.4f} Acc: {:.4f}, Auc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_auc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(net.state_dict())
                if phase == 'val':
                    print("Now epoch {} is the best epoch".format(best_epoch))

        net.load_state_dict(best_model_wts)
        torch.save(net.state_dict(), "{}_cv{}_best_epoch{}.pth".format(net._get_name(), idx, best_epoch))
        print("-" * 30, "\nFinish Training cv{}, {} Epoch\nThe Best is epoch {}\n".format(idx, MAX_EPOCH, best_epoch))


def predict():
    for idx in range(5):
        net = SE_DenseNet169_plus()
        net.to("cpu")

        preds = []
        weight_path = glob.glob("SE_DenseNet169_plus_cv{}*pth".format(idx))
        if len(weight_path) != 0:
            print("-" * 30, "\nLoading weight")
            print("Test data size: {}\nTest data batch: {}\n".format(
                len(test_data), len(test_data_loader)))
            net.load_state_dict(torch.load(weight_path[0]))
        else:
            raise FileExistsError("Not exist weight file!")

        # batch_size, n_crops, c, h, w = data.size()
        # data = data.view(-1, c, h, w)
        # output = model(data)
        # output = output.view(batch_size, n_crops, -1).mean(1)

        net.to(DEVICE)
        net.eval()
        print("Start testing cv-{}".format(idx))

        with torch.no_grad():

            for batch_i, (x_test, target) in tqdm(enumerate(test_data_loader)):
                test_batch_size, n_crops, c, h, w = x_test.size()
                x_test = x_test.view(-1, c, h, w)
                x_test = x_test.to(DEVICE)
                out = net(x_test)

                batch_pred = nn.Sigmoid()(out)
                batch_pred = batch_pred.view(test_batch_size, n_crops, -1).mean(1)

                batch_pred = list(batch_pred.detach().cpu().numpy())

                preds.append(batch_pred)

        test_pred = pd.DataFrame({"imgs": test_data.image_data,
                                  "preds": np.vstack(preds).reshape(-1)})
        test_pred["imgs"] = test_pred["imgs"].apply(lambda x: x.split("/")[-1][:-4])

        sub = pd.merge(TEST_CSV, test_pred, left_on="id", right_on="imgs")
        sub = sub[['id', 'preds']]
        sub.columns = ['id', 'label']

        if os.path.exists("./output"):
            sub.to_csv("./output/{}_{}_cv{}_TTA_proba.csv".format(time.strftime("%m%d"), net._get_name(), idx))
        else:
            os.mkdir("./output")
            sub.to_csv("./output/{}_{}_cv{}_TTA_proba.csv".format(time.strftime("%m%d"), net._get_name(), idx))
        print("File Saved!")
