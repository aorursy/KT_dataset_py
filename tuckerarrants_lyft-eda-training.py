#python basics

from matplotlib import pyplot as plt

import math, os, re, time, random

import numpy as np, pandas as pd, seaborn as sns

from tqdm import tqdm

from typing import Dict



#for deep learning

import torch

from torch import nn, optim, Tensor

from torchvision.models.resnet import resnet18



#for scene visualization

from IPython.display import display, clear_output

from IPython.display import HTML

import PIL

from matplotlib import animation, rc



os.chdir('/kaggle/input')
# set env variable for data

PATH_TO_DATA = "../input/lyft-motion-prediction-autonomous-vehicles"

os.environ["L5KIT_DATA_FOLDER"] = PATH_TO_DATA
cfg = {

    'model_params': {

        'model_architecture': 'resnet18',

        

        'history_num_frames': 10,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },

    

    'raster_params': {

        'raster_size': [300, 300],

        'pixel_size': [0.5, 0.5],

        'ego_center': [0.25, 0.5],

        'map_type': 'py_semantic',

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },

    

    'train_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 32,

        'shuffle': True,

        'num_workers': 4

    },

    

    'sample_data_loader': {

        'key': 'scenes/sample.zarr',

        'batch_size': 32,

        'shuffle': True,

        'num_workers': 4

    },

    

    'test_data_loader': {

        'key': 'scenes/test.zarr',

        'batch_size': 8,

        'shuffle': False,

        'num_workers': 4

    },

    

    'train_params': {

        'checkpoint_every_n_steps': 5000,

        'max_num_steps': 25,

        'image_coords': True

        

    },

        

    'test_params': {

        'image_coords': True

        

    }

}
from l5kit.data import LocalDataManager, ChunkedDataset



dm = LocalDataManager()

train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()



#let's see what one of the objects looks like

print(train_zarr)
#frames = train_zarr.frames

#coords = np.zeros((len(frames), 2))

#for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):

    #frame = train_zarr.frames[idx_data]

    #coords[idx_coord] = frame["ego_translation"][:2]





#plt.scatter(coords[:, 0], coords[:, 1], marker='.')

#axes = plt.gca()

#axes.set_xlim([-2500, 1600])

#axes.set_ylim([-2500, 1600])
from l5kit.rasterization import build_rasterizer



rasterizer = build_rasterizer(cfg, dm)
from l5kit.dataset import EgoDataset



AV_ds = EgoDataset(cfg, train_zarr, rasterizer)
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points



sample = AV_ds[50]



im = sample["image"].transpose(1, 2, 0)

im = AV_ds.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(sample["target_positions"] + sample["centroid"][:2], sample["world_to_image"])

draw_trajectory(im, target_positions_pixels, sample["target_yaws"], TARGET_POINTS_COLOR)



_, ax = plt.subplots(figsize = (7, 7))

plt.imshow(im[::-1])

plt.show()
cfg["raster_params"]["map_type"] = "py_satellite"

rasterizer = build_rasterizer(cfg, dm)

AV_ds = EgoDataset(cfg, train_zarr, rasterizer)

sample = AV_ds[50]



im = sample["image"].transpose(1, 2, 0)

im = AV_ds.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(sample["target_positions"] + sample["centroid"][:2], sample["world_to_image"])

draw_trajectory(im, target_positions_pixels, sample["target_yaws"], TARGET_POINTS_COLOR)



_, ax = plt.subplots(figsize = (7, 7))

plt.imshow(im[::-1])

plt.show()
from l5kit.dataset import AgentDataset



cfg["raster_params"]["map_type"] = "py_semantic"

A_ds = AgentDataset(cfg, train_zarr, rasterizer)

sample = A_ds[50]



im = sample["image"].transpose(1, 2, 0)

im = A_ds.rasterizer.to_rgb(im)

target_positions_pixels = transform_points(sample["target_positions"] + sample["centroid"][:2], sample["world_to_image"])

draw_trajectory(im, target_positions_pixels, sample["target_yaws"], TARGET_POINTS_COLOR)



_, ax = plt.subplots(figsize = (7, 7))

plt.imshow(im[::-1])

plt.show()
def animate_solution(images):

    def animate(i):

        im.set_data(images[i])

        

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    

    return animation.FuncAnimation(fig, animate, frames = len(images), interval = 60)

 

cfg["raster_params"]["map_type"] = "py_semantic"

rasterizer = build_rasterizer(cfg, dm)

A_ds = EgoDataset(cfg, train_zarr, rasterizer)

scene_idx = 34

indexes = AV_ds.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    data = A_ds[idx]

    im = data["image"].transpose(1, 2, 0)

    im = A_ds.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

    

HTML(animate_solution(images).to_jshtml())
from torch.utils.data import DataLoader



train_cfg = cfg["train_data_loader"]

rasterizer = build_rasterizer(cfg, dm)

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset,

                              shuffle = cfg["train_data_loader"]["shuffle"],

                              batch_size = cfg["train_data_loader"]["batch_size"],

                              num_workers = cfg["train_data_loader"]["num_workers"])



print(len(train_dataloader))
class LyftModel(nn.Module):



    def __init__(self, cfg: Dict, num_modes=3):

        super().__init__()



        architecture = cfg["model_params"]["model_architecture"]

        backbone = eval(architecture)(pretrained=True, progress=True)

        self.backbone = backbone



        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels



        self.backbone.conv1 = nn.Conv2d(

            num_in_channels,

            self.backbone.conv1.out_channels,

            kernel_size=self.backbone.conv1.kernel_size,

            stride=self.backbone.conv1.stride,

            padding=self.backbone.conv1.padding,

            bias=False,

        )

        

        if architecture == "resnet50":

            backbone_out_features = 2048

        else:

            backbone_out_features = 512



        # X, Y coords for the future positions (output shape: batch_sizex50x2)

        self.future_len = cfg["model_params"]["future_num_frames"]

        num_targets = 2 * self.future_len



        # You can add more layers here.

        self.head = nn.Sequential(

            # nn.Dropout(0.2),

            nn.Linear(in_features=backbone_out_features, out_features=4096),

        )



        self.num_preds = num_targets * num_modes

        self.num_modes = num_modes



        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)



    def forward(self, x):

        x = self.backbone.conv1(x)

        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)



        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        x = self.backbone.layer4(x)



        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)



        x = self.head(x)

        x = self.logit(x)

        

        return x
def forward(data, model, device, criterion):

    inputs = data["image"].to(device)

    target_availabilities = data["target_availabilities"].to(device)

    targets = data["target_positions"].to(device)

    matrix = data["world_to_image"].to(device)

    centroid = data["centroid"].to(device)[:,None,:].to(torch.float)



    # Forward pass

    outputs = model(inputs)



    bs,tl,_ = targets.shape

    assert tl == cfg["model_params"]["future_num_frames"]



    if cfg['train_params']['image_coords']:

        targets = targets + centroid

        targets = torch.cat([targets,torch.ones((bs,tl,1)).to(device)], dim=2)

        targets = torch.matmul(matrix.to(torch.float), targets.transpose(1,2))

        targets = targets.transpose(1,2)[:,:,:2]

        rs = cfg["raster_params"]["raster_size"]

        ec = cfg["raster_params"]["ego_center"]



        bias = torch.tensor([rs[0] * ec[0], rs[1] * ec[1]])[None, None, :].to(device)

        targets = targets - bias



    confidences, pred = outputs[:,:3], outputs[:,3:]

    pred = pred.view(bs, 3, tl, 2)

    assert confidences.shape == (bs, 3)

    confidences = torch.softmax(confidences, dim=1)



    loss = criterion(targets, pred, confidences, target_availabilities)

    loss = torch.mean(loss)



    if cfg['train_params']['image_coords']:

        matrix_inv = torch.inverse(matrix)

        pred = pred + bias[:,None,:,:]

        pred = torch.cat([pred,torch.ones((bs,3,tl,1)).to(device)], dim=3)

        pred = torch.stack([torch.matmul(matrix_inv.to(torch.float), pred[:,i].transpose(1,2)) 

                            for i in range(3)], dim=1)

        pred = pred.transpose(2,3)[:,:,:,:2]

        pred = pred - centroid[:,None,:,:]



    return loss, pred, confidences
# from https://www.kaggle.com/corochann/lyft-training-with-multi-mode-confidence

def pytorch_neg_multi_log_likelihood_batch(

    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor

) -> Tensor:



    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"

    batch_size, num_modes, future_len, num_coords = pred.shape



    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"

    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"

    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"

    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"

    assert torch.isfinite(pred).all(), "invalid value found in pred"

    assert torch.isfinite(gt).all(), "invalid value found in gt"

    assert torch.isfinite(confidences).all(), "invalid value found in confidences"

    assert torch.isfinite(avails).all(), "invalid value found in avails"



    # convert to (batch_size, num_modes, future_len, num_coords)

    gt = torch.unsqueeze(gt, 1)  # add modes

    avails = avails[:, None, :, None]  # add modes and cords



    # error (batch_size, num_modes, future_len)

    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    

    if cfg['train_params']['image_coords']:

        error = error / 4



    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it

        # error (batch_size, num_modes)

        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time



    # use max aggregator on modes for numerical stability

    # error (batch_size, num_modes)

    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one

    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes

    # print("error", error)

    return torch.mean(error)





def pytorch_neg_multi_log_likelihood_single(

    gt: Tensor, pred: Tensor, avails: Tensor

) -> Tensor:



    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)

    # create confidence (bs)x(mode=1)

    batch_size, future_len, num_coords = pred.shape

    confidences = pred.new_ones((batch_size, 1))

    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LyftModel(cfg).to(device)

optimizer = optim.Adam(model.parameters(), lr = 1e-3)

criterion = pytorch_neg_multi_log_likelihood_batch
if device.type == 'cpu': print('Training on CPU')

if device.type == 'cuda': print('Training on GPU')
tr_it = iter(train_dataloader)



progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))

losses_train = []



for itr in progress_bar:



    try:

        data = next(tr_it)

    except StopIteration:

        tr_it = iter(train_dataloader)

        data = next(tr_it)



    model.train()

    torch.set_grad_enabled(True)



    loss, pred, confidences = forward(data, model, device, criterion)



    # Backward pass

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    losses_train.append(loss.item())

        

    #save model during training

    if (itr+1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not DEBUG:

        torch.save(model.state_dict(), f'model_state_{itr}.pth')

    

    #display training progress

    progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
test_cfg = cfg["test_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



# Test dataset/dataloader

test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()

test_mask = np.load(f"{PATH_TO_DATA}/scenes/mask.npz")["arr_0"]

test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataloader = DataLoader(test_dataset,

                             shuffle=test_cfg["shuffle"],

                             batch_size=test_cfg["batch_size"],

                             num_workers=test_cfg["num_workers"])





print(test_dataset)
def run_prediction(predictor, data_loader):

    predictor.eval()



    pred_coords_list = []

    confidences_list = []

    timestamps_list = []

    track_id_list = []



    with torch.no_grad():

        dataiter = tqdm(data_loader)

        for data in dataiter:

            image = data["image"].to(device)

            inputs = data["image"].to(device)

            target_availabilities = data["target_availabilities"].to(device)

            targets = data["target_positions"].to(device)

            matrix = data["world_to_image"].to(device)

            centroid = data["centroid"].to(device)[:,None,:].to(torch.float)

            

            pred, confidences = predictor(image)

            

            if cfg['test_params']['image_coords']:

                matrix_inv = torch.inverse(matrix)

                pred = pred + bias[:,None,:,:]

                pred = torch.cat([pred,torch.ones((bs,3,tl,1)).to(device)], dim=3)

                pred = torch.stack([torch.matmul(matrix_inv.to(torch.float), pred[:,i].transpose(1,2)) 

                                    for i in range(3)], dim=1)

                pred = pred.transpose(2,3)[:,:,:,:2]

                pred = pred - centroid[:,None,:,:]



            pred_coords_list.append(pred.cpu().numpy().copy())

            confidences_list.append(confidences.cpu().numpy().copy())

            timestamps_list.append(data["timestamp"].numpy().copy())

            track_id_list.append(data["track_id"].numpy().copy())

            

    timestamps = np.concatenate(timestamps_list)

    track_ids = np.concatenate(track_id_list)

    coords = np.concatenate(pred_coords_list)

    confs = np.concatenate(confidences_list)

    return timestamps, track_ids, coords, confs
LOAD_MODEL = False



if LOAD_MODEL:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predictor = LyftModel(cfg)

    saved_model_path = ""

    predictor.load_state_dict(torch.load(saved_model_path))

    predictor.to(device)
from l5kit.evaluation import write_pred_csv



INFER = False



if INFER:

    timestamps, track_ids, coords, confs = run_prediction(predictor, test_dataloader)

    write_pred_csv('submission.csv',timestamps=timestamps,

    track_ids=track_ids, coords=coords, confs=confs)
train_dataset[0].keys()
def sample_dataset():

    # Build Rasterizer

    rasterizer = build_rasterizer(cfg, dm)

    

    train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

    return train_dataset[100]
sizes = [[200, 200], [224, 224], [250, 250], [350, 350], [450, 450], [500, 500]]



f, ax = plt.subplots(2, 3, figsize=(20, 12))

ax = ax.flatten()



for i in range(6):

    cfg['raster_params']['raster_size'] = sizes[i]

    sample = sample_dataset()

    

    ax[i].imshow(sample['image'][-3:].transpose(1, 2, 0))

    ax[i].get_xaxis().set_visible(False)

    ax[i].get_yaxis().set_visible(False)

    ax[i].set_title(f"Raster size: {sizes[i]}")



#reset to default

cfg['raster_params']['raster_size'] = [224, 224]
sizes = [[.2, .2,], [.3, .3], [.4, .4], [.5, .5], [.6, .6], [.7, .7]]



f, ax = plt.subplots(2, 3, figsize=(20, 12))

ax = ax.flatten()



for i in range(6):

    cfg['raster_params']['pixel_size'] = sizes[i]

    sample = sample_dataset()

    

    ax[i].imshow(sample['image'][-3:].transpose(1, 2, 0))

    ax[i].get_xaxis().set_visible(False)

    ax[i].get_yaxis().set_visible(False)

    ax[i].set_title(f"Pixel size: {sizes[i]}")



#reset to default

cfg['raster_params']['pixel_size'] = [0.5, 0.5]
sizes = [[[200, 200],[.5, .5]], [[250, 250],[.4, .4]], [[350, 350],[.3, .3]], [[500, 500],[.2, .2]]]



f, ax = plt.subplots(1, 4, figsize=(20, 12))

ax = ax.flatten()



for i in range(4):

    cfg['raster_params']['pixel_size'] = sizes[i][1]

    cfg['raster_params']['raster_size'] = sizes[i][0]

    sample = sample_dataset()

    

    ax[i].imshow(sample['image'][-3:].transpose(1, 2, 0))

    ax[i].get_xaxis().set_visible(False)

    ax[i].get_yaxis().set_visible(False)

    ax[i].set_title(f"Raster/Pixel size: {sizes[i]}")



#reset to default

cfg['raster_params']['raster_size'] = [224, 224]
sizes = [[[400, 400],[.2, .2]], [[500, 500],[.2, .2]], [[600, 600],[.2, .2]], [[700, 700],[.2, .2]]]



f, ax = plt.subplots(1, 4, figsize=(20, 12))

ax = ax.flatten()



for i in range(4):

    cfg['raster_params']['pixel_size'] = sizes[i][1]

    cfg['raster_params']['raster_size'] = sizes[i][0]

    sample = sample_dataset()

    

    ax[i].imshow(sample['image'][-3:].transpose(1, 2, 0))

    ax[i].get_xaxis().set_visible(False)

    ax[i].get_yaxis().set_visible(False)

    ax[i].set_title(f"Raster/Pixel size: {sizes[i]}")
#for augmentations

import albumentations as A



#for better visualization

cfg['raster_params']['raster_size'] = [500, 500]

cfg['raster_params']['pixle_size'] = [.25, .25]
def show_images(aug_cfg):

    dm = LocalDataManager()

    train_zarr = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"])).open()

    cfg["raster_params"]["map_type"] = 'py_semantic'

    rasterizer = build_rasterizer(cfg, dm)

    sem_ds = AgentDataset(cfg, train_zarr, rasterizer)

    cfg["raster_params"]["map_type"] = 'py_satellite'

    rasterizer = build_rasterizer(cfg, dm)

    sat_ds = AgentDataset(cfg, train_zarr, rasterizer)



    #get a random sample

    random_index = int(np.random.random()*len(AV_ds))

    sat_sample = sat_ds[random_index]

    sem_sample = sem_ds[random_index]



    sat_im = sat_sample["image"].transpose(1, 2, 0)

    sat_im = sat_ds.rasterizer.to_rgb(sat_im)

    sem_im = sem_sample["image"].transpose(1, 2, 0)

    sem_im = sem_ds.rasterizer.to_rgb(sem_im)

    

    fig, ax = plt.subplots(len(aug_cfg), 2, figsize=(15,15))

    

    for i, (key, aug) in enumerate(aug_cfg.items()):

        if aug is None:

            ax[i, 0].imshow(sat_im[::-1])

            ax[i, 0].set_title(key)

        else:

            sat_im_ = aug(image=sat_im)['image']

            ax[i, 0].imshow(sat_im_[::-1])

            ax[i, 0].set_title(key)

            

    for i, (key, aug) in enumerate(aug_cfg.items()):

        if aug is None:

            ax[i, 1].imshow(sem_im[::-1])

            ax[i, 1].set_title(key)

        else:

            sem_im_ = aug(image=sem_im)['image']

            ax[i, 1].imshow(sem_im_[::-1])

            ax[i, 1].set_title(key)



    plt.tight_layout();
aug_dict = {'Original': None,

            

            'Cutout':A.Cutout(num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, 

                              always_apply=False, p=1),

            

            'CoarseDropout': A.CoarseDropout(max_holes=10, max_height=20, max_width=20, 

                                             min_holes=None, min_height=None, min_width=None,

                                             fill_value=0, always_apply=False, p=1),

            

            'GridDropout': A.GridDropout(ratio=.4, unit_size_min=None, unit_size_max=None,

                                         holes_number_x=None, holes_number_y=None,

                                         shift_x=0, shift_y=0, p=1)}



show_images(aug_dict)
aug_dict = {'Original': None,

            

            'RandomRain': A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20,

                                      drop_width=1,blur_value=7, brightness_coefficient=0.7,

                                      rain_type=None, always_apply=False, p=1),

            

            'RandomFog': A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08,

                                     always_apply=False, p=1),

            

            'RandomSnow': A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3,

                                        brightness_coeff=2.5, always_apply=False, p=1)}



show_images(aug_dict)