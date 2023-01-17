!pip uninstall -y typing
!pip install l5kit
import numpy as np
import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm
from typing import Dict

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.evaluation import write_pred_csv
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

# Training notebook's output.
WEIGHT_FILE = "/kaggle/input/model-resnet50/model_resnet50.pth"
cfg = {
    'format_version': 4,
    'model_params': {
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
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4
    }

}
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
def build_model(cfg: Dict) -> torch.nn.Module:
    # load pre-trained Conv2D model
    model = resnet50(pretrained=True, progress=True)

    # change input channels number to match the rasterizer's output
    num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
    num_in_channels = 3 + num_history_channels
    # change the first layer input
    model.conv1 = nn.Conv2d(
        num_in_channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    # change output size to (X, Y) * number of future states
    num_targets = 2 * cfg["model_params"]["future_num_frames"]
    model.fc = nn.Linear(in_features=2048, out_features=num_targets)

    return model


def forward(data, model, device, criterion):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)
    # not all the output steps are valid, but we can filter them out from the loss using availabilities
    loss = loss * target_availabilities
    loss = loss.mean()
    return loss, outputs
criterion = nn.MSELoss(reduction="none")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = build_model(cfg)
model.to(device)

model_state = torch.load(WEIGHT_FILE, map_location=device)
model.load_state_dict(model_state)
# ===== INIT DATASET
test_cfg = cfg["test_data_loader"]

# Rasterizer
rasterizer = build_rasterizer(cfg, dm)

# Test dataset/dataloader
test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataloader = DataLoader(test_dataset,
                             shuffle=test_cfg["shuffle"],
                             batch_size=test_cfg["batch_size"],
                             num_workers=test_cfg["num_workers"])


print(test_dataloader)
# # ==== EVAL LOOP
# model.eval()
# torch.set_grad_enabled(False)

# # store information for evaluation
# future_coords_offsets_pd = []
# timestamps = []
# agent_ids = []

# progress_bar = tqdm(test_dataloader)
# for data in progress_bar:
#     _, ouputs = forward(data, model, device, criterion)
    
#     # convert agent coordinates into world offsets
#     agents_coords = ouputs.cpu().numpy()
#     world_from_agents = data["world_from_agent"].numpy()
#     centroids = data["centroid"].numpy()
#     coords_offset = []
    
#     for agent_coords, world_from_agent, centroid in zip(agents_coords, world_from_agents, centroids):
#         coords_offset.append(transform_points(agent_coords, world_from_agent) - centroid[:2])
    
#     future_coords_offsets_pd.append(np.stack(coords_offset))
#     timestamps.append(data["timestamp"].numpy().copy())
#     agent_ids.append(data["track_id"].numpy().copy())
model.eval()

future_coords_offsets_pd = []
timestamps = []
agent_ids = []

with torch.no_grad():
    dataiter = tqdm(test_dataloader)
    
    for data in dataiter:
        inputs = data["image"].to(device)
        target_availabilities = data["target_availabilities"].unsqueeze(-1).to(device)
        targets = data["target_positions"].to(device)

        _, outputs = forward(data, model, device, criterion)
        
        future_coords_offsets_pd.append(outputs.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())
write_pred_csv('submission.csv',
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd))
