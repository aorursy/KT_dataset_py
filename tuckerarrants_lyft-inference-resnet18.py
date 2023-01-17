import numpy as np

import torch

import os



from torch import nn, optim

from torch.utils.data import DataLoader

from torchvision.models.resnet import resnet18

from tqdm import tqdm

from typing import Dict



from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset, EgoDataset

from l5kit.rasterization import build_rasterizer

from l5kit.evaluation import write_pred_csv
cfg = {

    'format_version': 4,

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

        'raster_size': [350, 350],

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

        'num_workers': 0

    },

    

    'val_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 12,

        'shuffle': False,

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

        'max_num_steps': 25000,

        

    }

}
PATH_TO_DATA = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

os.environ["L5KIT_DATA_FOLDER"] = PATH_TO_DATA
#get test.zarr into DataLoader form

test_cfg = cfg["test_data_loader"]

dm = LocalDataManager()



rasterizer = build_rasterizer(cfg, dm)



test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()

test_mask = np.load(f"{PATH_TO_DATA}/scenes/mask.npz")["arr_0"]

test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataloader = DataLoader(test_dataset,

                             shuffle=test_cfg["shuffle"],

                             batch_size=test_cfg["batch_size"],

                             num_workers=test_cfg["num_workers"])





print(test_dataset)
class LyftModel(nn.Module):

    

    def __init__(self, cfg: Dict):

        super().__init__()

        

        self.backbone = resnet18(pretrained=False)

        

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

        

        backbone_out_features = 512



        # X, Y coords for the future positions (output shape: Bx50x2)

        num_targets = 2 * cfg["model_params"]["future_num_frames"]



        self.head = nn.Sequential(

            # nn.Dropout(0.2),

            nn.Linear(in_features=backbone_out_features, out_features=4096),

        )



        self.logit = nn.Linear(4096, out_features=num_targets)

        

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
#from training notebook

PRE_TRAINED_MODEL_PATH = "../input/lyftpretrainedmodels/model_state_24999.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = LyftModel(cfg)

model.to(device)



model_state = torch.load(PRE_TRAINED_MODEL_PATH, map_location=device)

model.load_state_dict(model_state)
#prediction loop

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

        outputs = model(inputs).reshape(targets.shape)

        

        future_coords_offsets_pd.append(outputs.cpu().numpy().copy())

        timestamps.append(data["timestamp"].numpy().copy())

        agent_ids.append(data["track_id"].numpy().copy())
#save predictions as a csv file

write_pred_csv('submission.csv',

               timestamps=np.concatenate(timestamps),

               track_ids=np.concatenate(agent_ids),

               coords=np.concatenate(future_coords_offsets_pd))