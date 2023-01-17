import os

import numpy as np

from numpy import ma

from torch.utils.data import DataLoader

from tqdm import tqdm

from pykalman import AdditiveUnscentedKalmanFilter

from joblib import Parallel, delayed

import math
!pip install --no-index -f ../input/kaggle-l5kit pip==20.2.2 >/dev/nul

!pip install --no-index -f ../input/kaggle-l5kit -U l5kit > /dev/nul
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset

from l5kit.rasterization import build_rasterizer

from l5kit.evaluation import write_pred_csv
cfg = {

    'format_version': 4,

    'model_params': {

        'history_num_frames': 100,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },

    

    'raster_params': {

        'raster_size': [1, 1],

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

        'batch_size': 64,

        'shuffle': False,

        'num_workers': 4

    }

}
params = {

    'Q_std': 0.00039548740307155435, 

    'acc_decay': 0.9466336363139376, 

    'acc_std': 0.006214926973039985, 

    'ang_lim': 0.0, 

    'ang_speed_std': 0.17307676721270504, 

    'ang_std': 0.03379979585323599, 

    'obs_std': 0.04800698362225296, 

    'speed_std': 1.6644181926567871}
DIR_INPUT = "../input/lyft-motion-prediction-autonomous-vehicles"

os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

dm = LocalDataManager(None)
rasterizer = build_rasterizer(cfg, dm)



test_zarr = ChunkedDataset(dm.require(cfg['test_data_loader']["key"])).open()

test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]

test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)

test_dataloader = DataLoader(test_dataset, 

                             shuffle=False, 

                             batch_size=cfg['test_data_loader']["batch_size"], 

                             num_workers=cfg['test_data_loader']["num_workers"])
def f(cs, ang_rng=None):

    res = np.zeros(6)

    res[0] = cs[0] + cs[2]*np.cos(cs[3])

    res[1] = cs[1] + cs[2]*np.sin(cs[3])

    res[2] = cs[2] + cs[5]

    res[3] = cs[3] + cs[4]

    if ang_rng is not None:

        res[3] = np.clip(res[3], ang_rng[0], ang_rng[1])

    res[4] = cs[4]

    res[5] = params['acc_decay']*cs[5]

    return res



def g(cs):

    res = np.zeros(2)

    res[0] = cs[0]

    res[1] = cs[1]

    return res
timestamps = []

agent_ids = []

future_coords_offsets_pd = []



for batch_idx, data in enumerate(tqdm(test_dataloader)):

    

    history_positions = data['history_positions'].cpu().numpy()

    history_availabilities = data['history_availabilities'].cpu().numpy()

    timestamp = data["timestamp"].cpu().numpy()

    track_id = data["track_id"].cpu().numpy()

    

    def run(hp,ha,ts,ti):



        measurements = hp[::-1]



        ang_std = params['ang_std']

        Q = params['Q_std']*np.diag([1, 1, params['speed_std'], ang_std**2, params['ang_speed_std']*ang_std**2, params['acc_std']])

        m0 = measurements[-1]



        kf = AdditiveUnscentedKalmanFilter(initial_state_mean = [m0[0],m0[1],0,0,0,0], 

                                           n_dim_obs=2,

                                           transition_functions = f,

                                           observation_functions = g,

                                           transition_covariance = Q,

                                           initial_state_covariance = Q,

                                           observation_covariance = params['obs_std']**2*np.eye(2))



        X = ma.array(measurements)

        X[ha[::-1] < 0.5] = ma.masked



        z = kf.smooth(X)



        pred = np.zeros((51,6))

        pred[0] = z[0][-1]

        ang_rng = (z[0][-10:,3].min() - params['ang_lim'], z[0][-10:,3].max() + params['ang_lim'])

        for i in range(1,51):

            pred[i] = f(pred[i-1], ang_rng)

        pred = pred[1:,:2]

        

        return ts, ti, np.expand_dims(pred,0)



    res = Parallel(n_jobs=4)(delayed(run)(history_positions[i], history_availabilities[i], 

                                          timestamp[i], track_id[i]) for i in range(len(data['history_positions'])))

    

    timestamps.append(np.stack([r[0] for r in res]))

    agent_ids.append(np.stack([r[1] for r in res]))

    future_coords_offsets_pd.append(np.concatenate([r[2] for r in res]))



print(np.concatenate(future_coords_offsets_pd).shape)

write_pred_csv("submission.csv",

       timestamps=np.concatenate(timestamps),

       track_ids=np.concatenate(agent_ids),

       coords=np.concatenate(future_coords_offsets_pd),

      )