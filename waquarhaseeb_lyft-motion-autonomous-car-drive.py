from IPython.display import IFrame

IFrame('https://player.vimeo.com/video/389096888', width=640, height=360, frameborder="0", allow="autoplay; fullscreen", allowfullscreen=True)
# Running this pip install code takes time, we can skip it when we attach utility script correctly!

# !pip install -U l5kit
import l5kit



l5kit.__version__
import gc

import os

from pathlib import Path

import random

import sys



from tqdm.notebook import tqdm

import numpy as np

import pandas as pd

import scipy as sp





import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio

pio.templates.default = "plotly_dark"



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



# --- setup ---

pd.set_option('max_columns', 50)
from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset



from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points

from tqdm import tqdm

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable



import os



from matplotlib import animation, rc

from IPython.display import HTML



rc('animation', html='jshtml')
# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    filenames.sort()

    for filename in filenames:

        print(os.path.join(dirname, filename))
my_arr = np.zeros(3, dtype=[("color", (np.uint8, 3)), ("label", np.bool)])



my_arr[0]["color"] = [0, 218, 130]

my_arr[0]["label"] = True

my_arr[1]["color"] = [245, 59, 255]

my_arr[1]["label"] = True



my_arr
import zarr



z = zarr.open("./dataset.zarr", mode="w", shape=(500,), dtype=np.float32, chunks=(100,))



# We can write to it by assigning to it. This gets persisted on disk.

z[0:150] = np.arange(150)
print(z.info)
!ls -l ./*
print(z[::20]) # Read every 20th value
# set env variable for data

os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"

# get config

cfg = load_config_data("/kaggle/input/lyft-config-files/visualisation_config.yaml")

print(cfg)
dm = LocalDataManager()

dataset_path = dm.require('scenes/sample.zarr')

zarr_dataset = ChunkedDataset(dataset_path)

zarr_dataset.open()

print(zarr_dataset)
dataset_path
frames = zarr_dataset.frames



## This is slow.

# coords = np.zeros((len(frames), 2))

# for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):

#     frame = zarr_dataset.frames[idx_data]

#     coords[idx_coord] = frame["ego_translation"][:2]



# This is much faster!

coords = frames["ego_translation"][:, :2]



plt.scatter(coords[:, 0], coords[:, 1], marker='.')

axes = plt.gca()

axes.set_xlim([-2500, 1600])

axes.set_ylim([-2500, 1600])

plt.title("ego_translation of frames")
# 'map_type': 'py_semantic' for cfg.

semantic_rasterizer = build_rasterizer(cfg, dm)

semantic_dataset = EgoDataset(cfg, zarr_dataset, semantic_rasterizer)
def visualize_trajectory(dataset, index, title="target_positions movement with draw_trajectory"):

    data = dataset[index]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"]) #change it from original notebook

    draw_trajectory(im, target_positions_pixels, rgb_color=TARGET_POINTS_COLOR,  yaws=data["target_yaws"]) # change it from original notebook

    plt.title(title)

    plt.imshow(im[::-1])

    plt.show()



visualize_trajectory(semantic_dataset, index=0)
# map_type was changed from 'py_semantic' to 'py_satellite'.

cfg["raster_params"]["map_type"] = "py_satellite"

satellite_rasterizer = build_rasterizer(cfg, dm)

satellite_dataset = EgoDataset(cfg, zarr_dataset, satellite_rasterizer)



visualize_trajectory(satellite_dataset, index=0)
type(satellite_rasterizer), type(semantic_rasterizer)
agent_dataset = AgentDataset(cfg, zarr_dataset, satellite_rasterizer)

visualize_trajectory(agent_dataset, index=0)
from IPython.display import display, clear_output

import PIL

 

dataset = semantic_dataset

scene_idx = 34

indexes = dataset.get_scene_indices(scene_idx)

images = []



for idx in indexes:

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))
%%capture

# From https://www.kaggle.com/jpbremer/lyft-scene-visualisations by @jpbremer

def animate_solution(images):



    def animate(i):

        im.set_data(images[i])

        return (im,)

 

    fig, ax = plt.subplots()

    im = ax.imshow(images[0])

    def init():

        im.set_data(images[0])

        return (im,)

    

    return animation.FuncAnimation(fig, animate, init_func=init, frames=len(images), interval=60, blit=True)



anim = animate_solution(images)
HTML(anim.to_jshtml())