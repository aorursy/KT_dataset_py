from IPython.display import HTML

HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/tlThdr3O5Qo?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
!pip install pymap3d==2.1.0

!pip install -U l5kit
import l5kit, os

from l5kit.rasterization import build_rasterizer

from l5kit.configs import load_config_data

from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR

from l5kit.geometry import transform_points

from tqdm import tqdm

from collections import Counter

from l5kit.data import PERCEPTION_LABELS

from prettytable import PrettyTable

from l5kit.data import ChunkedDataset, LocalDataManager

from l5kit.dataset import EgoDataset, AgentDataset

from IPython.display import display, clear_output

import PIL

from IPython.display import display, clear_output

from IPython.display import HTML

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from PIL import Image

from datetime import time,date

import nltk

import spacy

import re
os.environ["L5KIT_DATA_FOLDER"] = "../input/lyft-motion-prediction-autonomous-vehicles"

dm = LocalDataManager()

sample_path = '../input/lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr'

sample_dataset = ChunkedDataset(sample_path)

sample_dataset.open()

print(sample_dataset)
sample_agents = sample_dataset.agents

sample_agents = pd.DataFrame(sample_agents)

sample_agents.columns = ["data"]; features = ['centroid', 'extent', 'yaw', 'velocity', 'track_id', 'label_probabilities']



for i, feature in enumerate(features):

    sample_agents[feature] = sample_agents['data'].apply(lambda x: x[i])

sample_agents.drop(columns=["data"],inplace=True)

sample_agents.head()
del sample_agents
test_path = '../input/lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr'

test_dataset = ChunkedDataset(test_path)

test_dataset.open()

print(test_dataset)
train_path = '../input/lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr'

train_dataset = ChunkedDataset(train_path)

train_dataset.open()

print(train_dataset)
valid_path = '../input/lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr'

valid_dataset = ChunkedDataset(valid_path)

valid_dataset.open()

print(valid_dataset)
cfg = {}

raster_params = {'raster_size':[512,512],

                 'pixel_size':[0.5,0.5],

                 'ego_center':[0.25,0.5],

                 'map_type':'py_semantic',

                 'satellite_map_key': 'aerial_map/aerial_map.png',

                 'semantic_map_key': 'semantic_map/semantic_map.pb',

                 'dataset_meta_key': 'meta.json',

                 'filter_agents_threshold': 0.5}

model_params ={'model_architecture': 'effnetB5',

               'history_num_frames': 0,

               'history_step_size': 1,

               'history_delta_time': 0.1,

               'future_num_frames': 50,

               'future_step_size': 1,

               'future_delta_time': 0.1}

cfg['raster_params'] = raster_params

cfg['model_params'] = model_params

rast = build_rasterizer(cfg,dm)

dataset = EgoDataset(cfg, sample_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
cfg = {}

raster_params = {'raster_size':[512,512],

                 'pixel_size':[0.5,0.5],

                 'ego_center':[0.25,0.5],

                 'map_type':'py_semantic',

                 'satellite_map_key': 'aerial_map/aerial_map.png',

                 'semantic_map_key': 'semantic_map/semantic_map.pb',

                 'dataset_meta_key': 'meta.json',

                 'filter_agents_threshold': 0.5}

model_params ={'model_architecture': 'effnetB5',

               'history_num_frames': 0,

               'history_step_size': 1,

               'history_delta_time': 0.1,

               'future_num_frames': 50,

               'future_step_size': 1,

               'future_delta_time': 0.1}

cfg['raster_params'] = raster_params

cfg['model_params'] = model_params

rast = build_rasterizer(cfg,dm)

dataset = EgoDataset(cfg, train_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
cfg = {}

raster_params = {'raster_size':[512,512],

                 'pixel_size':[0.5,0.5],

                 'ego_center':[0.25,0.5],

                 'map_type':'py_semantic',

                 'satellite_map_key': 'aerial_map/aerial_map.png',

                 'semantic_map_key': 'semantic_map/semantic_map.pb',

                 'dataset_meta_key': 'meta.json',

                 'filter_agents_threshold': 0.5}

model_params ={'model_architecture': 'effnetB5',

               'history_num_frames': 0,

               'history_step_size': 1,

               'history_delta_time': 0.1,

               'future_num_frames': 50,

               'future_step_size': 1,

               'future_delta_time': 0.1}

cfg['raster_params'] = raster_params

cfg['model_params'] = model_params

rast = build_rasterizer(cfg,dm)

dataset = EgoDataset(cfg, valid_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
cfg = {}

raster_params = {'raster_size':[512,512],

                 'pixel_size':[0.5,0.5],

                 'ego_center':[0.25,0.5],

                 'map_type':'py_semantic',

                 'satellite_map_key': 'aerial_map/aerial_map.png',

                 'semantic_map_key': 'semantic_map/semantic_map.pb',

                 'dataset_meta_key': 'meta.json',

                 'filter_agents_threshold': 0.5}

model_params ={'model_architecture': 'effnetB5',

               'history_num_frames': 0,

               'history_step_size': 1,

               'history_delta_time': 0.1,

               'future_num_frames': 50,

               'future_step_size': 1,

               'future_delta_time': 0.1}

cfg['raster_params'] = raster_params

cfg['model_params'] = model_params

rast = build_rasterizer(cfg,dm)

dataset = EgoDataset(cfg, test_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, sample_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, train_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, test_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, valid_dataset, rast)

scene_idx = 2

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

    display(PIL.Image.fromarray(im[::-1]))
cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, sample_dataset, rast)

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

    display(PIL.Image.fromarray(im[::-1]))
import matplotlib.animation as animation

cfg["raster_params"]["map_type"] = "py_satellite"

rast = build_rasterizer(cfg, dm)

dataset = EgoDataset(cfg, sample_dataset, rast)

scene_idx = 34

indexes = dataset.get_scene_indices(scene_idx)

images = []

fig = plt.figure()

for idx in indexes:

    

    data = dataset[idx]

    im = data["image"].transpose(1, 2, 0)

    im = dataset.rasterizer.to_rgb(im)

    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])

    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]

    draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

    clear_output(wait=True)

    images.append(PIL.Image.fromarray(im[::-1]))

im = plt.imshow(images[0], animated=True)

plt.axis('off')

def animate(i):

    im.set_data(images[i])

ani = animation.FuncAnimation(fig, animate, interval=100, blit=False,

                                repeat_delay=1000)

HTML(ani.to_html5_video())
import imageio

import IPython.display

imageio.mimsave("/tmp/gif.gif", images, duration=0.0001)

IPython.display.Image(filename="/tmp/gif.gif", format='png')
image = cv2.imread('../input/lyft-motion-prediction-autonomous-vehicles/aerial_map/aerial_map.png')

image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(32,32))

plt.imshow(image)
from l5kit.data.map_api import MapAPI

from l5kit.rasterization.rasterizer_builder import _load_metadata



semantic_map_filepath = dm.require(cfg["raster_params"]["semantic_map_key"])

dataset_meta = _load_metadata(cfg["raster_params"]["dataset_meta_key"], dm)

world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)



map_api = MapAPI(semantic_map_filepath, world_to_ecef)

MAP_LAYERS = ["junction", "node", "segment", "lane"]





def element_of_type(elem, layer_name):

    return elem.element.HasField(layer_name)





def get_elements_from_layer(map_api, layer_name):

    return [elem for elem in map_api.elements if element_of_type(elem, layer_name)]





class MapRenderer:

    

    def __init__(self, map_api):

        self._color_map = dict(drivable_area='#a6cee3',

                               road_segment='#1f78b4',

                               road_block='#b2df8a',

                               lane='#474747')

        self._map_api = map_api

    

    def render_layer(self, layer_name):

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_axes([0, 0, 1, 1])

        

    def render_lanes(self):

        all_lanes = get_elements_from_layer(self._map_api, "lane")

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_axes([0, 0, 1, 1])

        for lane in all_lanes:

            self.render_lane(ax, lane)

        return fig, ax

        

    def render_lane(self, ax, lane):

        coords = self._map_api.get_lane_coords(MapAPI.id_as_str(lane.id))

        self.render_boundary(ax, coords["xyz_left"])

        self.render_boundary(ax, coords["xyz_right"])

        

    def render_boundary(self, ax, boundary):

        xs = boundary[:, 0]

        ys = boundary[:, 1] 

        ax.plot(xs, ys, color=self._color_map["lane"], label="lane")

        

        

renderer = MapRenderer(map_api)

fig, ax = renderer.render_lanes()
print("scenes", sample_dataset.scenes)

print("scenes[0]", sample_dataset.scenes[0])

print("scenes", test_dataset.scenes)

print("scenes[0]", test_dataset.scenes[0])

print("scenes", train_dataset.scenes)

print("scenes[0]", train_dataset.scenes[0])

print("scenes", valid_dataset.scenes)

print("scenes[0]", valid_dataset.scenes[0])