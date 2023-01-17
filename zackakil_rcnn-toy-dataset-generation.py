%matplotlib inline
import matplotlib.pyplot as plt
# %%writefile box_gen.py

import numpy as np
from PIL import Image, ImageDraw

FRAME_SIZE = [5,50]
BOX_WIDTH = 3


def get_rect(x, y, width, height):
    rect = np.array([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1), (0, 0)])
    offset = np.array([x, y])
    transformed_rect = rect + offset
    return transformed_rect

def get_array_with_box_at_pos(x):
    data = np.zeros(FRAME_SIZE)
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    rect = get_rect(x=x, y=1, width=BOX_WIDTH, height=BOX_WIDTH)
    draw.polygon([tuple(p) for p in rect], fill=1)
    new_data = np.asarray(img)
    return new_data
sway_offset = 1
sway_start = sway_offset
sway_end = (FRAME_SIZE[1]-1) - BOX_WIDTH
sway_range = sway_end - sway_offset
sway_start, sway_end, sway_range
DATA_POINTS = 100
base = (np.arange(DATA_POINTS)/DATA_POINTS)* 6 *np.pi
sined = (np.sin(base) + 1 )/2
plt.scatter(base, sined)
plt.show()
def sin_to_pos(sin_val):
    return (sin_val*sway_range)+sway_offset
frames = []

print_every_n_frames = DATA_POINTS//10
for i,t in enumerate(sined):
    frame = get_array_with_box_at_pos(sin_to_pos(t))
    if(i % print_every_n_frames)==0:
        plt.imshow(frame, interpolation='nearest')
        plt.show()
    frames.append(frame)
y = sin_to_pos(sined[1:])
X = frames[:-1]

len(X), len(y)
data = {'X':X, 'y':y}
from sklearn.externals import joblib
joblib.dump(data, 'sythetic_data.pkl')
