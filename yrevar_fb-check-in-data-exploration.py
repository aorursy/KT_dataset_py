import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bokeh.plotting import figure, show, output_notebook

%matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 15.0) # set default size of plots
output_notebook()
train_dir = "../input"
train_file = "train.csv"

fbcheckin_train_tbl = pd.read_csv(os.path.join(train_dir, train_file))
# Few statistics
fbcheckin_train_stats_df = fbcheckin_train_tbl.describe()
fbcheckin_train_stats_df
num_train = len(fbcheckin_train_tbl)
print("Train samples: {}".format(num_train))
print("Unique places: {}".format(fbcheckin_train_tbl.place_id.unique().size))
print("Avg samples per places: {}".format(num_train/float(fbcheckin_train_tbl.place_id.unique().size)))
# Sort by place_id
fbcheckin_train_tbl = fbcheckin_train_tbl.sort_values(by="place_id")
# Take few samples for the visualization
sample_fbcheckin_train_tbl = fbcheckin_train_tbl[:10000].copy()
ax = sample_fbcheckin_train_tbl.plot(kind='hexbin', x='x', y='y', C='place_id', colormap='RdYlGn')
ax.set_xlabel("GPS-X")
ax.set_ylabel("GPS-Y")
ax.set_title("Topology of a few places users checked-in based on their last GPS co-ordinates")
ax = sample_fbcheckin_train_tbl.plot(kind='hexbin', x='x', y='y', C='accuracy')
ax.set_xlabel("GPS-X")
ax.set_ylabel("GPS-Y")
ax.set_title("Accuracy of the GPS locations")
acc_min, acc_max = fbcheckin_train_tbl["accuracy"].min(), fbcheckin_train_tbl["accuracy"].max()
print("Locations with accuracy above average: {}%".format(
        sum(sample_fbcheckin_train_tbl["accuracy"] > (acc_max-acc_min)/2.0)*100/float(sample_fbcheckin_train_tbl.shape[0])))
place_id = sample_fbcheckin_train_tbl.place_id.unique()[7]
df_place = fbcheckin_train_tbl[fbcheckin_train_tbl["place_id"]==place_id]

fig, ax = plt.subplots()
cax = plt.scatter(df_place["x"], df_place["y"], c=df_place["accuracy"], s=150.0, cmap=plt.cm.Reds)
cbar = fig.colorbar(cax, ticks=[df_place["accuracy"].min(), 
                        (df_place["accuracy"].max()+df_place["accuracy"].min())/2, df_place["accuracy"].max()])
print("X min:{}, max:{}, var:{}".format(df_place["x"].min(), df_place["x"].max(), df_place["x"].var()))
print("Y min:{}, max:{}, var:{}".format(df_place["y"].min(), df_place["y"].max(), df_place["y"].var()))
place_id = sample_fbcheckin_train_tbl.place_id.unique()[7]
df_place = fbcheckin_train_tbl[fbcheckin_train_tbl["place_id"]==place_id]

x_wt = df_place["accuracy"]*df_place["x"]
x_wt_mean = x_wt.sum()/float(sum(df_place["accuracy"]))

y_wt = df_place["accuracy"]*df_place["y"]
y_wt_mean = y_wt.sum()/float(sum(df_place["accuracy"]))

fig, ax = plt.subplots()
cax = plt.scatter(df_place["x"], df_place["y"], c=df_place["accuracy"], s=150.0, cmap=plt.cm.Reds)
cbar = fig.colorbar(cax, ticks=[df_place["accuracy"].min(), 
                        (df_place["accuracy"].max()+df_place["accuracy"].min())/2, df_place["accuracy"].max()])
plt.plot(x_wt_mean, y_wt_mean, "x", c="red", markersize=40)
plt.plot(df_place["x"].mean(), df_place["y"].mean(), "x", c="green", markersize=40)
# bokeh plot: x, y, accuracy
colors = [
    "#%02x%02x%02x" % (int((place % (2**24)) >> 16 & 0x0000FF), 
                           int((place % (2**24)) >> 8 & 0x0000FF), 
                           int((place % (2**24)) & 0x0000FF)) for place in sample_fbcheckin_train_tbl["place_id"]
]

acc_min, acc_max  = fbcheckin_train_tbl["accuracy"].min(), fbcheckin_train_tbl["accuracy"].max()
circle_rad_max = 1.5

radii = [
    
     circle_rad_max*acc/(acc_max-acc_min) for acc in sample_fbcheckin_train_tbl["accuracy"]
]

p = figure(title = "Places sample distribution over (x,y)")
p.xaxis.axis_label = 'x'
p.yaxis.axis_label = 'y'

p.circle(sample_fbcheckin_train_tbl["x"], sample_fbcheckin_train_tbl["y"],
         radius=radii,fill_color=colors, fill_alpha=0.2, size=10)

show(p)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sample_fbcheckin_train_tbl["x"], sample_fbcheckin_train_tbl["y"],
           sample_fbcheckin_train_tbl["accuracy"], c=sample_fbcheckin_train_tbl["place_id"])

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Accuracy')
plt.title("Places sample distribution over (x,y,accuracy)")
plt.show()
