import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.plot([2, 4, 6, 8],
        [4, 8, 12, 16])
plt.plot([2, 4, 6, 8],
        [4, 8, 12, 16], color='red')
plt.plot([2, 4, 6, 8],
        [4, 8, 12, 16])
plt.xlabel('x', fontsize=15, color='green')   # naming x-axis
plt.ylabel('2*x', fontsize=15, color='green') # naming y-axis
plt.plot([4, 8, 12, 16]) # means this is y-axis and x-axis its assume as index
x = np.linspace(start = 0, stop = 10, num = 50) # Give value in ascending order
plt.plot(x, np.sin(x))   # (x, y)
plt.xlabel('x', fontsize=15, color='green')
plt.ylabel('Sin(x)', fontsize=15, color='green')
plt.tick_params(axis='y',
               color='red',
               labelcolor='blue',
               labelsize='xx-large')
plt.tick_params(axis='x',
               bottom=False,
               labelbottom=False)
plt.plot(x, np.sin(x), label='sin curve')
plt.xlabel('x', fontsize=15, color='green')
plt.ylabel('sin(x)', fontsize=15, color='green')
plt.legend()   # for label
plt.title('Playing with Plots')  # for title
plt.plot(x, np.sin(x), label='sin curve')
plt.xlabel('x', fontsize=15, color='green')
plt.ylabel('sin(x)', fontsize=15, color='green')
plt.legend()   # for label
plt.title('Playing with Plots')  # for title
plt.xlim(1, 5)  # limit x-axis our paramter
plt.plot(x, np.sin(x), label='sin curve')
plt.xlabel('x', fontsize=15, color='green')
plt.ylabel('sin(x)', fontsize=15, color='green')
plt.legend()   # for label
plt.title('Playing with Plots')  # for title
plt.xlim(1, 5)  # limit x-axis our paramter
plt.ylim(-1, 0.5)  # limit y-axis our paramter
x = np.linspace(start = 0, stop = 10, num = 50)
plt.plot(x, np.sin(x))
plt.plot(x, np.sin(x), label='sine curve')
plt.plot(x, np.cos(x), label='cosine curve')
plt.legend()
plt.title('Playing with Plots')
plt.plot(x, np.sin(x), label='sine curve', color='green')
plt.plot(x, np.cos(x), label='cosine curve', color='m')
plt.legend()
plt.title('Playing with Plots')
random_array = np.random.randn(20)
plt.plot(random_array,
        color='green')
plt.show()
plt.plot(random_array,
        color='green',
        linestyle=':')  # dot style graph
plt.show()
plt.plot(random_array,
        color='green',
        linestyle='--') # line style
plt.show()
plt.plot(random_array,
        color='green',
        linestyle=':')
plt.show()
plt.plot(random_array,
        color='green',
        linestyle='--',
        linewidth=3)  # line size
plt.show()
plt.plot(random_array,
        color='green',
        marker ='d') # diamond shape default size is 6
plt.show()
plt.plot(random_array,
        color='green',
        marker ='d', # diamond shape 
        markersize=10) # changing size of diamond shape
plt.show()
plt.plot(random_array,
        color='green',
        marker ='d', # diamond shape 
        markersize=10, # changing size of diamond shape
        linestyle='None') # Remove the line show only diamond shape
plt.show()
plt.scatter(range(0, 20),  # Scatter plot
           random_array,
           color='green',
           marker='d')
plt.show()
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.show()
type(ax)
# This for comparison between two graphs
fig = plt.figure()
ax1 = fig.add_axes([0, 0.6, 1, 0.4])
ax2 = fig.add_axes([0, 0, 0.8, 0.4])
plt.show()
x = np.linspace(start = 0, stop = 10, num = 50)
fig = plt.figure()

ax1 = fig.add_axes([0, 0.6, 1, 0.4])
ax2 = fig.add_axes([0, 0, 0.8, 0.4])

ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))

plt.show()
fig = plt.figure()

ax1 = fig.add_axes([0, 0.6, 1, 0.4])
ax2 = fig.add_axes([0, 0, 0.8, 0.4])

ax1.plot(x, np.sin(x))
ax1.set_xlabel('x', fontsize=15, color='r')
ax1.set_ylabel('sin(x)', fontsize=15, color='r')

ax2.plot(x, np.cos(x))
ax2.set_xlabel('x', fontsize=15, color='r')
ax2.set_ylabel('cos(x)', fontsize=15, color='r')

plt.show()
# Figure inside another figure
fig = plt.figure()
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.5, 0.5, 0.4, 0.4])
plt.show()
# Figure inside another figure
fig = plt.figure(figsize=(8,8))  # change size of the fig
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.5, 0.5, 0.4, 0.4])
plt.show()
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(221) # create 2*2 figure and 1 represent no of the figure
plt.show()
type(ax1)
isinstance(ax1, matplotlib.axes._axes.Axes)
fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(221) # create 2*2 figure and 1 represent no of the figure
ax1.plot([1, 2, 3, 4],
        [2, 4, 6, 8])

ax2 = fig.add_subplot(222) # create 2*2 figure and 2 represent no of the figure
ax2.plot(x, np.sin(x))
fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(221) # create 2*2 figure and 1 represent no of the figure
ax1.plot([1, 2, 3, 4],
        [2, 4, 6, 8])

ax2 = fig.add_subplot(222) # create 2*2 figure and 2 represent no of the figure
ax2.plot(x, np.sin(x))

ax3 = fig.add_subplot(223) # create 2*2 figure and 3 represent no of the figure
ax3.plot(x, np.cos(x))
fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(221) # create 2*2 figure and 1 represent no of the figure
ax1.plot([1, 2, 3, 4],
        [2, 4, 6, 8])

ax2 = fig.add_subplot(222) # create 2*2 figure and 2 represent no of the figure
ax2.plot(x, np.sin(x))

ax3 = fig.add_subplot(224) # create 2*2 figure and 4 represent no of the figure
ax3.plot(x, np.cos(x))


ax1 = plt.subplot2grid((2, 3), (0, 0))
ax1.plot(x, np.sin(x))
ax1.set_label('sine curve')

ax2 = plt.subplot2grid((2, 3), (0, 1))
ax2.plot(x, np.cos(x))
ax2.set_label('cosine curve')

ax3 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
ax3.plot([1, 2, 3, 4],
        [2, 4, 5, 8])
ax3.set_label('straight line')
ax3.yaxis.tick_right()

ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
ax4.plot(x, np.exp2(x))
ax4.set_label('exponential curve')
# A simple way to get a figure with one set of axes
fig, ax = plt.subplots()
type(fig)
type(ax)
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
ax.text(1, 4, 'Do not distribute',
       fontsize=30,
       color='red',
       ha='left',   # horizontal alignment
       va='bottom', # vertical alignment
       alpha=0.5)
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
ax.text(1, 4, 'Do not distribute',
       fontsize=30,
       color='red',
       ha='right',   # horizontal alignment
       va='top', # vertical alignment
       alpha=0.5)
fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(221)
ax1.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
ax1.set_label('straight line')
ax1.text(1, 4, 'Do not distribute',
       fontsize=20,
       color='red',
       ha='left',   # horizontal alignment
       va='bottom', # vertical alignment
       alpha=0.5)
fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(221)
ax1.plot([1, 2, 3, 4],
       [2, 4, 6, 8])
ax1.set_label('straight line')
ax1.text(1, 4, 'Do not distribute',
       fontsize=20,
       color='red',
       ha='left',   # horizontal alignment
       va='bottom', # vertical alignment
       alpha=0.5)

ax2 = fig.add_subplot(222)
ax2.plot(x, np.sin(x))

ax3 = fig.add_subplot(223)
ax3.plot(x, np.cos(x))

from subprocess import check_output
import pandas as pd
print(check_output(["ls", "../input"]).decode("utf8"))
stock_data = pd.read_csv('../input/matplotlib/stocks.csv')
stock_data.head()
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.head()
fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])
fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])

ax1.plot(stock_data['Date'],
        stock_data['AAPL'],
        color='green')
ax1.set_title('AAPL vs IBM (inset)')
fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])

ax1.plot(stock_data['Date'],
        stock_data['AAPL'],
        color='green')
ax1.set_title('AAPL vs IBM (inset)')

ax2.plot(stock_data['Date'],
        stock_data['IBM'],
        color='blue')
fig = plt.figure(figsize=(10, 6))
fig.suptitle('Stock price comparison 2007-2017', fontsize=20)

ax1 = fig.add_subplot(221)
ax1.set_title('MSFT')
ax1.plot(stock_data['Date'],
        stock_data['MSFT'],
        color='green')

ax2 = fig.add_subplot(222)
ax2.set_title('GOOG')
ax2.plot(stock_data['Date'],
        stock_data['GOOG'],
        color='purple')

ax3 = fig.add_subplot(223)
ax3.set_title('SBUX')
ax3.plot(stock_data['Date'],
        stock_data['SBUX'],
        color='magenta')

ax3 = fig.add_subplot(224)
ax3.set_title('CVX')
ax3.plot(stock_data['Date'],
        stock_data['CVX'],
        color='orange')
import matplotlib.patches as patches
fig, ax = plt.subplots()
print(fig)
print(ax)
ax.add_patch(     # A patch in matplotlib represents 2D objects
    patches.Rectangle(
        (0.1, 0.1),  # (left edge, bottom edge) of rectangle
        0.5,         # width of the rectangle
        0.5,         # height of the rectagle
        fill=False   # not fill with the color
    )
)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

ax.add_patch(     # A patch in matplotlib represents 2D objects
    patches.Rectangle(
        (0.1, 0.1),  # (left edge, bottom edge) of rectangle
        0.5,         # width of the rectangle
        0.5,         # height of the rectagle
        fill=False   # not fill with the color
    )
)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

ax.add_patch(     # A patch in matplotlib represents 2D objects
    patches.Rectangle(
        (0.1, 0.1),  # (left edge, bottom edge) of rectangle
        0.5,         # width of the rectangle
        0.5,         # height of the rectagle
        facecolor='yellow',
        edgecolor='green'
    )
)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.3, 0.6,
    hatch='.'
    ),
    patches.Rectangle((0.5, 0.1), 0.3, 0.6,
    hatch='\\',
    fill=False
    )
]:
    ax.add_patch(p)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, alpha=None,
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, alpha=1.0,
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, alpha=0.6,
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, alpha=0.1,
    )
]:
    ax.add_patch(p)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, alpha=None,
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, alpha=1.0,
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, alpha=0.6,
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, alpha=0.1,
    )
]:
    ax.add_patch(p)

ax.set_xlim(0, 1.5)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, facecolor=None # Default color blue
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, facecolor='none' # Not fill color
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, facecolor='red'  # fill red color
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, facecolor='#00ffff' # fill blue
    )
]:
    ax.add_patch(p)

ax.set_xlim(0, 1.5)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, fill=False, edgecolor=None
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, fill=False, edgecolor='none'
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, fill=False, edgecolor='red'
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, fill=False, edgecolor='#00ffff'
    )
]:
    ax.add_patch(p)

ax.set_xlim(0, 1.5)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Rectangle((0.1, 0.1), 0.2, 0.6, fill=False, linestyle='solid'  # Default
    ),
    patches.Rectangle((0.4, 0.1), 0.2, 0.6, fill=False, linestyle='dashed'
    ),
    patches.Rectangle((0.7, 0.1), 0.2, 0.6, fill=False, linestyle='dashdot'
    ),
    patches.Rectangle((1.0, 0.1), 0.2, 0.6, fill=False, linestyle='dotted'
    )
]:
    ax.add_patch(p)

ax.set_xlim(0, 1.5)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

for p in [
    patches.Circle((0.1, 0.4), 0.1,
    hatch='/'
    ),
    patches.Circle((0.5, 0.4), 0.1,
    hatch='*',
    facecolor='red'
    ),
    patches.Circle((0.9, 0.4), 0.1,
    hatch='\\',
    facecolor='green'
    ),
    patches.Circle((0.5, 0.7), 0.1,
    hatch='//',
    fill=False
    )
]:
    ax.add_patch(p)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

polygon = patches.Polygon([[0.1, 0.1],
                           [0.2, 0.8],
                           [0.5, 0.7],
                           [0.8, 0.1],
                           [0.4, 0.3]],
                          fill=False)

ax.add_patch(polygon)
plt.show()
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')

polygon = patches.Polygon([[0.1, 0.1],
                           [0.2, 0.8],
                           [0.5, 0.7],
                           [0.8, 0.1],
                           [0.4, 0.3]],
                          closed=False,
                          fill=False)

ax.add_patch(polygon)
plt.show()
# Arrow is polygon with seven sides
fig, ax = plt.subplots()
ax.set_aspect(aspect='equal')
polygon = patches.Arrow(0.1, 0.2, # centre of the bace of the arrow
                       0.7, 0.7)
ax.add_patch(polygon)
plt.show()
from matplotlib.path import Path
fig, ax = plt.subplots()
p = patches.PathPatch(Path([(0.1, 0.1), (0.8, 0.8), (0.8, 0.1), (0.4, 0.2)],
                              [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]),
                     fill=None)
ax.add_patch(p)
plt.show()
from matplotlib.path import Path
fig, ax = plt.subplots()
p = patches.PathPatch(Path([(0.1, 0.1), (0.8, 0.8), (0.8, 0.1), (0.4, 0.2)],
                              [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]),
                     fill=None)
ax.add_patch(p)
plt.show()
from matplotlib.path import Path
fig, ax = plt.subplots()
p = patches.PathPatch(Path([(0.1, 0.1), (0.8, 0.8), (0.8, 0.1), (0.4, 0.2)],
                              [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3]),
                     fill=None)
ax.add_patch(p)
plt.show()
from matplotlib.path import Path
fig, ax = plt.subplots()
p = patches.PathPatch(Path([(0.1, 0.1), (0.8, 0.8), (0.8, 0.1), (0.4, 0.2)],
                              [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.MOVETO]),
                     fill=None)
ax.add_patch(p)
plt.show()
fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('min value',
           xy=(1, 2),                   # points to datapoint
           xytext=(1.5, 2.0),           # where to write the text
           arrowprops=dict(color='g'))  # Arrow color
plt.show()
fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('min value',
           xy=(1, 2),                   # points to datapoint
           xytext=(1, 3),               # where to write the text
           arrowprops=dict(color='g'))  # Arrow color
plt.show()
fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('min value',
           xy=(1, 2),                   # points to datapoint
           xytext=(1, 3),               # where to write the text
           arrowprops=dict(facecolor='y', edgecolor='green', alpha=0.3))
plt.show()
fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('Significant point',
           xy=(2, 4),                   
           xytext=(2.0, 2.5),               
           arrowprops=dict(color='green')
           )
ax.plot([2], [4], 'ro')
plt.show()
fig, ax = plt.subplots()
ax.plot([1, 2, 3],
       [2, 4, 6])

ax.annotate('Significant point',
           xy=(2, 4),                   
           xytext=(2.0, 2.5),               
           arrowprops=dict(color='green', shrink=0.1)
           )
ax.plot([2], [4], 'ro')
plt.show()
import numpy as np

x1 = -1 + np.random.randn(100)
y1 = -1 + np.random.randn(100)

x2 = 1 + np.random.randn(100)
y2 = 1 + np.random.randn(100)
fig, ax = plt.subplots()
ax.scatter(x1, y1, color='r')
ax.scatter(x1, y2, color='g')
plt.show()
fig, ax = plt.subplots()
ax.scatter(x1, y1, color='r')
ax.scatter(x1, y2, color='g')

bbox_props = dict(boxstyle='square', facecolor='w', alpha=0.5)
ax.text(-2, -2, 'Sample A', ha='center', va='center', size=20, bbox=bbox_props)
ax.text(0, 2, 'Sample B', ha='center', va='center', size=20, bbox=bbox_props)
fig, ax = plt.subplots()
ax.scatter(x1, y1, color='r')
ax.scatter(x1, y2, color='g')

bbox_props = dict(boxstyle='square', facecolor='w', alpha=0.5)
ax.text(-2, -2, 'Sample A', ha='center', va='center', size=20, bbox=bbox_props)
ax.text(0, 2, 'Sample B', ha='center', va='center', size=20, bbox=bbox_props)

arrow_bbox_props= dict(boxstyle='rarrow',
                      facecolor='#EBF5FB',
                      edgecolor='b',
                      linewidth=2,
                      alpha=0.7)
ax.text(0, 0,
       'Direction',
       ha='center',
       va='center',
       rotation=45,
       size=15,
       bbox=arrow_bbox_props)
y = np.random.uniform(low=0.0, high=1000, size=(1000,))
y.sort()
x = np.arange(len(y))
plt.plot(x, y)
plt.grid(True)
plt.show()
plt.plot(x, y)
plt.grid(True)
plt.yscale('log')
plt.show()
plt.plot(x, y)
plt.grid(True)
plt.yscale('log', basey=2) # base log2
plt.show()
plt.plot(x, y)
plt.grid(True)
plt.yscale('log', basey=2) # base log2 on y-axis
plt.xscale('log', basex=2) # base log2 on x-axis
plt.show()
austin_weather = pd.read_csv('../input/matplotlib/austin_weather1.csv')
austin_weather.head()
austin_weather = austin_weather[['Date', 'TempAvgF', 'WindAvgMPH']].head(30)
austin_weather
fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False) # disable scale in x-axis
fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')
fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
ax_wind = ax_tempF.twinx() # The function creates another Y axis using the same X axis
fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
ax_wind = ax_tempF.twinx() # The function creates another Y axis using the same X axis

ax_wind.set_ylabel('Avg wind Speed (MPH)',
                  color='blue',
                  size='x-large')
ax_wind.tick_params(axis='y',
                  labelcolor='blue',
                  labelsize='large')
fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
ax_wind = ax_tempF.twinx() # The function creates another Y axis using the same X axis

ax_wind.set_ylabel('Avg wind Speed (MPH)',
                  color='blue',
                  size='x-large')
ax_wind.tick_params(axis='y',
                  labelcolor='blue',
                  labelsize='large')

ax_wind.plot(austin_weather['Date'],
            austin_weather['WindAvgMPH'],
            color='blue')
def fahrenheit2celsius(f):
    return (f - 32) * 5 / 9
fig, ax_tempF = plt.subplots()

fig.set_figwidth(12)
fig.set_figheight(6)

ax_tempF.set_xlabel('Date')

ax_tempF.tick_params(axis='x', bottom=False, labelbottom=False)
ax_tempF.set_ylabel('Temp (F)', color='red', size='x-large')

ax_tempF.tick_params(axis='y', labelcolor='red', labelsize='large')

ax_tempF.plot(austin_weather['Date'],
                            austin_weather['TempAvgF'],
             color='red')
ax_tempC = ax_tempF.twinx() # The function creates another Y axis using the same X axis

ymin, ymax = ax_tempF.get_ylim()

ax_tempC.set_ylim(fahrenheit2celsius(ymin),
                  fahrenheit2celsius(ymax))

ax_tempC.tick_params(axis='y',
                    labelcolor='blue',
                    labelsize='large')
x = np.random.randint(low=0, high=20, size=20)
x.sort()
x
plt.boxplot(x)
plt.show()
x = np.append(x, 22)
plt.boxplot(x)
plt.show()
x = np.append(x, 37)
x = np.append(x, 40)
plt.boxplot(x)
plt.show()
plt.boxplot(x, vert=False) # print in vertical
plt.show()
plt.boxplot(x, vert=False, notch=True)
plt.show()
plt.boxplot(x, vert=False, notch=True, showfliers=False)
plt.show()
# The default boxplot is a line2D object which only allows formatting of the edges.
# By setting patch_artist-True, we turn it into a 2D patch
bp = plt.boxplot(x, patch_artist=True)
bp
bp = plt.boxplot(x, patch_artist=True)
bp['boxes'][0].set(facecolor='lightyellow', edgecolor='maroon', hatch='.')
bp = plt.boxplot(x, patch_artist=True)
bp['boxes'][0].set(facecolor='lightyellow', 
                   edgecolor='maroon', hatch='.')
bp['whiskers'][0].set(color='red',
                     linewidth=2)
bp['whiskers'][1].set(color='blue')
bp = plt.boxplot(x, patch_artist=True)
bp['boxes'][0].set(facecolor='lightyellow', 
                   edgecolor='maroon', hatch='/')
bp['fliers'][0].set(marker='D', 
                   markerfacecolor='blue')
bp = plt.boxplot(x, patch_artist=True)
bp['boxes'][0].set(facecolor='lightyellow', 
                   edgecolor='maroon')
bp['medians'][0].set(linestyle='--',
                    linewidth=3)
print(check_output(["ls", "../input/score-of-exams/"]).decode("utf8"))
exam_data = pd.read_csv('../input/score-of-exams/exams.csv')
exam_data.head()
exam_scores = exam_data[['math score', 'reading score', 'writing score']]
exam_scores.head()
exam_scores.describe()
exam_scores = np.array(exam_scores) # convet the dataframe to array
bp = plt.boxplot(exam_scores)
plt.show()
bp = plt.boxplot(exam_scores, patch_artist=True)
plt.show()
colors = ['blue', 'grey', 'lawngreen']
bp = plt.boxplot(exam_scores, patch_artist=True)

for i in range(len(bp['boxes'])):
    bp['boxes'][i].set(facecolor=colors[i])
    bp['caps'][2*i+1].set(color=colors[i])

plt.show()
bp = plt.boxplot(exam_scores, patch_artist=True)

for i in range(len(bp['boxes'])):
    bp['boxes'][i].set(facecolor=colors[i])
    bp['caps'][2*i+1].set(color=colors[i])

plt.xticks([1, 2, 3], ['Math', 'Reading', 'Writing']) 
plt.show()
vp = plt.violinplot(exam_scores)
plt.show()
vp = plt.violinplot(exam_scores, showmedians=True)
plt.xticks([1, 2, 3], ['Math', 'Reading', 'Writing'])
plt.show()
vp = plt.violinplot(exam_scores, showmedians=True, vert=False)
plt.yticks([1, 2, 3], ['Math', 'Reading', 'Writing'])
plt.show()
vp
vp = plt.violinplot(exam_scores, showmedians=True, vert=False)
plt.yticks([1, 2, 3], ['Math', 'Reading', 'Writing'])

for i in range(len(vp['bodies'])):
    vp['bodies'][i].set(facecolor=colors[i])

plt.show()
vp = plt.violinplot(exam_scores, showmedians=True, vert=False)
plt.yticks([1, 2, 3], ['Math', 'Reading', 'Writing'])

for i in range(len(vp['bodies'])):
    vp['bodies'][i].set(facecolor=colors[i])

vp['cmaxes'].set(color='maroon')
vp['cmins'].set(color='black')
vp['cbars'].set(linestyle=':')
vp['cmedians'].set(linewidth=6)
    
plt.show()
vp = plt.violinplot(exam_scores, showmedians=True, vert=False)
plt.yticks([1, 2, 3], ['Math', 'Reading', 'Writing'])

for i in range(len(vp['bodies'])):
    vp['bodies'][i].set(facecolor=colors[i])

plt.legend(handles = [vp['bodies'][0], vp['bodies'][1]],
           labels = ['Math', 'Reading'],
           loc = 'upper left')
np_data = pd.read_csv('../input/matplotlib/national_parks.csv')
np_data.head()
np_data.describe()
plt.hist(np_data['GrandCanyon'],
        facecolor='cyan',
        edgecolor='blue',
        bins=10)
plt.show()
n, bins, patches = plt.hist(np_data['GrandCanyon'],
                            facecolor='cyan',
                            edgecolor='blue',
                            bins=10)
print('n: ', n)  # frequency of the data point
print('bins: ', bins)  # the middel value of the bin
print('patches: ', patches)
n, bins, patches = plt.hist(np_data['GrandCanyon'],
                            facecolor='cyan',
                            edgecolor='blue',
                            bins=10,
                            density=True)
print('n: ', n)  # frequency of the data point
print('bins: ', bins)  # the middel value of the bin
print('patches: ', patches)
n, bins, patches = plt.hist(np_data['GrandCanyon'],
                            facecolor='cyan',
                            edgecolor='blue',
                            bins=10,
                           cumulative=True)
plt.show()
data = pd.read_csv('../input/matplotlib/sector_weighting.csv')
data
plt.pie(data['Percentage'],
       labels=data['Sector'])
plt.show()
plt.pie(data['Percentage'],
       labels=data['Sector'])
plt.axis('equal')  # perfect circle
plt.show()
colors = ['deeppink', 'aqua', 'magenta', 'silver', 'lime']
plt.pie(data['Percentage'],
       labels=data['Sector'],
       colors=colors,  # color for each sector
       autopct='%.2f') # represents the format for the displyaed values
plt.axis('equal')
plt.show()
plt.pie(data['Percentage'],
       labels=data['Sector'],
       colors=colors,  # color for each sector
       autopct='%.2f', # represents the format for the displyaed values
       startangle=90,  # start angle
       counterclock=False)
plt.axis('equal')
plt.show()
explode = (0, 0.1, 0, 0.3, 0)
plt.pie(data['Percentage'],
       labels=data['Sector'],
       colors=colors,  # color for each sector
       autopct='%.2f', # represents the format for the displyaed values
       explode=explode)
plt.axis('equal')
plt.show()
wedges, texts, autotexts = plt.pie(data['Percentage'],
       labels=data['Sector'],
       colors=colors,  # color for each sector
       autopct='%.2f') # represents the format for the displyaed values
plt.axis('equal')

print('Wedges: ', wedges)
print('Texts: ', texts)
print('Autotexts: ', autotexts)
wedges, texts, autotexts = plt.pie(data['Percentage'],
                                   labels=data['Sector'],
                                   colors=colors,  # color for each sector
                                   autopct='%.2f', # represents the format for the displyaed values
                                   explode=explode)

plt.axis('equal')

wedges[1].set(edgecolor='blue', linewidth=2)
texts[1].set(family='cursive', size=20)
autotexts[1].set(weight='bold', size=15)
grand_canyon_data = pd.read_csv('../input/grand-visits/grand_canyon_visits.csv')
grand_canyon_data.head()
grand_canyon_data['NumVisits'].describe()
grand_canyon_data['NumVisits'] = grand_canyon_data['NumVisits'] / 1000 # oveflow the correlation
grand_canyon_data['NumVisits'].describe()
plt.figure(figsize=(16, 8))
plt.acorr(grand_canyon_data['NumVisits'],
         maxlags=20) # range of x-axis (-20, 20)
plt.show()
plt.figure(figsize=(16, 8))
lags, c, vlines, hline = plt.acorr(grand_canyon_data['NumVisits'],
         maxlags=20) # range of x-axis (-20, 20)
plt.show()
print('lags: ', lags, '\n')
print('c: ', c, '\n')             # correlation values
print('vlines: ', vlines, '\n')
print('hline: ', hline, '\n')
np_data = pd.read_csv('../input/matplotlib/national_parks.csv')
np_data.head()
x = np_data['Year']
y = np.vstack([np_data['Badlands'],
             np_data['GrandCanyon'],
             np_data['BryceCanyon']])
y
labels = ['Badlands',
         'GrandCanyon',
         'BryceCanyon']
plt.stackplot(x, y,
              labels=labels)
plt.legend(loc='upper left')
plt.show()
colors = ['sandybrown',
         'tomato',
         'skyblue']
plt.stackplot(x, y,
              labels=labels,
             colors=colors,
             edgecolor='grey')
plt.legend(loc='upper left')
plt.show()
np_data[['Badlands',
        'GrandCanyon',
        'BryceCanyon']] = np_data[['Badlands',
                                   'GrandCanyon',
                                   'BryceCanyon']].diff()
np_data.head()
# Stem plots are a good way to analyze fluctuating data
plt.figure(figsize=(10, 6))

plt.stem(np_data['Year'],
        np_data['Badlands'])
plt.title('Change in Number of Visitors')
plt.show()