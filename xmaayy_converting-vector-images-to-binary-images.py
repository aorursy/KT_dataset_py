import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from PIL import Image
import matplotlib.pyplot as plt
import os
draw_type="airplane"
out_dir = "./images/{0}/".format(draw_type)

data = pd.read_csv("../input/train_simplified/{0}.csv".format(draw_type))
data.head()
def get_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end
 
    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points
# First get the drawing data, of course
drawings = data["drawing"].values
drawing_ids = data["key_id"].values

# Make a new output directory for the images
try:
    os.makedirs(out_dir)
except:
    pass

# This is set to range(1) so you can see what an output image looks like, but to actually run it change it
# to :                                      which will iterate though all the drawings in the specified csv
#               range(len(drawings))
for draw_idx in range(1):
    # Next get it out of the nasty 'line by line' format that actually contains useful data about the drawing and
    # just make it into a large array with regex :)
    coords = re.findall(r"\[[^\[\]]+\]", drawings[draw_idx])

    # Again split off each line into its co-ordinates with more string ops
    sets = []
    for co_set in coords: 
        sets.append(np.int_(co_set.strip('[ ]').split(',')))

    # Initialize
    image = np.zeros([255,255])
    pixels = []
    endpair = []
    
    # While we still have sets of co-ordinates left
    while len(sets) > 0:
        x = sets.pop() # Get the X set
        y = sets.pop() # Get the Y set
        
        # Put it into [x1, y1] form because thats how I like it
        #             [x2, y2] 
        pairs = np.hstack([np.transpose(x).reshape(len(x),1), np.transpose(y).reshape(len(y),1)])
        
        # You could heavily optimize this by running all the line generation in parallel but thats
        # out of the scope of this terrible kernel
        for i in range(len(pairs)-1):
            pixels.extend(get_line(pairs[i],pairs[i+1]))
            
    # Set each pixel in the image to 1 if its one of the co-ordintates 
    # I'm not yet well versed enough in python to do this properly, so again here is a place you could optimize
    for pixel in pixels:
        image[pixel[0]-1,pixel[1]-1]=1

    # The fruit of our labor
    fig = plt.imshow(image, cmap='binary')
    
    # And if you want to save it and use all these as pre-processed images for a network
    plt.imsave(os.path.join(out_dir,'{0}.png'.format(drawing_ids[draw_idx])), image, cmap='binary')
