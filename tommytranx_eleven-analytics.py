# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
import json
import ast
import pandas as pd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
images_analytics = '../input/11-xhec/images_analytics/'
masks_analytics =  '../input/11-xhec/masks_analytics/'
labels_analytics_poly = '../input/11-xhec/poly/poly/' #shape: bitmap + origin point
labels_analytics_people = '../input/11-xhec/labels_analytics_people/' #shape: bounding box left upper and right lower point
f = open(labels_analytics_people+sorted(os.listdir(labels_analytics_people))[3])
text = f.read()
dic = ast.literal_eval(text)
dic

import zlib
import base64
import cv2
def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask

#draw mask
def mask_2_image(origin, matrix, mask):
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j]:
                matrix[origin[0]+i][origin[1]+j] += 255
        



sorted_images = [images_analytics+image for image in sorted(os.listdir(images_analytics))]
sorted_labels_poly = [labels_analytics_poly+label for label in sorted(os.listdir(labels_analytics_poly))]
sorted_labels_people = [labels_analytics_people+label for label in sorted(os.listdir(labels_analytics_people))]


#get list of possible tasks
tasks ={} 
for label in sorted_labels_poly+sorted_labels_people:
    if '.ipynb' in label:
        continue
    text = open(label).read()
    dic = ast.literal_eval(text)
    for obj in dic['objects']:
        if obj['classTitle'] not in tasks:
            tasks[obj['classTitle']] = 1
        else:
            tasks[obj['classTitle']]+=1


tasks
list(tasks.keys())
#outpus CSV with ID column corresponding to each new instance of a task
#timestamp and task level table
columns = ['date','time','task', 'position','data', 'Horizontal formwork_model',
 'Vertical formwork_model',
 'Rebars_model',
 'Rebars',
 'Vertical_formwork',
 'Concrete_pump_hose',
 'Horizontal_formwork',
 'People_model']
analytics_time_serie = pd.DataFrame(columns = ['date','time','task', 'position','data', 'ID'])

#to keep track of ongoing task to increment ID for each new task
tasks = {'Concrete_pump_hose':[0,False],
                   'Horizontal formwork_model': [0, False],
                   'Rebars_model': [0, False],
                   'Vertical formwork_model': [0, False],
                    'Rebars':[0, False],
                  'Horizontal_formwork': [0, False],
                  'Vertical_formwork': [0, False],

        
        } #[task number, task is ongoing]

#for each frame
for image, label in zip(sorted_images, sorted_labels_poly):
    if '.ipynb' in label:
        continue
    
    
    
    date, time = image.split('/')[-1][:10], image.split('/')[-1][11:-4]
    objects = ast.literal_eval(open(label).read())['objects']
    
    frame_tasks = []
    for obj in objects:
        frame_tasks.append(obj['classTitle'])
        if tasks[obj['classTitle']][1] == False: #if such a task was not ongoing
            tasks[obj['classTitle']][1] = True #set it as ongoing
            tasks[obj['classTitle']][0] +=1
            
    for task in tasks.keys():
        if task not in frame_tasks: #if a task is not in current frame, then it has stopped
            tasks[task][1] = False
        
            
    #each object in a frame
    for obj in objects:
        task = obj['classTitle']
        ID = task+'_'+str(tasks[task][0])
        position = (obj['bitmap']['origin'][0], obj['bitmap']['origin'][1])
        bitmap = obj['bitmap']['data']
        analytics_time_serie = analytics_time_serie.append(pd.DataFrame([[date, time, task, position, bitmap, ID]], 
                                                                            columns =  analytics_time_serie.columns)
                                                           
                                                            )


            

    
    
pd.set_option('display.max_columns', None)

analytics_time_serie
#timestamp level table with one hot encoded task columns
analytics_time_serie.apply(lambda row: 'Concrete_pump_hose_9' in row.ID, axis = 1)
analytics_time_serie[analytics_time_serie.task == 'Concrete_pump_hose_model']
analytics_time_serie.task.value_counts()
import seaborn as sns

def animate(building_site, boxes=False):
    fig = plt.figure(figsize=(12, 6))
    ims = []
    for i in tqdm(range(len(building_site))):
        

        img = plt.imread(images_analytics+sorted(os.listdir(images_analytics))[i], format='jpg')
        im = plt.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True,
                                    repeat_delay=1000)

    return ani.to_html5_video()
animation_html = animate(sorted(os.listdir(images_analytics)))
Html_file= open("animation_analytics_images.html","w")
Html_file.write(animation_html)
Html_file.close()
heat = [[0 for i in range(1280)] for i in range(1080)]
for text_file in sorted_labels_people:
    f = open(text_file)
    text = f.read()
    dic = ast.literal_eval(text)
    objects = dic['objects']
    
    for obj in objects:
        if obj['classTitle'] == 'People_model':
            #mid_x =  (obj['points']['exterior'][0][0] + obj['points']['exterior'][1][0])/2
            #mid_y =  (obj['points']['exterior'][0][1] + obj['points']['exterior'][1][1])/2
            upper_right =  obj['points']['exterior'][1]
            lower_left =  obj['points']['exterior'][0]
            
            for i in range (lower_left[0], upper_right[0]+1):
                for j in range (lower_left[1], upper_right[1]+1):
                    try:
                        heat[j][i]+=1
                    except:
                        print(i,j)
                        

                
            
        
                
    

import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

def add(image, heat_map, alpha=0.6, display=False, save=None, cmap='coolwarm', axis='on', verbose=False):

    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    # display
    plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
        plt.savefig(save, bbox_inches='tight', pad_inches=0)

from scipy import ndimage
from skimage import io
import numpy as np

# read image
image = io.imread(sorted_images[15])

# create heat map
add(image, np.array(heat), alpha=0.7, save='heat_map.png')



