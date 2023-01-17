import cv2
import math
import random
import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# draw image function
def draw_cv2(raw_strokes, base_size = 256, img_size=128, lw=6, time_color=True):
    img = np.zeros((base_size, base_size), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if img_size != base_size:
        return cv2.resize(img, (img_size, img_size))
    else:
        return img   
# read test file
test = pd.read_csv('../input/test_simplified.csv')
# choose a random image
i = test[test['key_id'] == 9000052667981386].iloc[0]['drawing']
img = draw_cv2(ast.literal_eval(i),img_size=256)
plt.imshow(img)
# deform one stroke in a whole image
# scale is pre-defined constant
def deform_single_line(line, scale = 10, eps = 0.00001):
    x,y = line[0], line[1]
    
    # get start and end point x/y
    x_pre, y_pre = x[0],y[0]
    x_end, y_end = x[-1],y[-1]
    
    # line distance between start point & end point
    l_dis = math.sqrt((x_pre-x_end)**2 + (y_pre-y_end)**2)
    
    curve_dis = 0
    for idx in range(len(x)-1):
        sx,sy = x[idx],y[idx]
        ex,ey = x[idx+1],y[idx+1]
        
        curve_dis += math.sqrt((sx-ex)**2 + (sy-ey)**2)

    # ratio between line distance and curve distance
    ratio = float(l_dis)/(curve_dis+eps)
    if ratio > 1:
        return [x,y]
    
    # disturbance direction
    r1 = (random.uniform(0,1)<=0.5)*2-1
    r2 = (random.uniform(0,1)<=0.5)*2-1
    res_x,res_y = x.copy(),y.copy()
    
    for idx in range(1,len(x)-1):
        
        # a little move
        res_x[idx] += int(scale*r1*ratio*abs(np.random.randn()))
        res_y[idx] += int(scale*r2*ratio*abs(np.random.randn()))
        
    res_line = [res_x, res_y]
    return res_line

# deform whole image by deform each strokes
def local_deform(lines):
    res = []
    for line in lines:
        res.append(deform_single_line(line))
    return res
            
def scatter_line(line, base_size = 256, c ='b'):
    x,y = line[0], line[1]
    for idx,ele in enumerate(y):
        y[idx] = base_size - ele
    plt.scatter(x, y, c = c)
    plt.xlim(0,base_size)
    plt.ylim(0,base_size)

before_ = ast.literal_eval(i)[0]
after_ = deform_single_line((ast.literal_eval(i)[0]))

scatter_line(before_,c = 'b')
scatter_line(after_,c = 'r')
plt.show()
img = draw_cv2(local_deform(ast.literal_eval(i)),img_size=256)
plt.imshow(img)
