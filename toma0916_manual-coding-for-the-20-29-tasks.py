import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np



for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    

from pathlib import Path



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))

test_tasks = sorted(os.listdir(test_path))

print(len(training_tasks), len(evaluation_tasks), len(test_tasks))
cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,

# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown

plt.figure(figsize=(5, 2), dpi=200)

plt.imshow([list(range(10))], cmap=cmap, norm=norm)

plt.xticks(list(range(10)))

plt.yticks([])

plt.show()



def plot_task(task):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()

    

def get_data(task_filename):

    with open(task_filename, 'r') as f:

        task = json.load(f)

    return task



num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

color2num = {c: n for n, c in enumerate(num2color)}
def check(task, pred_func):

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        t_pred = pred_func(t_in)

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Train-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Train-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        axs[2][fig_num].imshow(t_pred, cmap=cmap, norm=norm)

        axs[2][fig_num].set_title(f'Train-{i} pred')

        axs[2][fig_num].set_yticks(list(range(t_pred.shape[0])))

        axs[2][fig_num].set_xticks(list(range(t_pred.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        t_pred = pred_func(t_in)

        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[0][fig_num].set_title(f'Test-{i} in')

        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[1][fig_num].set_title(f'Test-{i} out')

        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))

        axs[2][fig_num].imshow(t_pred, cmap=cmap, norm=norm)

        axs[2][fig_num].set_title(f'Test-{i} pred')

        axs[2][fig_num].set_yticks(list(range(t_pred.shape[0])))

        axs[2][fig_num].set_xticks(list(range(t_pred.shape[1])))

        fig_num += 1
def task_train020(x):



    H, W = x.shape

    bese_color = x[0, 0]

    row_count = 1

    column_count = 1

    for yy in range(H):

        if x[yy, 0] != bese_color:

            row_count += 1

    for xx in range(W):

        if x[0, xx] != bese_color:

            column_count += 1

    y = bese_color * np.ones((row_count, column_count))

    return y



task = get_data(str(training_path / training_tasks[20]))

check(task, task_train020)

# plot_task(task)
def task_train021(x):

    

    H, W = x.shape

    y = np.zeros((3, 3))

    

    def search_around(yy, xx):

        for dy in [-1, 0, 1]:

            for dx in [-1, 0, 1]:

                if (0 <= yy + dy < H) and (0 <= xx + dx < W) and x[yy + dy, xx + dx] != color2num['black']:

                    y[dy + 1, dx + 1] = x[yy + dy, xx + dx] 

                    

    for yy in range(H):

        for xx in range(W):

            if x[yy, xx] == color2num['gray']:

                search_around(yy, xx)

    

    return y



task = get_data(str(training_path / training_tasks[21]))

check(task, task_train021)

# plot_task(task)
# I have not yet wrote down the rule of task Train022 .

plot_task(get_data(str(training_path / training_tasks[22])))
def task_train023(x):

    

    red_pos = []

    green_pos = []

    blue_pos = []

    

    H, W = x.shape

    y = x.copy()



    for yy in range(H):

        for xx in range(W):

            if x[yy, xx] == color2num['red']:

                red_pos.append([yy, xx])

            elif x[yy, xx] == color2num['green']:

                green_pos.append([yy, xx])

            elif x[yy, xx] == color2num['blue']:

                blue_pos.append([yy, xx])

    

    for r_pos in red_pos:

        for yy in range(H):

            y[yy, r_pos[1]] = color2num['red']

    for g_pos in green_pos:

        for xx in range(W):

            y[g_pos[0], xx] = color2num['green']

    for b_pos in blue_pos:

        for xx in range(W):

            y[b_pos[0], xx] = color2num['blue']            

    

    return y



task = get_data(str(training_path / training_tasks[23]))

check(task, task_train023)

# plot_task(task)
def task_train024(x):



    if np.sum(np.sum(x, axis=1) == 0) == 0:

        is_vertical = True

    else:

        is_vertical = False

        x = x.T

        

    H, W = x.shape

    y = x.copy()

    

    lines = {}

    for xx in range(W):

        if (np.unique(x[:, xx]).shape[0] == 1) and np.sum(np.unique(x[:, xx])) != 0:

            lines[x[0, xx]] = xx

                

    for yy in range(H):

        for xx in range(W): 

            if xx in lines.values():

                continue

            if y[yy, xx] in lines.keys():

                if xx < lines[y[yy, xx]]:

                    xx_new = lines[y[yy, xx]]-1

                else:

                    xx_new = lines[y[yy, xx]]+1

                if xx != xx_new:

                    y[yy, xx_new] = y[yy, xx]

                    y[yy, xx] = color2num['black']

            else:

                y[yy, xx] = color2num['black']

    if not is_vertical:

        y = y.T        

        

    return y



task = get_data(str(training_path / training_tasks[24]))

check(task, task_train024)

# plot_task(task)
def task_train025(x):

    

    y = np.zeros((5, 3))

    

    x1 = x[:, :3]

    x2 = x[:, 4:]



    H, W = y.shape

    for yy in range(H):

        for xx in range(W):

            if x1[yy, xx] == 0 and x2[yy, xx] == 0:

                y[yy, xx] = color2num['sky']



    return y



task = get_data(str(training_path / training_tasks[25]))

check(task, task_train025)

# plot_task(task)
def task_train026(x):

    

    H, W = x.shape

    y = x.copy()

    

    x_min = np.min(np.where(0 < np.sum(y, axis=1)))

    x_max = np.max(np.where(0 < np.sum(y, axis=1))) + 1

    y_max = np.max(np.where(0 < np.sum(y, axis=0))) + 1

    y_min = y_max - (x_max - x_min)

    

    y[x_min:x_max, y_min:y_max] = np.rot90(y[x_min:x_max, y_min:y_max], 1)

    y[np.where(y-x==1)] = color2num['red'] 

    y[np.where(y-x==-1)] = color2num['blue']

    

    return y



task = get_data(str(training_path / training_tasks[26]))

check(task, task_train026)

# plot_task(task)
def task_train027(x):

    

    H, W = x.shape

    y = x.copy()

    

    for yy in range(H):

        for xx in range(W):

            if x[yy, xx] != 0:

                color = x[yy, xx]

                y[yy, :] = color

                

                if yy < H//2:

                    y[0, :] = color

                    y[:H//2, 0] = color

                    y[:H//2, -1] = color

                    cnt = 1

                else:

                    y[-1, :] = color

                    y[H//2:, 0] = color

                    y[H//2:, -1] = color



    return y



task = get_data(str(training_path / training_tasks[27]))

check(task, task_train027)

# plot_task(task)
def task_train028(x):

    

    y = x.copy()

    for color in color2num.values():

        z = x.copy()

        z[np.where(z != color)] = color2num['black']

        

        if np.where(z==color)[0].shape[0] == 0:

            continue

        

        h_min = np.min(np.where(z==color)[0])

        h_max = np.max(np.where(z==color)[0])

        w_min = np.min(np.where(z==color)[1])

        w_max = np.max(np.where(z==color)[1])

        

        if np.unique(x[h_min, w_min:w_max+1]).shape[0] != 1:

            continue

        if np.unique(x[h_max, w_min:w_max+1]).shape[0] != 1:

            continue

        if np.unique(x[h_min:h_max+1, w_min]).shape[0] != 1:

            continue    

        if np.unique(x[h_min:h_max+1, w_max]).shape[0] != 1:

            continue

        

        y = x[h_min+1:h_max, w_min+1:w_max]

        

        



    

    return y



task = get_data(str(training_path / training_tasks[28]))

check(task, task_train028)

# plot_task(task)
def task_train029(x):

    

    blue_h_ = np.where(x==color2num['blue'])[0]

    blue_h_min, blue_h_max = np.min(blue_h_), np.max(blue_h_)

    

    blue_ = x.copy()

    blue_[np.where(blue_ != color2num['blue'])] = color2num['black']

    red_ = x.copy()

    red_[np.where(red_ != color2num['red'])] = color2num['black']

    yellow_ = x.copy()

    yellow_[np.where(yellow_ != color2num['yellow'])] = color2num['black']

    

    def shift(x):

        x_ = x.copy()

        x_[0, :] = x[-1, :]

        x_[1:, :] = x[:-1, :]

        return x_

    

    def check_pos(x):

        h_ = np.where(x!=color2num['black'])[0]

        h_min, h_max = np.min(h_), np.max(h_)

        if h_min == blue_h_min and h_max == blue_h_max:

            return True

        else:

            return False

        

    while not check_pos(red_):

        red_ = shift(red_)

    while not check_pos(yellow_):

        yellow_ = shift(yellow_)

    

    y = blue_ + red_ + yellow_



    return y



task = get_data(str(training_path / training_tasks[29]))

check(task, task_train029)

# plot_task(task)