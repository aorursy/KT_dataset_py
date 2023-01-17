# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np

dir_path = os.path.dirname(os.path.abspath(''))

connector_path = '/input/abstraction-and-reasoning-challenge/'



traning_dir = dir_path + connector_path + "training"

evaluation_dir = dir_path + connector_path + "evaluation"

testing_dir = dir_path + connector_path + "test"

output_dir = dir_path+"/output/"



train_filenames = os.listdir(traning_dir)

eval_filenames = os.listdir(evaluation_dir)

test_filenames = os.listdir(testing_dir)



sample_submission = pd.read_csv(dir_path+"/input/abstraction-and-reasoning-challenge/sample_submission.csv")

from pathlib import Path



## solving abstract reasoning tasks

#  train your algorithm to acquire ARC-relevant cognitive priors



def return_arrays(json_file): #train_filenames[0]

    with open(traning_dir+"/"+json_file, 'r') as f:

        task = json.load(f)

    return task



def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred



def get_imgs(json_file):

    print(json_file)

    task = return_arrays(json_file)



    total_train_show = np.arange(len(task['train'])).tolist()

    total_test_show = np.arange(len(task['test'])).tolist()

    show_train = []

    show_test = []

    print("Train : " + str(len(total_train_show)) + " Test : " + str(len(total_test_show)))

    for r in total_train_show:

        show_train.append([task['train'][r]['input'],task['train'][r]['output']])



    for r in total_test_show:

        show_test.append([task['test'][r]['input'],task['test'][r]['output']])

    return [show_train,show_test]



print("Shape and Size of json images")





# 0 for train 1 for test

from itertools import chain

from matplotlib import colors



cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00','#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)



def plot_task(demonstration):

    fig=plt.figure(figsize=(10,10))

    columns = 2

    rows = int(np.ceil(len(demonstration) / 2))

    range = np.arange(1, columns*rows+1)

    flatten_task_demo = list(chain(*demonstration))

    idx = 0

    for i in range:

        fig.add_subplot(rows, columns, i)

        plt.imshow(flatten_task_demo[idx],cmap=cmap,norm = norm) ### what you want you can plot

        idx = idx + 1

    plt.show()
###  0 for train   ###  1 for test

show_file = '794b24be.json' # OR   train_filenames[246]

print("Showing json " + str(show_file))

task_demo = get_imgs(show_file)
print("Task Demonstration")

plot_task(task_demo[0])  

plt.show()
print("Test Inputs")

plot_task(task_demo[1])  

plt.show()