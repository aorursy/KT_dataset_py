import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np
from pathlib import Path



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))
# kudo for: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridline



def plot_one(task, ax, i,train_or_test,input_or_output):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    

    input_matrix = task[train_or_test][i][input_or_output]

    ax.imshow(input_matrix, cmap=cmap, norm=norm)

    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    

    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])

    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.set_title(train_or_test + ' '+input_or_output)

    



def plot_task(task):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """    

    num_train = len(task['train'])

    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))

    for i in range(num_train):     

        plot_one(task, axs[0,i],i,'train','input')

        plot_one(task, axs[1,i],i,'train','output')        

    plt.tight_layout()

    plt.show()        

        

    num_test = len(task['test'])

    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))

    if num_test==1: 

        plot_one(task, axs[0],0,'test','input')

        plot_one(task, axs[1],0,'test','output')     

    else:

        for i in range(num_test):      

            plot_one(task, axs[0,i],i,'test','input')

            plot_one(task, axs[1,i],i,'test','output')  

    plt.tight_layout()

    plt.show() 
def find_constant_factor(input_lengths, output_lengths):

    factors = [o / i for i, o in zip(input_lengths, output_lengths)]

    

    factor_consistent = len(list(set(factors))) == 1

    if factor_consistent:

        return factors[0]

    else:

        return None



def predict_shape(train_inputs, train_outputs, test_input):

    input_shapes = [x.shape for x in train_inputs] 

    output_shapes = [x.shape for x in train_outputs]

    

    is_same_shape_x = all([shape_i[0] == shape_o[0] for shape_i, shape_o in zip(input_shapes, output_shapes)])

    is_same_shape_y = all([shape_i[1] == shape_o[1] for shape_i, shape_o in zip(input_shapes, output_shapes)])

    if is_same_shape_x and is_same_shape_y:

        return test_input.shape, "same shape"

    

    is_output_shape_consistent = len(list(set(output_shapes))) == 1

    if is_output_shape_consistent:

        return output_shapes[0], "output shape consistent"



    x_factor = find_constant_factor([e[0] for e in input_shapes], [e[0] for e in output_shapes])

    y_factor = find_constant_factor([e[1] for e in input_shapes], [e[1] for e in output_shapes])

    if x_factor is not None and y_factor is not None:

        description = "(x_out, y_out) =  (x_in * {}, y_in * {})".format(x_factor, y_factor)

        return (test_input.shape[0] * x_factor, test_input.shape[1] * y_factor), description

    

    return None, "no suitable rule"



n_correct = 0

tasks_mistaken_train = []

tasks_unknown_train = []



for i, filename in enumerate(training_tasks):

    task_file = str(training_path / filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    inputs = [np.array(t["input"]) for t in task["train"]]

    outputs = [np.array(t["output"]) for t in task["train"]]

    test_input = np.array(task["test"][0]["input"])

    test_output_shape = np.array(task["test"][0]["output"]).shape

    

    test_output_shape_predicted, description = predict_shape(inputs, outputs, test_input)

    

    if test_output_shape == test_output_shape_predicted:

        n_correct += 1

    else:

#         print("task: ", filename)

#         print(description)

        if test_output_shape_predicted is not None:

            tasks_mistaken_train.append((filename, description, test_output_shape, test_output_shape_predicted))

#             print("**wrong: expected {} but got {}".format(test_output_shape, test_output_shape_predicted))

        else:

            tasks_unknown_train.append(filename)

print("{} out of {} correct".format(n_correct, len(training_tasks)))
for i, (filename, description, _, _) in enumerate(tasks_mistaken_train):

    task_file = str(training_path / filename)

    print(description)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

    plot_task(task)
for i, filename in enumerate(tasks_unknown_train[:3]):

    task_file = str(training_path / filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

    plot_task(task)
n_correct_eval = 0

tasks_mistaken_eval = []

tasks_unknown_eval = []



for i, filename in enumerate(evaluation_tasks):

    task_file = str(evaluation_path / filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    inputs = [np.array(t["input"]) for t in task["train"]]

    outputs = [np.array(t["output"]) for t in task["train"]]

    test_input = np.array(task["test"][0]["input"])

    test_output_shape = np.array(task["test"][0]["output"]).shape

    

    test_output_shape_predicted, description = predict_shape(inputs, outputs, test_input)

    

    if test_output_shape == test_output_shape_predicted:

        n_correct_eval += 1

    else:

#         print("task: ", filename)

#         print(description)

        if test_output_shape_predicted is not None:

            tasks_mistaken_eval.append((filename, description, test_output_shape, test_output_shape_predicted))

#             print("**wrong: expected {} but got {}".format(test_output_shape, test_output_shape_predicted))

        else:

            tasks_unknown_eval.append(filename)

print("{} out of {} correct".format(n_correct_eval, len(evaluation_tasks)))