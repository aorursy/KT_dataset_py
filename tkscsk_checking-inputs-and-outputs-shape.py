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
tasks_input_shape_consistent = []

tasks_output_shape_consistent = []

tasks_both_shape_consistent = []

tasks_input_and_output_same_shape = []

tasks_others = []



for i, filename in enumerate(training_tasks):

    task_file = str(training_path / filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    input_shapes = [np.array(t["input"]).shape for t in task["train"]]

    output_shapes = [np.array(t["output"]).shape for t in task["train"]]

    is_input_shape_consistent = len(list(set(input_shapes))) == 1

    is_output_shape_consistent = len(list(set(output_shapes))) == 1

    if is_input_shape_consistent:

        tasks_input_shape_consistent.append(task_file)

    if is_output_shape_consistent:

        tasks_output_shape_consistent.append(task_file)

    if is_input_shape_consistent and is_output_shape_consistent:

        tasks_both_shape_consistent.append(task_file)

        

    is_input_and_output_same_shape = all([shape_i == shape_o for shape_i, shape_o in zip(input_shapes, output_shapes)])

    if is_input_and_output_same_shape:

        tasks_input_and_output_same_shape.append(task_file)

        

    if not (is_input_shape_consistent or is_output_shape_consistent or is_input_and_output_same_shape):

        tasks_others.append(task_file)



        

for i, filename in enumerate(evaluation_tasks):

    task_file = str(evaluation_path / filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    input_shapes = [np.array(t["input"]).shape for t in task["train"]]

    output_shapes = [np.array(t["output"]).shape for t in task["train"]]

    is_input_shape_consistent = len(list(set(input_shapes))) == 1

    is_output_shape_consistent = len(list(set(output_shapes))) == 1

    if is_input_shape_consistent:

        tasks_input_shape_consistent.append(task_file)

    if is_output_shape_consistent:

        tasks_output_shape_consistent.append(task_file)

    if is_input_shape_consistent and is_output_shape_consistent:

        tasks_both_shape_consistent.append(task_file)

        

    is_input_and_output_same_shape = all([shape_i == shape_o for shape_i, shape_o in zip(input_shapes, output_shapes)])

    if is_input_and_output_same_shape:

        tasks_input_and_output_same_shape.append(task_file)

        

    if not (is_input_shape_consistent or is_output_shape_consistent or is_input_and_output_same_shape):

        tasks_others.append(task_file)
print("all tasks:                   {:4d}".format(len(training_tasks)))

print("input shape consistent:      {:4d}".format(len(tasks_input_shape_consistent)))

print("output shape consistent:     {:4d}".format(len(tasks_output_shape_consistent)))

print("both shape consistent:       {:4d}".format(len(tasks_both_shape_consistent)))

print("input.shape == output.shape: {:4d}".format(len(tasks_input_and_output_same_shape)))

print("others:                      {:4d}".format(len(tasks_others)))
tasks_input_shape_consistent == tasks_output_shape_consistent
all([x == y for x, y in zip(tasks_input_shape_consistent, tasks_output_shape_consistent)])
tasks_input_consistent_output_inconsistent = set(tasks_input_shape_consistent) - set(tasks_output_shape_consistent)

tasks_input_inconsistent_output_consistent = set(tasks_output_shape_consistent) - set(tasks_input_shape_consistent)



print("only input shape consistent:   {:4d}".format(len(tasks_input_consistent_output_inconsistent)))

print("only output shape consistent:  {:4d}".format(len(tasks_input_inconsistent_output_consistent)))
for i, filename in enumerate(list(tasks_input_consistent_output_inconsistent)[:5]):

    task_file = str(filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

    

    plot_task(task)
for i, filename in enumerate(list(tasks_input_inconsistent_output_consistent)[:5]):

    task_file = str(filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

    

    plot_task(task)
for i, filename in enumerate(list(tasks_others)[:10]):

    task_file = str(filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

    

    plot_task(task)
task_file = str(training_path / tasks_others[1])



with open(task_file, 'r') as f:

    task = json.load(f)



plot_task(task)
task_file = str(training_path / tasks_others[3])



with open(task_file, 'r') as f:

    task = json.load(f)



plot_task(task)
tasks_shape_inconsitent = tasks_input_consistent_output_inconsistent | tasks_input_inconsistent_output_consistent | set(tasks_others)



print("only input shape consistent:         {:4d}".format(len(tasks_input_consistent_output_inconsistent)))

print("only output shape consistent:        {:4d}".format(len(tasks_input_inconsistent_output_consistent)))

print("both input and output inconsistent:  {:4d}".format(len(tasks_others)))

print("----")

print("union of those three:                {:4d}".format(len(tasks_shape_inconsitent)))
tasks_height_larger = set()

tasks_width_larger = set()

tasks_height_smaller = set()

tasks_width_smaller = set()

tasks_height_same = set()

tasks_width_same = set()

tasks_height_larger_or_same = set()

tasks_width_larger_or_same = set()

tasks_height_smaller_or_same = set()

tasks_width_smaller_or_same = set()



for i, t in enumerate(tasks_shape_inconsitent):

    task_file = str(training_path / t)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    input_shapes = [np.array(t["input"]).shape for t in task["train"]]

    output_shapes = [np.array(t["output"]).shape for t in task["train"]]



    if all([i[0] < o[0] for i, o in zip(input_shapes, output_shapes)]):

        tasks_height_larger.add(t)

    if all([i[1] < o[1] for i, o in zip(input_shapes, output_shapes)]):

        tasks_width_larger.add(t) 

    if all([i[0] > o[0] for i, o in zip(input_shapes, output_shapes)]):

        tasks_height_smaller.add(t)

    if all([i[1] > o[1] for i, o in zip(input_shapes, output_shapes)]):

        tasks_width_smaller.add(t)

    if all([i[0] == o[0] for i, o in zip(input_shapes, output_shapes)]):

        tasks_height_same.add(t)

    if all([i[1] == o[1] for i, o in zip(input_shapes, output_shapes)]):

        tasks_width_same.add(t)

    if all([i[0] <= o[0] for i, o in zip(input_shapes, output_shapes)]):

        tasks_height_larger_or_same.add(t)

    if all([i[1] <= o[1] for i, o in zip(input_shapes, output_shapes)]):

        tasks_width_larger_or_same.add(t) 

    if all([i[0] >= o[0] for i, o in zip(input_shapes, output_shapes)]):

        tasks_height_smaller_or_same.add(t)

    if all([i[1] >= o[1] for i, o in zip(input_shapes, output_shapes)]):

        tasks_width_smaller_or_same.add(t)
tasks_large_small_consitent = tasks_height_larger | tasks_width_larger | tasks_height_smaller | tasks_width_smaller | tasks_height_same | tasks_width_same

print(len(tasks_large_small_consitent))
tasks_large_small_inconsitent = tasks_shape_inconsitent - (tasks_height_larger | tasks_width_larger | tasks_height_smaller | tasks_width_smaller | tasks_height_same | tasks_width_same)

for i, filename in enumerate(tasks_large_small_inconsitent):

    task_file = str(training_path / filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

    

    plot_task(task)
tasks_large_small_or_same_consitent = tasks_height_larger_or_same | tasks_width_larger_or_same | tasks_height_smaller_or_same | tasks_width_smaller_or_same

print(len(tasks_large_small_or_same_consitent))
tasks_smaller_one_fits_in_larger_one = set()



for i, t in enumerate(tasks_shape_inconsitent):

    task_file = str(training_path / t)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

        

    input_shapes = [np.array(t["input"]).shape for t in task["train"]]

    output_shapes = [np.array(t["output"]).shape for t in task["train"]]

        

    larger_one_shapes = [i if i[0] * i[1] >= o[0] * o[1] else o for i, o in zip(input_shapes, output_shapes)]

    smaller_one_shapes = [i if i[0] * i[1] < o[0] * o[1] else o for i, o in zip(input_shapes, output_shapes)]

    

    if all([l[0] >= s[0] and l[1] >= s[1] for l, s in zip(larger_one_shapes, smaller_one_shapes)]):

        tasks_smaller_one_fits_in_larger_one.add(t)



len(tasks_smaller_one_fits_in_larger_one)
for i, filename in enumerate(tasks_shape_inconsitent - tasks_smaller_one_fits_in_larger_one):

    task_file = str(training_path / filename)

    

    with open(task_file, 'r') as f:

        task = json.load(f)

    

    plot_task(task)