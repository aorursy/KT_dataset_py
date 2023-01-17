from pathlib import Path

import os



data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = list(training_path.glob('*'))

evaluation_tasks = list(evaluation_path.glob('*'))

test_tasks = list(test_path.glob('*'))
import numpy as np



def get_color_set(img):

    return set(np.array(img).flatten())



def get_task_color_set(task):

    s = set()



    for problems in task.values():

        for problem in problems:

            try:

                in_, out_ = problem.values()

                

                s.update(get_color_set(in_))

                s.update(get_color_set(out_))

            except ValueError:

                in_, = problem.values()

                

                s.update(get_color_set(in_))

                

    return s
import json

from tqdm import tqdm



task_files = training_tasks



color_set = set()

for task_files in [training_tasks, evaluation_tasks, test_tasks]:

    for task_file in tqdm(task_files):

        with open(task_file, 'r') as f:

            task = json.load(f)



        color_set.update(get_task_color_set(task))

    

        

print(f'Total color labels used: {len(color_set)}.')

print(f'Color set: {color_set}')
def has_color_difference(task):

    s_in, s_out = set(), set()

    

    for problems in task.values():

        for problem in problems:

                in_, out_ = problem.values()

                

                s_in.update(get_color_set(in_))

                s_out.update(get_color_set(out_))

                

                if len(s_in.difference(s_out)) > 0:

                    return True

        

    return False

                

diff_vector = []

for task_files in [training_tasks, evaluation_tasks]:

    for task_file in tqdm(task_files):

        with open(task_file, 'r') as f:

            task = json.load(f)



        diff_vector.append(has_color_difference(task))



diff_vector = np.array(diff_vector)
print(f'{diff_vector.mean() * 100} % of tasks have different colors in the output from the ones in the input.')
def get_color_count(task):

    s = get_task_color_set(task)



    color_counts = np.zeros(10)

    

    for c in s:

        color_counts[c] += 1



    return color_counts
count_vector = np.zeros(10)

for task_files in [training_tasks, evaluation_tasks, test_tasks]:

    for task_file in tqdm(task_files):

        with open(task_file, 'r') as f:

            task = json.load(f)



        count_vector += get_color_count(task)
color_dist = count_vector / count_vector.max() # Max as zero appears everywhere.
import seaborn as sns



sns.barplot(np.arange(10), color_dist)