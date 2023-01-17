# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from random import Random

from time import time

import inspyred

import csv
display=True

# display=False 



prng = Random()

prng.seed(time())
problem = inspyred.benchmarks.Kursawe(3)

ea = inspyred.ec.emo.NSGA2(prng)
ea.variator = [inspyred.ec.variators.crossovers.blend_crossover,

               inspyred.ec.variators.gaussian_mutation]



ea.terminator = inspyred.ec.terminators.generation_termination
final_pop = ea.evolve(generator=problem.generator,

                      evaluator=problem.evaluator,

                      pop_size=100,

                      maximize=problem.maximize,

                      bounder=problem.bounder,

                      max_generations=100,

                      crossover_rate = 1,

                      mutation_rate=0.1)
if display:

    final_arc = ea.archive

    #print('Best Solutions: \n')

    #for f in final_arc:

        #print(f)

    import matplotlib.pyplot as plt

    x = []

    y = []

    for f in final_arc:

        x.append(f.fitness[0])

        y.append(f.fitness[1])

    plt.scatter(x, y, color='b')

    plt.xlabel("Function 1")

    plt.ylabel("Function 2")

    #Plot

    plt.show()
fieldnames = [ 'x1', 'x2','f1','f2']

listCandidate = []

for f in final_arc:

    listCandidate.append(f.candidate + f.fitness.values)

    

my_list = []

for values in listCandidate:

    temp = zip(fieldnames, values)

    inner_dict = dict(temp)

    my_list.append(inner_dict)
# Sampes:

pd.DataFrame(my_list).sample(10)