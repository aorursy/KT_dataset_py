# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from numpy.linalg import multi_dot

import math

from scipy.stats import norm

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras import layers



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Given Data

no_of_assets = 5

cor_bm = [[1.0, 0.79, 0.82, 0.91, 0.84],

    [0.79, 1.0, 0.73, 0.80, 0.76],

    [0.82, 0.73, 1.0, 0.77, 0.72],

    [0.91, 0.80, 0.77, 1.0, 0.90],

    [0.84, 0.76, 0.72, 0.90, 1.0]]

vol_list = np.array([0.518, 0.648, 0.623, 0.570, 0.530])

curr_stock = np.ones(no_of_assets)

t = 1

k = 1

no_exercise_days = 8

r = 0.05

w = np.array([0.381, 0.065, 0.057, 0.270, 0.227])



# dt = ((t*365)/no_exercise_days)/365

exercise_days = [float(i/8) for i in range(1, no_exercise_days+1)]

no_of_paths = 1000

zero_mean = np.zeros(no_of_assets)

vol_diag_mat = np.diag(vol_list)

cov_bm = vol_diag_mat * cor_bm * vol_diag_mat

dz_mat = np.random.multivariate_normal(zero_mean, cov_bm, (no_of_paths,no_exercise_days))

sim_stock_mat = np.zeros((no_of_paths, no_exercise_days + 1, no_of_assets))



#Updating current stock price as first column

for path in range(0, no_of_paths):

    sim_stock_mat[path,0] = curr_stock



#The next piece of code can be performed with parallel processing for each path - to be done later



#Updating the simulated stock prices along the paths on exercise days by exact solution method



for path in range(0, no_of_paths):

    for day in range(1, no_exercise_days + 1):

        drift = (np.add(np.full(no_of_assets, r), - 0.5 * np.square(vol_list))) * (exercise_days[day-1])

        stoch_part = ((exercise_days[day-1])**0.5) *  dz_mat[path, day-1]

        sim_stock_mat[path, day] = np.multiply(curr_stock, np.exp(np.add(drift, stoch_part)))

#Pricing of Bermudan options using Neural Networks



intrinsic_value_mat = np.zeros((no_of_paths, no_exercise_days + 1))

continuation_value_mat = np.zeros((no_of_paths, no_exercise_days + 1))

option_value_mat = np.zeros((no_of_paths, no_exercise_days + 1))



#Find Intrinsic Value for the current exercise time



for path in range(0, no_of_paths):

    for day in range(0, no_exercise_days + 1):

        intrinsic_value_mat[path, day] = max(k - np.sum((np.multiply(w,sim_stock_mat[path, day]))), 0)



option_value_mat[:,no_exercise_days] = intrinsic_value_mat[:,no_exercise_days]

 

#Find Continuation value for the current exercise time





for day in range(no_exercise_days-1,0, -1):

        #Neural Network Function using RELU activaion function

        #Build Neural Network model(using Stock prices at next exercise period and their option value)

        

        batch_size = int(no_of_paths/8)

        no_of_epochs = 500

        no_of_hidden_nodes = 16

        no_of_output_nodes = 1

        nnet_model = Sequential()

        

        nnet_model.add(Dense(no_of_hidden_nodes, input_dim=no_of_assets, activation='relu', 

                     kernel_initializer='he_normal', use_bias=True, 

                     bias_initializer='he_normal', kernel_regularizer=None, bias_regularizer=None, 

                     activity_regularizer=None, bias_constraint=None, kernel_constraint=None))

        nnet_model.add(Dense(no_of_output_nodes, input_dim=no_of_assets, activation='linear', 

                     kernel_initializer='he_normal', use_bias=True, 

                     bias_initializer='he_normal', kernel_regularizer=None, bias_regularizer=None, 

                     activity_regularizer=None, bias_constraint=None, kernel_constraint=None))

        nnet_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'accuracy'], 

                     loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

        

        nnet_output = nnet_model.fit(x=np.log(sim_stock_mat[:,day+1]), y=option_value_mat[:,day+1] , 

                      batch_size=batch_size, epochs=no_of_epochs, verbose=0)

        

        for path in range(0, no_of_paths):

        

            node_wise_expectation = np.zeros(no_of_hidden_nodes)



            for node in range(0,no_of_hidden_nodes):

                w_vect = np.array(nnet_model.layers[0].get_weights()[0])[:,node]

                hidden_bias = - np.array(nnet_model.layers[0].get_weights()[1])[node]

                dt = (exercise_days[day] - exercise_days[day-1])

                mean_sum = (r * dt) + np.add(np.log(sim_stock_mat[path,day]), (np.square(vol_list) * (dt/2)))

                mu_node = np.dot(np.transpose(w_vect), mean_sum)

                var_node = multi_dot([np.transpose(w_vect), cov_bm, w_vect]) * dt

                first_term_expectation = ((var_node ** 0.5) / ((2 * math.pi) ** 0.5)) * np.exp(- 0.5 * (((hidden_bias - mu_node) / (var_node ** 0.5) ) ** 2)) 

                cdf_factor = (hidden_bias - mu_node) / (var_node ** 0.5)

                second_term_expectation = (mu_node - hidden_bias) * (1 - (norm.cdf(cdf_factor)))

                node_wise_expectation[node] = first_term_expectation + second_term_expectation



            output_weights = np.array(nnet_model.layers[1].get_weights()[0])

            output_bias = np.array(nnet_model.layers[1].get_weights()[1])



            continuation_value_mat[path,day] = (np.dot(np.transpose(output_weights), node_wise_expectation) + output_bias) * np.exp(- r * dt)



            #Update the option value for the current exercise time

            option_value_mat[path, day] = max(intrinsic_value_mat[path, day], continuation_value_mat[path,day])



print(np.mean(option_value_mat[:,1]) * np.exp(-r * (exercise_days[0])))



