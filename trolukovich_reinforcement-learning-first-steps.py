import gym

import numpy as np

import matplotlib.pyplot as plt

import requests



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.initializers import he_normal



from PIL import Image

from io import BytesIO
r = requests.get('https://gym.openai.com/videos/2019-10-21--mqt8Qj1mwo/CartPole-v1/poster.jpg')



plt.figure(figsize = (10, 6))

plt.imshow(Image.open(BytesIO(r.content)))

plt.axis('off')

plt.show()
env = gym.make('CartPole-v0') # Create an environment



# To reset the environment we can use reset() function which returns an array with 4 values

# This 4 values is an observation, which tells us a position of pole. We don't need to know what these values mean

# this is a job for our ANN.

observation = env.reset()

print(f'Observation, returned by reset() function: {observation}')



# To see action space we can use action_space attribute

# Discrete(2) means that actions can be 0 or 1 which can be left or right 

print('Action space: ', env.action_space)



# To make step we need to use step(action) function, it returns 4 values:

# Obseravtion - current observation after action

# Reward - recieved reward after action

# Done - whether game over or not

# Debug data which we don't need

observation, reward, done, debug = env.step(env.action_space.sample()) # Doing random action

print(f'Observation after action: {observation}')

print(f'Reward for the action: {reward}')

print(f'Game is over: {done}')

print(f'Debug data: {debug}')
def generate_data(env = gym.make('CartPole-v0'), n_games = 1000, model = None, percentile = 70):

    '''

       env - an environment to solve

       n_games - number of games to play to collect raw data

       model - if None, the random actions will be taken to collect data, to predict actions a model must be passed

       percentile - (100% - percentile%) of the best games will be selected as training data

    '''

    observation = env.reset() # Resetting the environment to get our first observation



    train_data = [] # List to store raw data

    rewards = [] # List to store total rewards of each game

    

    print(f'Playing {n_games} games...')

    

    # Step 1 of the algorithm - Play N numbers of games using random actions or actions predicted by model to collect raw data.

    for i in range(n_games):

        temp_reward = 0 # Counts a current game total reward

        temp_data = [] # Stores (observation, action) tuples for each step

        

        # Playing a current game until done

        while True:

            # Use model to predict actions if passed, otherwise take random actions

            if model:

                action = model.predict(observation.reshape((-1, 4)))

                action = int(np.rint(action))

            else:

                action = env.action_space.sample()

            

            temp_data.append((list(observation), action)) # Appending (observation, action) tuple to temp_data list



            observation, reward, done, _ = env.step(action) # Making action



            temp_reward += reward # Counting reward

            

            # If game over - reset environment and break while loop

            if done:

                observation = env.reset()

                break

        

        # Append data of last game to train_data list and total reward of last game to rewards list

        train_data.append(temp_data)

        

        # Step 2 of the algorithm - Collect a total reward for each game and calculate threshold - 70 percentile of all total rewards.

        rewards.append(temp_reward)

        

    print('Done playing games\n')

    

    # Calculating threshold value using rewards list an np.percentile function

    thresh = int(np.percentile(rewards, percentile))

    print(f'Score threshold value: {thresh}')

    

    print(f'Selecting games according to threshold...')

    # Step 3 of the algorithm - Select games from raw data which have total reward more than threshold.

    train_data = [episode for (i, episode) in enumerate(train_data) if rewards[i] >= thresh]

    

    # Now train_data list contains lists of tuples: [[(observation, action), ...], [(observation, action), ...], ...]

    # The next string flattens train_data list: [(observation, action), (observation, action), ...]

    train_data = [observation for episode in train_data for observation in episode]

    

    # Creating labels array

    labels = np.array([observation[1] for observation in train_data])

    

    # Storing only observations in train_data array

    train_data = np.array([observation[0] for observation in train_data])

    print(f'Total observations: {train_data.shape[0]}' )

    

    return train_data, labels
# Generating first training data

train_data, labels = generate_data(n_games = 2000)
# Weights initializer

init = he_normal(seed = 666)



model = Sequential()



# We are using observations from environment as input data, so input shape of our ANN is (4, )

model.add(Dense(64, input_shape = (4,), activation = 'relu', kernel_initializer = init))

model.add(Dense(128, activation = 'relu', kernel_initializer = init))



# Because our action can be only 0 or 1, I'll use Dense layer with one neuron and sigmoid activation function

model.add(Dense(1, activation = 'sigmoid'))



# Compile model using SGD and binary_crossentropy

model.compile(optimizer = 'sgd', loss = 'binary_crossentropy')
def plot_loss():    

    H = model.history.history

    

    plt.figure(figsize = (15, 5))

    plt.plot(H['loss'], label = 'loss')

    plt.plot(H['val_loss'], label = 'val_loss')

    plt.grid()

    plt.legend()

    plt.show()
# Model training

model.fit(train_data, labels, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 0)

plot_loss()
# env = gym.make('CartPole-v0')

# observation = env.reset()



# for i in range(3):

#     temp_reward = 0

#     while True:

#         env.render()



#         action = model.predict(observation.reshape((-1, 4)))

#         action = int(np.rint(action))       



#         observation, reward, done, _ = env.step(action)



#         temp_reward += reward

        

#         if done:

#             print(temp_reward)

#             observation = env.reset()

#             break



# env.close()
# Generating data using actions, predicted by the model

train_data, labels = generate_data(model = model)
# Train model on new data

model.fit(train_data, labels, epochs = 30, batch_size = 32, validation_split = 0.2, verbose = 0)

plot_loss()