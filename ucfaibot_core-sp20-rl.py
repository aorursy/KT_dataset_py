from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-sp20-rl").exists():

    DATA_DIR /= "ucfai-core-sp20-rl"

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-rl/data

    DATA_DIR = Path("data")
import gym

from gym import wrappers



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import matplotlib.patches as mpatches

from matplotlib.colors import ListedColormap



import io

import base64

from IPython.display import HTML

from tqdm import tqdm, trange
# Making our environment



env = gym.make('MountainCar-v0')

env.seed(1); np.random.seed(1)
print(env.action_space)
print(env.observation_space)

print(env.observation_space.low)

print(env.observation_space.high)
episodes = 1000

steps = 200

env._max_episode_steps = steps



successes = 0

position_history = []

reward_history = []
for i in trange(episodes):

    # Initial state

    done = False

    running_reward = 0

    max_position = -0.4

    state = env.reset()



    # Run the episode

    while not done:

        # Select random action

        action = env.action_space.sample()

        

        # Execute action

        state, reward, done, info = env.step(action)



        # Get car's current position and update furthest position for this episode

        current_position = state[0]

        if current_position > max_position:

          max_position = current_position



        # Track reward

        running_reward += reward



    # Document this episodes total reward and furthest position

    reward_history.append(running_reward)

    position_history.append(max_position)



    # Document success if car reached 0.5 or further

    if max_position >= 0.5:

      successes += 1
print('Successful Episodes: {}'.format(successes))

print('Success Rate: {}%'.format(round((successes/episodes)*100,2)))



plt.figure(1, figsize=[10,5])

plt.subplot(211)



# Calculate and plot moving average of each episodes furthest position

p = pd.Series(position_history)

plt.plot(p, alpha=0.8)

plt.plot(p.rolling(50).mean())

plt.ylabel('Position')

plt.title('Cars Furthest Position')



# Calculate and plot moving average of each episodes total reward

plt.subplot(212)

p = pd.Series(reward_history)

plt.plot(p, alpha=0.8)

plt.plot(p.rolling(50).mean())

plt.xlabel('Episode')

plt.ylabel('Reward')

plt.show()
pos_space = np.linspace(-1.2, 0.6, 20)

vels_space = np.linspace(-0.07, 0.07, 20)
def get_state(observation):

  pos, vel = observation

  pos_bin = np.digitize(pos, pos_space)

  vel_bin = np.digitize(vel, vels_space)



  return (pos_bin, vel_bin)
# Create list of possible states

states = []

for pos in range(21):

  for vel in range(21):

    states.append((pos, vel))



# Initialize Q table

Q = {}

for state in states:

  for action in [0,1,2]:

    Q[state, action] = 0
def max_action(Q, state, actions=[0,1,2]):

  action_choices = np.array([Q[state, a] for a in actions])



  return np.argmax(action_choices)
# Parameters

episodes = 2000

steps = 1000

epsilon = 1

gamma = 0.99

lr = 0.1



env._max_episode_steps = steps



successes = 0

position_history = []

reward_history = []
for i in range(episodes):

    if i % 100 == 0 and i > 0:

          print('episode ', i, 'score ', running_reward, 'epsilon %.3f' % epsilon)



    # Initial state

    done = False

    running_reward = 0

    max_position = -0.4

    obs = env.reset()

    state = get_state(obs)



    # Run the episode

    while not done:

        # Esilon-greedy action selection

        if np.random.random() < epsilon:

            action = env.action_space.sample()

        else:

            action = max_action(Q, state)

        

        # Execute chosen action

        next_obs, reward, done, info = env.step(action)



        # Get car's current position and update furthest position for this episode

        current_position = next_obs[0]

        if current_position > max_position:

          max_position = current_position



        # Track reward

        running_reward += reward



         # Bucketize the state

        next_state = get_state(next_obs)



        # Select the next best action (used in Q-learning algorithm)

        next_action = max_action(Q, next_state)



         # Update our Q policy with what we learned from the previous state

        Q[state, action] = Q[state, action] + lr*(reward + gamma*Q[next_state, next_action] - Q[state, action])



        state = next_state



    # Document this episodes total reward and furthest position

    reward_history.append(running_reward)

    position_history.append(max_position)



    # Document success if car reached 0.5 or further

    if max_position >= 0.5:

      successes += 1

    

    # Decrease epsilon (lower bounded at 0.01 so theres always some chance of a random action)

    epsilon = epsilon - 2/episodes if epsilon > 0.01 else 0.01
print('Successful Episodes: {}'.format(successes))

print('Success Rate: {}%'.format(round((successes/episodes)*100,2)))



plt.figure(1, figsize=[10,5])

plt.subplot(211)



# Calculate and plot moving average of each episodes furthest position

p = pd.Series(position_history)

plt.plot(p, alpha=0.8)

plt.plot(p.rolling(50).mean())

plt.ylabel('Position')

plt.title('Cars Furthest Position')



# Calculate and plot moving average of each episodes total reward

plt.subplot(212)

p = pd.Series(reward_history)

plt.plot(p, alpha=0.8)

plt.plot(p.rolling(50).mean())

plt.xlabel('Episode')

plt.ylabel('Reward')

plt.show()
# List of possible positions

X = np.random.uniform(-1.2, 0.6, 10000)

# List of possible velocities

Y = np.random.uniform(-0.07, 0.07, 10000)



# For each possible state, retreive the most rewarding action and record it

actions = []

for i in range(len(X)):

    state = get_state([X[i], Y[i]])

    action = max_action(Q, state)

    actions.append(action)



actions = pd.Series(actions)

colors = {0:'blue',1:'lime',2:'red'}

colors = actions.apply(lambda x:colors[x])

labels = ['Left','Right','Nothing']
# Visualize the policy



fig = plt.figure(3, figsize=[7,7])

ax = fig.gca()

plt.set_cmap('brg')

surf = ax.scatter(X,Y, c=actions)

ax.set_xlabel('Position')

ax.set_ylabel('Velocity')

ax.set_title('Policy')

recs = []

for i in range(0,3):

     recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))

plt.legend(recs,labels,loc=4,ncol=3)

plt.show()
# With an improved reward function, 1000 episodes with 200 steps/ep should be plenty to learn an effective policy.

episodes = 2000

steps = 1000

epsilon = 1

gamma = 0.99

lr = 0.1



env._max_episode_steps = steps



successes = 0

position_history = []

reward_history = []
# Reset our Q table

for state in states:

  for action in [0,1,2]:

    Q[state, action] = 0
for i in range(episodes):

    # Initial state

    done = False

    running_reward = 0

    max_position = -0.4

    obs = env.reset()

    state = get_state(obs)



    # Run the episode

    while not done:

        # Esilon-greedy action selection

        if np.random.random() < epsilon:

            action = env.action_space.sample()

        else:

            action = max_action(Q, state)

        

        # Execute chosen action

        next_obs, reward, done, info = env.step(action)



        # Get car's current position and update furthest position for this episode

        current_position = next_obs[0]

        if current_position > max_position:

          max_position = current_position



        # Make your adjustments or additions to the reward below

        # YOUR CODE HERE

        raise NotImplementedError()



        # Track reward

        running_reward += reward



         # Bucketize the state

        next_state = get_state(next_obs)



        # Select the next best action (used in Q-learning algorithm)

        next_action = max_action(Q, next_state)



         # Update our Q policy with what we learned from the previous state

        Q[state, action] = Q[state, action] + lr*(reward + gamma*Q[next_state, next_action] - Q[state, action])



        state = next_state

        obs = next_obs



    # Document this episodes total reward and furthest position

    reward_history.append(running_reward)

    position_history.append(max_position)



    # Document success if car reached 0.5 or further

    if max_position >= 0.5:

      successes += 1

    

    # Decrease epsilon (lower bounded at 0.01 so theres always some chance of a random action)

    epsilon = epsilon - 2/episodes if epsilon > 0.01 else 0.01
print('Successful Episodes: {}'.format(successes))

print('Success Rate: {}%'.format(round((successes/episodes)*100,2)))



plt.figure(1, figsize=[10,5])

plt.subplot(211)



# Calculate and plot moving average of each episodes furthest position

p = pd.Series(position_history)

plt.plot(p, alpha=0.8)

plt.plot(p.rolling(50).mean())

plt.ylabel('Position')

plt.title('Cars Furthest Position')



# Calculate and plot moving average of each episodes total reward

plt.subplot(212)

p = pd.Series(reward_history)

plt.plot(p, alpha=0.8)

plt.plot(p.rolling(50).mean())

plt.xlabel('Episode')

plt.ylabel('Reward')

plt.show()
X = np.random.uniform(-1.2, 0.6, 10000)

Y = np.random.uniform(-0.07, 0.07, 10000)



actions = []

for i in range(len(X)):

    state = get_state([X[i], Y[i]])

    action = max_action(Q, state)

    actions.append(action)



actions = pd.Series(actions)

colors = {0:'blue',1:'lime',2:'red'}

colors = actions.apply(lambda x:colors[x])

labels = ['Left','Right','Nothing']
fig = plt.figure(5, figsize=[7,7])

ax = fig.gca()

plt.set_cmap('brg')

surf = ax.scatter(X,Y, c=actions)

ax.set_xlabel('Position')

ax.set_ylabel('Velocity')

ax.set_title('Policy')

recs = []

for i in range(0,3):

     recs.append(mpatches.Rectangle((0,0),1,1,fc=sorted(colors.unique())[i]))

plt.legend(recs,labels,loc=4,ncol=3)

plt.show()