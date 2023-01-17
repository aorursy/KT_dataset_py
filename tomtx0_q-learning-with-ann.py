'''
.--------------------.
| 🐹 | ❄️ | ❄️ | ❄️ | 
|- - - - -  - - - - -|
| ❄️ | 🕳 | ❄️ | 🕳 |
|- - - - -  - - - - -|
| ❄️ | ❄️ | ❄️ | 🕳 |
|- - - - -  - - - - -|
| 🕳 | ❄️ | ❄️ | 🥜 |
.--------------------.
'''
'''
  input        weights         output
 (states)     (Q-values)       (Q-values for actions in that particular state)  
  
             U   D   L   R
           .---------------.
0    o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
1    o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
2    o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
3    o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
4    o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
5    o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
6    o-   -| Q | Q | Q | Q |-   -o U
           | - - - - - - - |
7    o-   -| Q | Q | Q | Q |-   -o D
           | - - - - - - - |
8    o-   -| Q | Q | Q | Q |-   -o L
           | - - - - - - - |
9    o-   -| Q | Q | Q | Q |-   -o R
           | - - - - - - - |
10   o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
11   o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
12   o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
13   o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
14   o-   -| Q | Q | Q | Q |- 
           | - - - - - - - |
15   o-   -| Q | Q | Q | Q |- 
           .---------------.
'''
import gym
import mxnet as mx
import matplotlib.pyplot as plt
import random
from time import time
EPISODES = 2000
STEPS = 100
BANDIT_ARMS = 4
BANDIT_TRAINING = True
EPSILON = 1 # setting the agent for exploration only -> try to learn the environment
LEARNING_RATE = 0.2 # the learning rate for our optimization algorithm
GAMMA = 0.95
env = gym.make('FrozenLake-v0')
states = env.observation_space.n
actions = env.action_space.n
# Input Layer
# -> create neurons for the input layer representing states of environemnt
input_layer = mx.sym.Variable(name='ann_states', shape=(1, states), dtype='float32')

# Weights Matrix
# -> create a matrix with weights representing Q-values in every state of environment
weights = mx.sym.Variable(name='ann_Q_values', shape=(states, actions), dtype='float32')

# Output Layer
# -> create neurons for the output layer representing Q-values in a particular state
output_layer = mx.sym.dot(name='ann_Q_values_state', lhs=input_layer, rhs=weights)
def train_ann(output_weights, label_weights, learning_rate):
    # compute gradients of the loss function by differentiating a graph of NDArray operations with the chain rule
    # -> define parameters of loss function for the following automatic differentiation
    loss_params = mx.nd.broadcast_sub(lhs=label_weights, rhs=output_weights)
    # -> attach gradient buffers to variables that require gradient
    loss_params.attach_grad()
    # -> wrap operations that need to be differentiated into a graph
    with mx.autograd.record():
        # add an operation for the loss function (L2)
        diff_opp = mx.nd.sum(data=mx.nd.square(data=loss_params))
    # -> calculate the gradients of wrapped operations with respect to their parameters
    diff_opp.backward()
    # -> get the gradients of the loss function
    loss_gradients = loss_params.grad
    
    # update the ANN label weights via the Batch Gradient Descent (BGD) optimization algorithm
    updated_weights = label_weights - learning_rate * loss_gradients
    return updated_weights
def run_epsilon_greedy_bandit(arm_rewards, arms, training, epsilon, episode):
    # check if bandit is used for training
    if training:        
        # -> start with the full exploration and slowly reduce it as this algorithm learns
        epsilon = mx.nd.exp(data=mx.nd.array([-0.01 * episode]))

    # flip the coin (randomly select the number between 0 to 1)
    random_number = random.random()

    # select the arm
    if random_number > epsilon:
        # Exploit
        # -> find arms with highest rewards
        max_reward_arms = [i for i, e in enumerate(arm_rewards) if e == max(arm_rewards)]
        # -> select the best arm
        if len(max_reward_arms) == 1:
            # get the best arm
            arm = max_reward_arms[0]
        else:
            # randomly choose an arm from the best arms
            arm = random.choice(max_reward_arms)
    else:
        # Explore
        # -> randomly choose an arm
        arm = random.randrange(arms)
        
    return arm
# log the training start
training_start = time()

# store the training progress of this algorithm for each episode
episode_rewards = []
episode_steps = []

# prepare the model & initialize all trainable variables before the training starts
# -> prepare a matrix for slicing states in the input layer 
input_layer_init = mx.nd.one_hot(mx.nd.arange(16), 16)
# -> manually initilize weights (Q-values) with random values close to zero
weights_init = mx.nd.random_uniform(low=0, high=0.01, shape=(states, actions), dtype='float32')
ex = weights.eval(ctx = mx.cpu(), ann_Q_values=weights_init)
Q = ex[0]

# solve the environment over certain amount of episodes
for episode in range(EPISODES):
    # reset the environment, rewards, and steps for the new episode
    s = env.reset()
    episode_reward = 0
    step = 0

    # find the solution over certain amount of attempts (steps in each episode)
    while step < STEPS:
        # get Q-values for the current state by feeding it forward through the ANN
        ex = output_layer.eval(ctx = mx.cpu(), ann_states=input_layer_init[s:s+1], ann_Q_values=Q)
        Q_current = ex[0][0]

        # select the action in the current state by running the multiarmed bandit
        a = run_epsilon_greedy_bandit(Q_current, BANDIT_ARMS, BANDIT_TRAINING, EPSILON, episode)

        # enter the environment and get the experience from it by performing there an action
        # -> get observation (new state), reward, done (success/failure), and information
        observation, reward, done, info = env.step(a)

        # get Q-values for the observed state by feeding it through the ANN 
        ex = output_layer.eval(ctx = mx.cpu(), ann_states=input_layer_init[observation:observation+1], ann_Q_values=Q)
        Q_observed = ex[0][0]

        # estimate the Q-value for the current state and action
        # -> calculate this Q-value with the update rule using the experience from the environment
        Q_target = Q_current.copy()
        Q_target[a] = reward + GAMMA * mx.nd.max(Q_observed)

        # train the ANN to minimize the estimation error by optimizing the Q-values
        Q_optimized = train_ann(Q_target, Q_current, LEARNING_RATE)

        # manually update the weights matrix with the trained Q-values for this state
        Q[s] = Q_optimized

        # add the reward to others during this episode
        episode_reward += reward

        # change the state to the observed state for the next iteration
        s = observation

        # check if the environment has been exited
        if done:
            # -> store the collected rewards & number of steps in this episode     
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            # -> quit the episode
            break

        # continue looping
        step += 1
        
# log the training end
training_end = time()
print("trained Q-values", Q)
# show the success rate for solving the environment & elapsed training time
success_rate = round((sum(episode_rewards) / EPISODES) * 100, 2)
elapsed_training_time = int(training_end - training_start)
print("\nThis environment has been solved", str(success_rate), "% of times over",  str(EPISODES), "episodes within", str(elapsed_training_time), "seconds!")

# plot the rewards and number of steps over all training episodes
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(episode_rewards, '-g', label = 'reward')
ax1.set_yticks([0,1])
ax2 = ax1.twinx()
ax2.plot(episode_steps, '+r', label = 'step')
ax1.set_xlabel("episode")
ax1.set_ylabel("reward")
ax2.set_ylabel("step")
ax1.legend(loc=2)
ax2.legend(loc=1)
plt.title("Training Stats")
plt.show()
GAME_EPISODES = 10
BANDIT_TRAINING = False
EPSILON = 0 # setting the agent for pure exploitation -> use only learned Q values from training
for episode in range(GAME_EPISODES):
    s = env.reset()
    step = 0
    while step < STEPS:
        # take the action that have the maximum expected future reward
        a = run_epsilon_greedy_bandit(Q[s, :], BANDIT_ARMS, BANDIT_TRAINING, EPSILON, episode)
        
        # enter the environment and get the experience from it
        observation, reward, done, info = env.step(a)

        # change the state to observed state for the next iteration
        s = observation

        # check if the environment has been exited
        if done:
            # print results of each episode
            print("\n-----------------------------")
            print(step, "steps in episode", episode)
            print("The last action & state was:")
            env.render()
            print("-----------------------------")
            # quit the episode
            break

        # continue looping
        step += 1