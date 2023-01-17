import numpy as np

import matplotlib

import pandas as pd



from scipy.stats import norm



import matplotlib.pyplot as plt 



from tqdm import tqdm 



import time



%matplotlib inline
S0 = 100      # initial stock price

K = 100       # stike

r = 0.05      # risk-free rate

sigma = 0.15  # volatility

mu = 0.05     # drift

M = 1         # maturity



T = 12        # number of time steps

N_MC = 1000   # number of paths



delta_t = M / T                # time interval

gamma = np.exp(- r * delta_t)  # discount factor
N = 25 # number of discrete states

I = (2 + np.log(np.log(N))) * sigma * np.sqrt(T * delta_t)

p = np.log(S0) + (2 * np.arange(1, N+1) - N - 1) / (N - 1) * I

p = pd.DataFrame(p, index=range(1, N+1))

c = (p.loc[2:,:] + p.shift(1).loc[2:,:]) / 2

c.loc[1,:] = float('-inf')

c.loc[N+1,:] = float('inf')

c = c.sort_index(axis=0)
transition_probability = pd.DataFrame([], index=range(1, N+1), columns=range(1, N+1))

for i in range(1, N+1):

    for j in range(1, N+1):

        a = (c.loc[j+1,0] - p.loc[i,0] - (mu - 0.5*sigma**2)*delta_t)/(sigma*np.sqrt(delta_t))

        b = (c.loc[j,0] - p.loc[i,0] - (mu - 0.5*sigma**2)*delta_t)/(sigma*np.sqrt(delta_t))

        transition_probability.loc[i,j] = norm.cdf(a) - norm.cdf(b)

transition_probability = transition_probability.astype('float')
transition_probability
starttime = time.time()



# state variable

X = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))

X.loc[:,0] = np.log(S0)



# position of each state variable on the grid

point = pd.DataFrame([], index=range(1, N_MC+1), columns=range(T+1))

point.loc[:,0] = (N + 1) / 2



for k in range(1, N_MC+1):

    x = (N + 1) / 2

    for t in range(1, T+1):

        next_x = np.random.choice(np.arange(1,N+1),replace=True,p=transition_probability.loc[x,:])

        x = next_x

        point.loc[k,t] = x

        X.loc[k,t] = p.loc[x,0]



point = point.astype('int')



endtime = time.time()

print('\nTime Cost:', endtime - starttime, 'seconds')
# plot 5 paths

plt.plot(X.T.iloc[:,[100,200,300,400,500]])

plt.xlabel('Time Steps')

plt.ylabel('State Variable Value')

plt.twinx()

plt.plot(point.T.iloc[:,[100,200,300,400,500]])

plt.xlabel('Time Steps')

plt.ylabel('Position on Grid')

plt.title('State Variable Sample Paths')

plt.show()
# stock price

S = X.apply(lambda x: np.exp((x + (mu - 1/2 * sigma**2) * np.arange(T+1) * delta_t).astype('float')), axis=1)

delta_S = S.loc[:, 1:T].values - np.exp(r * delta_t) * S.loc[:,0:T-1]

delta_S_hat = delta_S.apply(lambda x: x - np.mean(x), axis=0)



# plot 5 paths

plt.plot(S.T.iloc[:,[100,200,300,400,500]])

plt.xlabel('Time Steps')

plt.title('Stock Price')

plt.show()
BLOCK_SIZE = 1000

NUM_BLOCKS = 10

NUM_S      = 12  #number of discrete values of S

NUM_TIME_STEPS = 10

dt         = 1 # time step

sigma      = 0.1 # volatility

nu         = 1 # market friction parameter

S0         = 1 # initial stock price

lmbda      = 0.01 # risk aversion parameter
EPSILON = 0.1# Probability for exploration



ALPHA = 0.5# Step size



GAMMA = 1 # Discount factor for Q-Learning and Sarsa
ACTIONS = [0, 1, 2, 3]
START = [NUM_BLOCKS - 1, S0, 0]
def step(state, action):

    X, S, t = state

    

    # You can't sell more stock than you have

    if action > X: 

        action = X

    

    # Calculate the number of stocks remaining

    X_next = X - action

    

    # Calculate the resulting price movement

    S_next = S*np.exp(1 - nu*action) + sigma*S*np.sqrt(dt)*np.random.randn() 

    # Quantise S_next to an allowed value

    S_next = np.clip(np.ceil(S_next), 0, NUM_S-1)

    

    next_state = [X_next, np.int(S_next), t+dt]

    

    # Calculate the reward earned from the sale

    mu = (np.exp(1 - nu * action) - 1) / dt

    var = S_next**2*np.exp(2*mu*dt + sigma**2*dt)*(np.exp(sigma**2*dt)-1)

    

    reward = BLOCK_SIZE * action*S - lmbda*BLOCK_SIZE * X_next**2*var

    

    return next_state, reward



# Check some state, action pairs and the associated reward

print(step(START, 0))

print(step([1,30,2], 0))
# Choose an action based on epsilon greedy algorithm

def choose_action(state, q_value, eps=EPSILON):

    # With probability `eps', simply choose a random action - 'Exploration'

    if np.random.binomial(1, eps) == 1:

        action = np.random.choice(ACTIONS)

    # Otherwise, choose from the actions with the highest

    # q-value for the given state - 'Exploitation'

    else:

        values_ = q_value[state[0], state[1], state[2], :]

        action = np.random.choice(

            [action_ for action_, value_ in enumerate(values_) 

                                 if value_ == np.max(values_)])

    # You cannot sell more stocks than you have

    if action > state[0]:

        action = state[0]

    

    return action
q_value = np.zeros((NUM_BLOCKS, NUM_S, NUM_TIME_STEPS, len(ACTIONS)))

[choose_action(START, q_value, EPSILON) for i in range(20)]
def sarsa(q_value, expected=False, step_size=ALPHA, eps=EPSILON):

    state = START

    action = choose_action(state, q_value,eps)

    rewards = 0.0

    while (state[2] < (NUM_TIME_STEPS-1)*dt) and (state[0] > 0):

        next_state, reward = step(state, action)

        next_action = choose_action(next_state, q_value, eps)

        rewards += reward

        if not expected:

            target = q_value[next_state[0], next_state[1], next_state[2], next_action]

        else:

            # Calculate the expected value of new state

            target = 0.0

            q_next = q_value[next_state[0], next_state[1], next_state[2], :]

            best_actions = np.argwhere(q_next == np.max(q_next))

            for action_ in ACTIONS:

                if action_ in best_actions:

                    target += ((1.0 - eps) / len(best_actions) 

                               + eps / len(ACTIONS)) * q_value[next_state[0], next_state[1], next_state[2], action_]

                else:

                    target += eps / len(ACTIONS) * q_value[next_state[0], next_state[1], next_state[2], action_]

        target *= GAMMA

        q_value[state[0], state[1], state[2], action] += step_size * (

                reward + target - q_value[state[0], state[1], state[2], action])

        state = next_state

        action = next_action

    return rewards
def q_learning(q_value, step_size=ALPHA, eps=EPSILON):

    state = START

    rewards = 0.0

    

    while (state[2] < (NUM_TIME_STEPS-1)*dt) and (state[0] > 0):

        action = choose_action(state, q_value, eps)

        next_state, reward = step(state, action)

        rewards += reward

        # Q-Learning update

        Qhere = q_value[state[0], state[1], state[2], action]

        

        bestQnext = max(q_value[next_state[0], next_state[1], next_state[2], :])

        

        nextQhere = Qhere + step_size*(reward + GAMMA*bestQnext - Qhere)

        

        q_value[state[0], state[1], state[2], action] += step_size * (reward 

            + GAMMA * np.max(q_value[next_state[0], next_state[1], next_state[2], :]) 

            - q_value[state[0], state[1], state[2], action])

        

        if nextQhere !=  q_value[state[0], state[1], state[2], action]:

            print()

        

        state = next_state

    return rewards
def figure_9_4(episodes=1000, runs=100):

    EPOCH=25

    

    # Initialise the rewards arrays

    rewards_sarsa = np.zeros(episodes)

    rewards_q_learning = np.zeros(episodes)

    

    for r in tqdm(range(runs)):

        # Initialise the state-action arrays 

        q_sarsa = np.zeros((NUM_BLOCKS, NUM_S, NUM_TIME_STEPS, len(ACTIONS)))

        q_q_learning = np.copy(q_sarsa)

        # Update the rewards and action value arrays for each episode

        for i in range(0, episodes):

            # The value of epsilon is decremented exponentially

            # after every EPOCH episodes

            eps = EPSILON*((1-EPSILON)**(i//EPOCH))

            rewards_sarsa[i] += sarsa(q_sarsa, eps=eps)

            rewards_q_learning[i] += q_learning(q_q_learning, eps=eps)

    

    # Averaging over independent runs

    rewards_sarsa /= runs

    rewards_q_learning /= runs

    

    # Draw reward curves

    plt.plot(rewards_sarsa, label='Sarsa')

    plt.plot(rewards_q_learning, label='Q-Learning')

    plt.xlabel('Episodes')

    plt.ylabel('Sum of rewards during episode')

    plt.legend()



    return q_sarsa, q_q_learning
q_sarsa, q_q_learning = figure_9_4(1200, 50)
def print_optimal_policy(q_value):

    optimal_policy = np.zeros((NUM_BLOCKS, NUM_S, NUM_TIME_STEPS))

    for i in range(0, NUM_BLOCKS):

        for j in range(0, NUM_S):

            for k in range(0, NUM_TIME_STEPS):

                optimal_policy[i,j,k] = np.argmax(q_value[i, j, k, :])

    for k in range(0, NUM_TIME_STEPS):

      print("========= time step " + str(k) + "======") 

      print(" price: 1,2,3,4,5,6,7,8,9,10,11,12")

      for i in range(0, NUM_BLOCKS):

        str_="inventory " + str(i) + ":"    

        for j in range(0, NUM_S): 

            str_+=str(np.int(optimal_policy[i,j,k])) + ','

        print(str_)
print('Sarsa Optimal Policy:')

print_optimal_policy(q_sarsa)

print('Q-Learning Optimal Policy:')

print_optimal_policy(q_q_learning)