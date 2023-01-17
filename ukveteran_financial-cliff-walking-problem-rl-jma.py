import numpy as np



import matplotlib.pyplot as plt 



from tqdm import tqdm



%matplotlib inline
WORLD_HEIGHT = 4 # The number of states

WORLD_WIDTH = 12 # The number of time steps
EPSILON = 0.1 # Probability for exploration



ALPHA = 0.001 # Step size



GAMMA = 1 # Discount factor for Q-Learning, Sarsa and Expected Sarsa
ACTION_UP = 1

ACTION_DOWN = 2

ACTION_ZERO = 0

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_ZERO]
START = [0, 0]

GOAL = [0, WORLD_WIDTH-1]
def step(state, action):

    i, j = state



    if action == ACTION_UP:

        next_state = [min(i + 1, WORLD_HEIGHT-1), min(j + 1, WORLD_WIDTH - 1)]

    elif action == ACTION_DOWN:

        next_state = [max(i - 1, 0), min(j + 1, WORLD_WIDTH - 1)]

    elif action == ACTION_ZERO:

        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]

    else:

        raise ValueError("action not recognised")

    

    # The reward is -1 for actions ACTION_UP and ACTION_DOWN. 

    # This is done to keep transactions to a minimum.

    reward = -1

    

    # ACTION_ZERO gets a zero reward since we want to minimize the number of transactions

    if action == ACTION_ZERO:

        reward = 0

    

    # Exceptions are 

    # i) If bankruptcy happens before WORLD_WIDTH time steps

    # ii) No deposit at initial state

    # iii) Redemption at initial state!

    # iv) Any action carried out from a bankrupt state

    if ((action == ACTION_DOWN and i == 1 and 1 <= j < 10) or (

        action == ACTION_ZERO and state == START) or (

        action == ACTION_DOWN and state == START )) or (

        i == 0 and 1 <= j <= 10):    

            reward = -100

        

    # Next exception is when we get to the final time step.

    if (next_state[1] == WORLD_WIDTH - 1): 

        if (next_state[0] == 0): # Action resulted in ending with zero balance in final time step

            reward = 10

        else:

            reward = -10   

        

    return next_state, reward
# Check some state, action pairs and the associated reward

print(step([0, 0], ACTION_UP))

print(step([2, 3], ACTION_DOWN))

print(step([1, 5], ACTION_DOWN))
def choose_action(state, q_value, eps=EPSILON):

    # With probability `eps', simply choose a random action - 'Exploration'

    if np.random.binomial(1, eps) == 1:

        action = np.random.choice(ACTIONS)

    # Otherwise, choose from the actions with the highest

    # q-value for the given state - 'Exploitation'

    else:

        values_ = q_value[state[0], state[1], :]

        action = np.random.choice(

            [action_ for action_, value_ in enumerate(values_) 

                                if value_ == np.max(values_)])

    # From the bankrupt state there is no meaningful action, 

    # so we will assign ACTION_ZERO by convention.

    if state[0] == 0 and state[1] > 0:

        action = ACTION_ZERO

    return action
# If q_value contains only zeroes, the action is always random

q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))

print(choose_action([0, 2], q_value, EPSILON))
q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))

q_value[1, 4, 2] = 1

print('action with highest q for state=[1, 4]:', np.argmax(q_value[1, 4]))

print('choices (eps=0):', *[choose_action([1, 4], q_value, 0) for i in range(10)])

print('choices (eps=1):', *[choose_action([1, 4], q_value, 1) for i in range(10)])



q_value[2, 7, 0] = 1

print('\naction with highest q for state=[2, 7]:', np.argmax(q_value[2, 7]))

print('choices (eps=0):  ', *[choose_action([2, 7], q_value, 0) for i in range(10)])

print('choices (eps=0.5):', *[choose_action([2, 7], q_value, 0.5) for i in range(10)])
def sarsa(q_value, expected=False, step_size=ALPHA, eps=EPSILON):

    state = START

    action = choose_action(state, q_value, eps)

    rewards = 0.0

    

    while (state[1] != WORLD_WIDTH-1) and not(1 <= state[1] <= 10 and state[0]==0):

        next_state, reward = step(state, action)

        next_action = choose_action(next_state, q_value, eps)

        rewards += reward

        if not expected:

            target = q_value[next_state[0], next_state[1], next_action]

        else:

            # Calculate the expected value of the new state

            target = 0.0

            q_next = q_value[next_state[0], next_state[1], :]

            best_actions = np.argwhere(q_next == np.max(q_next))

            for action_ in ACTIONS:

                if action_ in best_actions:

                    target += ((1.0 - eps) / len(best_actions) 

                               + eps / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]

                else:

                    target += eps / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]

        target *= GAMMA

        q_value[state[0], state[1], action] += step_size * (

                reward + target - q_value[state[0], state[1], action])

        state = next_state

        action = next_action

    return rewards
def q_learning(q_value, step_size=ALPHA, eps=EPSILON):

    state = START

    rewards = 0.0

    while state[1] != WORLD_WIDTH-1 and not(1 <= state[1] <= 10 and state[0]==0):

        action = choose_action(state, q_value, eps)

        next_state, reward = step(state, action)

        rewards += reward

        # Q-Learning update

        q_value[state[0], state[1], action] += step_size * (

                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -

                q_value[state[0], state[1], action])

        state = next_state

    return rewards
plt.plot([ EPSILON*((1-EPSILON)**(i//40)) for i in range(2000)])

plt.xlabel('Episode')

plt.ylabel('Epsilon');
# Sarsa converges to the safe path, while Q-Learning converges to the optimal path

def figure_9_4():

    # Number of episodes in each run

    episodes = 1500

    EPOCH = 40



    # Perform 100 independent runs

    runs = 100

    

    # Initialise the rewards arrays

    rewards_sarsa = np.zeros(episodes)

    rewards_q_learning = np.zeros(episodes)

    for r in tqdm(range(runs)):

        # Initialise the action value arrays 

        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))

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

    plt.figure(figsize=(15,7))

    plt.plot(rewards_sarsa, label='Sarsa')

    plt.plot(rewards_q_learning, label='Q-Learning')

    plt.xlabel('Episodes')

    plt.ylabel('Sum of rewards during episode')

    plt.ylim([-100, 20])

    plt.legend()



    return q_sarsa, q_q_learning
q_sarsa, q_q_learning = figure_9_4()
def print_optimal_policy(q_value):

    optimal_policy = []

    for i in range(WORLD_HEIGHT-1, -1, -1):

        optimal_policy.append([])

        for j in range(0, WORLD_WIDTH):

            if [i, j] == GOAL:

                optimal_policy[-1].append('G')

                continue

            bestAction = np.argmax(q_value[i, j, :])

            

            # Action in bankrupt state has been set as Z

            if i == 0 and j > 0 and j < WORLD_WIDTH-1:

                optimal_policy[-1].append('Z')

            elif bestAction == ACTION_UP:

                # When i = WORLDHEIGHT - 1, U and Z are identical, so we will use Z

                if i == WORLD_HEIGHT-1:

                    optimal_policy[-1].append('Z')

                else:

                    optimal_policy[-1].append('U')

            elif bestAction == ACTION_DOWN:

                optimal_policy[-1].append('D')

            elif bestAction == ACTION_ZERO:

                optimal_policy[-1].append('Z')

    

    for row in optimal_policy:

        print(row)
print('Sarsa Optimal Policy:')

print_optimal_policy(q_sarsa)

print('Q-Learning Optimal Policy:')

print_optimal_policy(q_q_learning)