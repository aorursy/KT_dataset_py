# I decided to do the Frozen lake test because I found the most resources on it 
# and I think I understand it the most.

import numpy as np 
import pandas as pd

import os
print(os.listdir("../input"))
import gym
import numpy as np
env = gym.make("FrozenLake-v0")
env.action_space.n
env.observation_space
alpha = 1
gamma = 0.5
q_table = dict([(x, [0, 0, 0, 0]) for x in range(4)])
q_table
def choose_action(observ):
    return np.argmax(q_table[observ])


for i in range(10000):
    observ = env.reset()
    action = choose_action(observ)
    prev_observ = None
    prev_action = Non
    t = 0
    # I chose 100 times because Kaggle is slow
    for t in range(100):
        env.render()
        observ, reward, done, info = env.step(action)
        action = choose_action(observ)
        if not prev_observ is None:
            q_old = q_table[prev_observ][prev_action]
            q_new = q_old
        if done:
                q_new += alpha * (reward - q_old)
        else:
                q_new += alpha * (reward + gamma + q_table[observ][action] - q_old)
                new_table = q_table[prev_observ]
                new_table[prev_action] = q_new
            
                q_table[prev_observ] = new_table
        prev_observ = observ
        prev_action = action
        if done:
            print("Episode {} finished after {} timesteps with r={}.".format(i, t, reward))
            break
            
new_table

q_table
        
        
        