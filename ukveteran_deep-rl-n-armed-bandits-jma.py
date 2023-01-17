def get_best_action(actions):

    best_action = 0

    max_action_value = 0

    for i in range(len(actions)): #A 

        cur_action_value = get_action_value(actions[i]) #B

        if cur_action_value > max_action_value:

            best_action = i

            max_action_value = cur_action_value

    return best_action
import numpy as np

from scipy import stats

import random

import matplotlib.pyplot as plt



n = 10

probs = np.random.rand(n) #A

eps = 0.1
def get_reward(prob, n=10):

    reward = 0;

    for i in range(n):

        if random.random() < prob:

            reward += 1

    return reward
reward_test = [get_reward(0.7) for _ in range(2000)]
np.mean(reward_test)
sum = 0

x = [4,5,6,7]

for j in range(len(x)): 

    sum = sum + x[j]

sum
plt.figure(figsize=(9,5))

plt.xlabel("Reward",fontsize=22)

plt.ylabel("# Observations",fontsize=22)

plt.hist(reward_test,bins=9)
# 10 actions x 2 columns

# Columns: Count #, Avg Reward

record = np.zeros((n,2))
def get_best_arm(record):

    arm_index = np.argmax(record[:,1],axis=0)

    return arm_index
def update_record(record,action,r):

    new_r = (record[action,0] * record[action,1] + r) / (record[action,0] + 1)

    record[action,0] += 1

    record[action,1] = new_r

    return record
fig,ax = plt.subplots(1,1)

ax.set_xlabel("Plays")

ax.set_ylabel("Avg Reward")

fig.set_size_inches(9,5)

rewards = [0]

for i in range(500):

    if random.random() > 0.2:

        choice = get_best_arm(record)

    else:

        choice = np.random.randint(10)

    r = get_reward(probs[choice])

    record = update_record(record,choice,r)

    mean_reward = ((i+1) * rewards[-1] + r)/(i+2)

    rewards.append(mean_reward)

ax.scatter(np.arange(len(rewards)),rewards)
def softmax(av, tau=1.12):

    softm = ( np.exp(av / tau) / np.sum( np.exp(av / tau) ) )

    return softm
probs = np.random.rand(n)

record = np.zeros((n,2))
fig,ax = plt.subplots(1,1)

ax.set_xlabel("Plays")

ax.set_ylabel("Avg Reward")

fig.set_size_inches(9,5)

rewards = [0]

for i in range(500):

    p = softmax(record[:,1],tau=0.7)

    choice = np.random.choice(np.arange(n),p=p)

    r = get_reward(probs[choice])

    record = update_record(record,choice,r)

    mean_reward = ((i+1) * rewards[-1] + r)/(i+2)

    rewards.append(mean_reward)

ax.scatter(np.arange(len(rewards)),rewards)
class ContextBandit:

    def __init__(self, arms=10):

        self.arms = arms

        self.init_distribution(arms)

        self.update_state()

        

    def init_distribution(self, arms):

        # Num states = Num Arms to keep things simple

        self.bandit_matrix = np.random.rand(arms,arms)

        #each row represents a state, each column an arm

        

    def reward(self, prob):

        reward = 0

        for i in range(self.arms):

            if random.random() < prob:

                reward += 1

        return reward

        

    def get_state(self):

        return self.state

    

    def update_state(self):

        self.state = np.random.randint(0,self.arms)

        

    def get_reward(self,arm):

        return self.reward(self.bandit_matrix[self.get_state()][arm])

        

    def choose_arm(self, arm):

        reward = self.get_reward(arm)

        self.update_state()

        return reward
import numpy as np

import torch



arms = 10

N, D_in, H, D_out = 1, arms, 100, arms
env = ContextBandit(arms=10)

state = env.get_state()

reward = env.choose_arm(1)

print(state)
model = torch.nn.Sequential(

    torch.nn.Linear(D_in, H),

    torch.nn.ReLU(),

    torch.nn.Linear(H, D_out),

    torch.nn.ReLU(),

)
loss_fn = torch.nn.MSELoss()
env = ContextBandit(arms)
def one_hot(N, pos, val=1):

    one_hot_vec = np.zeros(N)

    one_hot_vec[pos] = val

    return one_hot_vec
def running_mean(x,N=50):

    c = x.shape[0] - N

    y = np.zeros(c)

    conv = np.ones(N)

    for i in range(c):

        y[i] = (x[i:i+N] @ conv)/N

    return y
def train(env, epochs=5000, learning_rate=1e-2):

    cur_state = torch.Tensor(one_hot(arms,env.get_state())) #A

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rewards = []

    for i in range(epochs):

        y_pred = model(cur_state) #B

        av_softmax = softmax(y_pred.data.numpy(), tau=2.0) #C

        av_softmax /= av_softmax.sum() #D

        choice = np.random.choice(arms, p=av_softmax) #E

        cur_reward = env.choose_arm(choice) #F

        one_hot_reward = y_pred.data.numpy().copy() #G

        one_hot_reward[choice] = cur_reward #H

        reward = torch.Tensor(one_hot_reward)

        rewards.append(cur_reward)

        loss = loss_fn(y_pred, reward)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        cur_state = torch.Tensor(one_hot(arms,env.get_state())) #I

    return np.array(rewards)
rewards = train(env)
plt.plot(running_mean(rewards,N=500))