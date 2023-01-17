# Base

import numpy as np

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm



# Kaggle

from kaggle_environments import make

from kaggle_environments.envs.connectx.connectx import is_win,specification



# Deep learning

import gym

import torch

import torch.nn as nn

import torch.optim as optim

from torch.nn.modules.transformer import TransformerEncoderLayer

from torch.nn.modules.activation import MultiheadAttention



from torch.nn import Module

from torch.nn.modules.dropout import Dropout

from torch.nn.modules.linear import Linear

from torch.nn.modules.normalization import LayerNorm



from torch.nn import functional as F



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def board_array(observation, env, board3d=False):

    board = np.array(observation.board).reshape((env.configuration['rows'], env.configuration['columns']))

    if board3d:

        b0 = board==0

        b1 = board==env.mark

        b2 = board==1+env.mark%2

        board = np.concatenate([b0,b1,b2], axis=0).reshape((board.shape[0], board.shape[1], 3))

    return board



class PositionalEncoding(nn.Module):

    # Altered function from jadore801120

    def __init__(self, d_hid, n_position=200, n_dim=1, f=100, concat=False, device=None):

        super(PositionalEncoding, self).__init__()

        self.device = device

        self.concat = concat

        self.n_dim = n_dim

        self.f = f

        # Not a parameter

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid, n_dim))



    def _get_sinusoid_encoding_table(self, n_position, d_hid, n_dim):

        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy



        def get_position_angle_vec(position):

            # Altered formula with a product by Pi

            return [position * np.pi / np.power(self.f, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]



        # Only use the cosinus

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

        

        # Commented version that corresponds to the appropriate positional embedding (not useful here)

        #sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i

        #sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        

        # Only keep the cosin version

        sinusoid_table = np.cos(sinusoid_table[:, 1::2])

        

        # Combine the sinusoid_table for both rows and cols (choose the appropriate way to combine them)

        if n_dim==2:

            a = sinusoid_table.reshape((n_position, 1,-1))

            b = sinusoid_table.reshape((1, n_position,-1))

            sinusoid_table = a*b



        pos_table = torch.FloatTensor(sinusoid_table).unsqueeze(0)

        if self.device is not None:

            pos_table = pos_table.to(device)

        return pos_table



    def forward(self, x):

        unsqueezed = False

        if x.ndim-self.n_dim==1:

            x = x.unsqueeze(0)

            unsqueezed = True

        if x.ndim==3:

            pos = self.pos_table[:, :x.size(1)].clone().detach()

        elif x.ndim==4:

            pos = self.pos_table[:, :x.size(1), :x.size(2)].clone().detach()

        dims = [1]*x.ndim

        dims[0] = x.size(0)

        

        if self.concat:

            x = torch.cat([x, pos.repeat(dims)], x.ndim-1)

        else:

            x + pos.repeat(dims)

            

        if unsqueezed:

            x = x.squeeze(0)

        return x

    

class ConnectX(gym.Env):

    def __init__(self, switch_prob=0.5, nrows=6,ncols=7,inarow=4):

        configuration = {'timeout': 5, 'columns':ncols, 'rows':nrows, 'inarow':inarow, 'steps': 1000}

        self.env = make('connectx', debug=True, configuration=configuration)

        self.pair = [None, 'negamax']

        self.mark = 1

        self.trainer = self.env.train(self.pair)

        self.switch_prob = switch_prob



        # Define required gym fields (examples):

        self.configuration = self.env.configuration

        self.action_space = gym.spaces.Discrete(self.configuration.columns)

        self.observation_space = gym.spaces.Discrete(self.configuration.columns * self.configuration.rows)



    def switch_trainer(self):

        self.pair = self.pair[::-1]

        self.trainer = self.env.train(self.pair)

        self.mark = 1+self.mark%2



    def step(self, action):

        return self.trainer.step(action)

    

    def reset(self):

        #if np.random.random() < self.switch_prob:

            #self.switch_trainer()

        return self.trainer.reset()

    

    def render(self, **kwargs):

        return self.env.render(**kwargs)

    

class DeepModel(nn.Module):

    def __init__(self, n_rows, n_cols, n_heads, d_model, b=5, dropout=0.1, device=None):

        super(DeepModel, self).__init__()

        self.n_rows = n_rows

        self.n_cols = n_cols

        self.dropout = nn.Dropout(dropout)

        self.norm1 = LayerNorm(d_model)

        self.transformer1 = TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=8)

        self.linear1 = nn.Linear(d_model, 4)

        self.transformer2 = TransformerEncoderLayer(d_model=4, nhead=2, dim_feedforward=4)

        self.linear2 = nn.Linear(4, 4)

        self.linear3 = nn.Linear(5, 3)

        self.linear4 = nn.Linear(3, 1)



    def forward(self, x):

        # Set the upper and lower bounds a bit beyond possible produced values

        vmax = 25

        vmin = -22

        

        # Collect useful shapes

        n_rows = x.size(0)

        n_cols = x.size(1)

        batch_size = x.size(2)

        

        # Get a vector of the number of available slots per columns

        cnt_left = x[:,:,:,0].sum(dim=0).unsqueeze(2).unsqueeze(0).type(torch.float)

        cnt_left = torch.cat([cnt_left]*n_rows)

        

        # flatten and add the batch dimension

        x = x.view((n_rows*n_cols, batch_size, -1))

        

        # Apply the attention layers followed by a dense one

        x = self.transformer1(x)

        x = F.relu(self.linear1(x))

        x = self.transformer2(x)

        x = F.relu(self.linear2(x))

        

        # Reshape into the game board and batch size

        x = x.view((n_rows, n_cols,batch_size,-1))

        

        

        #x = x.sum(dim=0)

        #print(x.shape, cnt_left.shape)

        # Add the number of tokens left on the column to play

        x = torch.cat([x,cnt_left],dim=3)

        

        # Final dense layer

        x = F.relu(self.linear3(x))

        x = self.linear4(x).squeeze(3)

        

        # Softmax on the row dim and sum over it squared to keep one value per column

        x = torch.softmax(x, 0)

        x = torch.sum(x**2, 0)

        

        # Rescale the output in a range slightly beyond the actual range 

        # of possible values to help the model converge

        x = torch.sigmoid(x) * (vmax-vmin) + vmin

        

        return x





class DQN:

    def __init__(self, num_actions, n_rows, n_cols, n_heads, d_model, gamma, 

                 max_experiences, min_experiences, batch_size, lr, device=None):

        self.num_actions = num_actions

        self.batch_size = batch_size

        self.n_rows = n_rows

        self.n_cols = n_cols

        self.n_heads = n_heads

        self.d_model = d_model

        self.gamma = gamma

        self.model = DeepModel(n_rows, n_cols, n_heads, d_model, device=device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, )

        self.criterion = nn.MSELoss()

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []} # The buffer

        self.max_experiences = max_experiences

        self.min_experiences = min_experiences

        self.device = device

        self.positioner = PositionalEncoding(d_hid=6,n_position=20,n_dim=2,f=100,concat=True,device=device)        

        

    def predict(self, inputs):

        return self.model(inputs)



    def train(self, TargetNet, env):

        if len(self.experience['s']) < self.min_experiences:

            # Only start the training process when we have enough experiences in the buffer

            return 0



        # Randomly select n experience in the buffer, n is batch-size

        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)

        states = torch.cat([self.preprocess(self.experience['s'][i], env) for i in ids], dim=2)

        actions = torch.LongTensor([self.experience['a'][i] for i in ids]).to(self.device)

        rewards = torch.FloatTensor([self.experience['r'][i] for i in ids]).to(self.device)



        # Prepare labels for training process

        states_next = torch.cat([self.preprocess(self.experience['s2'][i], env) for i in ids], dim=2)

        dones = torch.FloatTensor([self.experience['done'][i] for i in ids]).to(self.device).type(torch.bool)



        # Encode actions

        actions = actions.unsqueeze(1)

        actions_one_hot = torch.FloatTensor(self.batch_size, self.num_actions).zero_().to(self.device)

        actions_one_hot = actions_one_hot.scatter_(1, actions, 1).T

        

        # Get action values for known past rewards

        selected_action_values = torch.sum(self.predict(states) * actions_one_hot, dim=0)

        

        # Compute the actual values obtained combined with the next expected state-value

        value_next = TargetNet.predict(states_next).max(dim=0).values

        actual_values = torch.where(dones, rewards, rewards+self.gamma*value_next).to(device)



        # Update weights

        self.optimizer.zero_grad()

        loss = self.criterion(selected_action_values, actual_values)

        loss.backward()

        self.optimizer.step()

        return loss.item()



    # Get an action by using epsilon-greedy

    def get_action(self, state, env, epsilon):

        if np.random.random() < epsilon:

            return int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0]))

        else:

            prediction = self.predict(self.preprocess(state, env)).detach().cpu().numpy()

            for i in range(self.num_actions):

                if state.board[i] != 0:

                    prediction[i] = -1e7

            return int(np.argmax(prediction))



    # Method used to manage the buffer

    def add_experience(self, exp):

        if len(self.experience['s']) >= self.max_experiences:

            for key in self.experience.keys():

                self.experience[key].pop(0)

        for key, value in exp.items():

            self.experience[key].append(value)



    def copy_weights(self, TrainNet):

        self.model.load_state_dict(TrainNet.state_dict())



    def save_weights(self, path):

        torch.save(self.model.state_dict(), path)



    def load_weights(self, path):

        self.model.load_state_dict(torch.load(path))

    

    # Each state will consist of the board and the mark

    # in the observations

    def preprocess(self, state, env):

        # Convert the observed state into a 3D boolean tensor

        board = board_array(state, env, True)

        board = torch.FloatTensor(board)

        if self.device is not None:

            board = board.to(self.device)

            

        # Add positioning

        board = self.positioner(board)

        

        # add the batch dimension

        board = board.view((self.n_rows, self.n_cols, 1, -1))

        

        return board

    

def play_game(env, TrainNet, TargetNet, epsilon, copy_step):

    rewards = 0

    losses = list()

    iter = 0

    done = False

    if np.random.rand()>.5:

        pass

        #env.switch_trainer()

    observations = env.reset()

    while not done:

        # Using epsilon-greedy to get an action

        action = TrainNet.get_action(observations, env, epsilon)



        # Caching the information of current state

        prev_observations = observations



        # Take action

        observations, reward, done, _ = env.step(action)



        # Apply new rules

        if done:

            if reward == 1: # Won

                reward = 20

            elif reward == 0: # Lost

                reward = -20

            else: # Draw

                reward = 10

        else:

            # Try to promote the agent to "struggle" when playing against negamax agent

            # as Magolor's (@magolor) idea

            reward = 0.5



        rewards += reward



        # Adding experience into buffer

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}

        TrainNet.add_experience(exp)



        # Train the training model by using experiences in buffer and the target model

        losses.append(TrainNet.train(TargetNet, env))

        iter += 1

        if iter % copy_step == 0:

            # Update the weights of the target model when reaching enough "copy step"

            TargetNet.copy_weights(TrainNet)

    return rewards, np.mean(losses)



def rephase_game(env, TrainNet, TargetNet, lr, n_rows, n_cols, inarow):

    """Keep weights of the networks but change everything else that 

    needs to be updated to match the new environment"""

    # Update the env

    env = ConnectX(ncols=n_cols, nrows=n_rows, inarow=inarow)

    

    # Reinitialize useful parameters on the TrainNet

    TrainNet.n_rows = n_rows

    TrainNet.n_cols = n_cols

    TrainNet.num_actions = n_cols

    TrainNet.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    TrainNet.optimizer = optim.Adam(TrainNet.model.parameters(), lr=lr, )

    

    # Reinitialize useful parameters on the TargetNet

    TargetNet.n_rows = n_rows

    TargetNet.n_cols = n_cols

    TargetNet.num_actions = n_cols

    TargetNet.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

    TargetNet.optimizer = optim.Adam(TargetNet.model.parameters(), lr=lr, )

    

    return env, TrainNet, TargetNet
n_possible_values = 3 # should never change

n_pos_dim = 3

max_length = 20



n_rows = 3

n_cols = 3

inarow = 3

n_heads = 6

d_model = n_possible_values + n_pos_dim



gamma = 0.99

copy_step = 25

hidden_units = [100, 200, 200, 100]

max_experiences = 10000

min_experiences = 100

batch_size = 32

lr = 3e-2

epsilon = 0.5

decay = 0.9999

min_epsilon = 0.01

episodes = 2000



precision = 7



env = ConnectX(ncols=n_cols, nrows=n_rows, inarow=inarow)

num_actions = env.action_space.n
all_total_rewards = list()

all_avg_rewards = list()

all_avg_losses = list()

all_epsilons = list()
# Initialize models

TrainNet = DQN(num_actions, n_rows, n_cols, n_heads, d_model, gamma, max_experiences, min_experiences, batch_size, lr, device=device)

TargetNet = DQN(num_actions, n_rows, n_cols, n_heads, d_model, gamma, max_experiences, min_experiences, batch_size, lr, device=device)
pbar = tqdm(range(episodes))



env, TrainNet, TargetNet = rephase_game(env, TrainNet,TargetNet,lr,n_rows, n_cols, inarow)

for n in pbar:

    epsilon = max(min_epsilon, epsilon * decay)

    total_reward, avg_loss = play_game(env, TrainNet, TargetNet, epsilon, copy_step)

    all_total_rewards.append(total_reward)

    avg_reward = np.mean(all_total_rewards[-100:])

    all_avg_rewards.append(avg_reward)

    all_avg_losses.append(avg_loss)

    avg_loss = np.mean(all_avg_losses[-100:])

    all_epsilons.append(epsilon)



    pbar.set_postfix({

        'episode reward': total_reward,

        'avg (100 last) reward': avg_reward,

        'avg (100 last) losses': avg_loss,

        'epsilon': epsilon

    })
# Analyze how the loss evolves over time

plt.plot(all_avg_losses)

plt.show()
# Analyze how the revard evolves over time

plt.plot(all_avg_rewards)

plt.show()
# Have a look at the model playing against negamax

env.switch_trainer()

print(env.mark)

play_game(env, TrainNet, TargetNet, 0.3, copy_step)

env.render(mode="ipython", width=500, height=450, )