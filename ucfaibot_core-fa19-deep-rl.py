from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-fa19-deep-rl").exists():

    DATA_DIR /= "ucfai-core-fa19-deep-rl"

elif DATA_DIR.exists():

    # no-op to keep the proper data path for Kaggle

    pass

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-deep-rl/data

    DATA_DIR = Path("data")
import gym

import numpy as np 

import torch

from torch import nn

from torch.autograd import Variable

from torch import optim

from torch.nn import functional as F

import matplotlib.pyplot as plt
def one_hot(ids, nb_digits):

    """

    ids: (list, ndarray) shape:[batch_size]

    """

    if not isinstance(ids, (list, np.ndarray)):

        raise ValueError("ids must be 1-D list or array")

        

    batch_size = len(ids)

    ids = torch.LongTensor(ids).view(batch_size, 1)

    out_tensor = Variable(torch.FloatTensor(batch_size, nb_digits))

    out_tensor.data.zero_()

    out_tensor.data.scatter_(dim=1, index=ids, value=1.)

    return out_tensor
def uniform_linear_layer(linear_layer):

    linear_layer.weight.data.uniform_()

    linear_layer.bias.data.fill_(-0.02)
lake = gym.make('FrozenLake-v0')
lake.reset()

lake.render()
s1, r, d, _ = lake.step(2)

lake.render()
s1, r, d, _ = lake.step(1)

lake.render()
print(r)

print(d)
class Agent(nn.Module):

  

    """

    Observation Space - How big is the state that the agent needs to observe?

    In this case, the only thing that changes about the lake is the position of the agent.

    Therefore, the observation space is 1

    

    Action Space - Similar to the O-Space, we can move up, down, left, and right 

    Because we need to measure the Q-value of every action, the action space in this 

    case will be 4

    """

    def __init__(self, observation_space_size, action_space_size):

        super(Agent, self).__init__()

        self.observation_space_size = observation_space_size

        self.hidden_size = observation_space_size

        

        # What is the difference between observation and state space?

         

        """

        Let's build the nueral network. In RL, you'll find that large networks 

        are largely unessesary. Oftentimes, you can get away with just 1 or 2 hidden layers

        The reason should be intuitive. What makes something a cat or a dog has many, many variables

        But "wich direction should I walk on a 2D grid" has a lot fewer.

        

        As you can see, the output layer is our action space size. This will be a table

        of our possible actions, each with a q-value

        """



        ## Create a simple 3 layer network using the observation space size as the input

        ## And the action space size as the output



        # YOUR CODE HERE

        raise NotImplementedError()

        

        # Why might a nueral network for deep RL be relatively smaller than what you might expect in something like image classification

    

    # Forward feed of our network

    def forward(self, state):

        obs_emb = one_hot([int(state)], self.observation_space_size)

        out1 = F.sigmoid(self.l1(obs_emb))

        return self.l2(out1).view((-1)) # 1 x ACTION_SPACE_SIZE == 1 x 4  =>  4
class Trainer:

    def __init__(self):

        self.agent = Agent(lake.observation_space.n, lake.action_space.n)

        self.optimizer = optim.Adam(params=self.agent.parameters())

        self.success = []

        self.jList = []

        self.running_success = []

    

    def train(self, epoch):

      

      # Let's start by resetting our enviroment

      # We don't want to just wander back and forth forever when the simulation starts

      # Therefore, we use a j value that stops our agent from taking more than 200 

      # actions in a simulation

        for i in range(epoch):

            s = lake.reset()

            j = 0



            """

            # Rearrange these in the correct order

                self.optimizer.zero_grad()

                s = s1

                target_q = r + 0.99 * torch.max(self.agent(s1).detach()) 

                self.optimizer.step()

                if d == True: break

                a = self.choose_action(s)

                j += 1

                loss = F.smooth_l1_loss(self.agent(s)[a], target_q)

                s1, r, d, _ = lake.step(a)

                loss.backward()

                if d == True and r == 0: r = -1

            """

            while j < 200:

                

                # YOUR CODE HERE

                raise NotImplementedError()

            # append results onto report lists

            if d == True and r > 0:

                self.success.append(1)

            else:

                self.success.append(0)

            

            self.jList.append(j)

            

            if i % 100 == 0:

                print("last 100 epoches success rate: " + str(sum(self.success[-100:])) + "%")

                self.running_success.append(sum(self.success[-100:]))



    def choose_action(self, s):

        # 0.1 is our epsilon

        # Normally, we want some fancy way to degrade this (over time, we should

        #   be taking fewer random actions)

        # We will cover this a little more, but for this really, really simple

        #   example, we can just use a set epsilon

        if (np.random.rand(1) < 0.1): 

            return lake.action_space.sample()

        # Now, if we don't want to act randomly, we are going to feed forward

        #   the network

        # Then, we take the action that has the highest Q-value (max index)

        else:

            agent_out = self.agent(s).detach()

            _, max_index = torch.max(agent_out, 0)

            return int(max_index.data.numpy())
t = Trainer()

t.train(5000)
plt.plot(t.success)
plt.plot(t.jList)
plt.plot(t.running_success)