import sys
import termcolor
import numpy as np
import pandas as pd
from IPython.display import display, clear_output


class FrozenLake:
    FROZEN, BLOCKED, HOLE, GOAL = range(4)
    TILE_NAMES = ["F", " ", "H", "G"]
    TILE_COLORS = ["grey", "white", "red", "green"]
    LEFT, TOP, RIGHT, BOTTOM = range(4)
    ACTIONS = [LEFT, TOP, RIGHT, BOTTOM]
    ACTION_MOVES = np.array([0, -1]), np.array([-1, 0]), np.array([0, 1]), np.array([1, 0])
    ACTION_NAMES = ["LEFT", "TOP", "RIGHT", "BOTTOM"]
    
    def __init__(self, grid, starting_pos, error_proba=0):
        self.grid = np.array(grid)
        self.states = list(range(self.grid.size))
        self.actions = self.ACTIONS
        self.starting_pos = np.array(starting_pos)
        self.error_proba = error_proba
        assert 0 <= self.error_proba <= 1
        self.reset()
    
    def state(self, pos=None):
        y, x = pos if pos is not None else self.pos 
        return y * len(self.grid[0]) + x
    
    def state_to_pos(self, state):
        y = state // len(self.grid[0])
        x = state % len(self.grid[0])
        return np.array([y, x])
        
    def reset(self, pos=None, state=None):
        if pos is None and state is None:
            pos = self.starting_pos
        elif state is not None:
            pos = self.state_to_pos(state)
        assert self.is_pos_valid(pos)
        self.pos = pos
        return self.state()

    def tile(self, pos=None):
        if pos is None:
            pos = self.pos
        y, x = pos
        return self.grid[y, x]
    
    def is_state_valid(self, state):
        return self.is_pos_valid(self.state_to_pos(state))
    
    def is_pos_valid(self, pos):
        y, x = pos
        if x < 0 or y < 0:
            return False
        try:
            tile = self.tile(pos=pos)
        except IndexError:
            return False
        return tile != self.BLOCKED
    
    def is_terminate_pos(self, pos):
        tile = self.tile(pos=pos)
        return tile in (self.GOAL, self.HOLE)
    
    def is_terminate_state(self, state):
        return self.is_terminate_pos(self.state_to_pos(state))

    def _reward(self, pos=None):
        tile = self.tile(pos=pos)
        if tile == self.GOAL:
            return 1
        elif tile == self.HOLE:
            return -1
        return 0

    def _perturbate_action(self, action):
        p = np.random.rand()
        if p < self.error_proba:
            return action
        if p < self.error_proba + (1-self.error_proba)/2:
            return self.ACTIONS[action-1]
        return self.ACTIONS[(action+1)%len(self.ACTIONS)]
    
    def step(self, action):
        if type(action) == str:
            action = self.ACTIONS[self.ACTION_NAMES.index(action)]
        pos = self.pos + self.ACTION_MOVES[self._perturbate_action(action)]
        if self.is_pos_valid(pos):
            self.pos = pos
        return self.state(), self._reward(), self.tile() in (self.GOAL, self.HOLE)

    def plot(self):
        print("")
        y, x = self.pos
        for i, _ in enumerate(self.grid):
            for j, _ in enumerate(self.grid[0]):
                if i == y and j == x:
                    termcolor.cprint("X", "blue", end="")
                else:
                    tile = self.tile([i, j])
                    termcolor.cprint(self.TILE_NAMES[tile], self.TILE_COLORS[tile], end="")
            print("")
        print("")
env = FrozenLake([
    [FrozenLake.FROZEN, FrozenLake.FROZEN, FrozenLake.FROZEN, FrozenLake.GOAL],
    [FrozenLake.FROZEN, FrozenLake.BLOCKED, FrozenLake.FROZEN, FrozenLake.HOLE],
    [FrozenLake.FROZEN, FrozenLake.FROZEN, FrozenLake.FROZEN, FrozenLake.FROZEN],
], [2, 0], error_proba=0.8)
env.plot()
state, reward, done = env.step(env.RIGHT)
def train_tabular(Q, env, episodes=1000, max_steps=100, discount_factor=0.9, lr=0.618, cb_episode=None):
    for episode in range(episodes):
        for state in env.states:
            if not env.is_state_valid(state) or env.is_terminate_state(state):
                continue
            for action in env.actions:
                env.reset(state=state)
                next_state, reward, _ = env.step(action)
                Q.iloc[state, action] = (1-lr) * Q.iloc[state, action] + lr * (reward + discount_factor * Q.loc[next_state].max())
        if not episode % 50:
            print('Episode:', episode, '/', episodes)
    
def init_Q_zeros(env):
    return pd.DataFrame(np.zeros((env.grid.size, len(env.ACTIONS))), columns=FrozenLake.ACTION_NAMES)

def plot_policy(env, Q):
    print(np.array([
        [
            Q.iloc[env.state(pos=(i, j))].idxmax() if env.is_pos_valid((i, j)) and not env.is_terminate_pos((i, j)) else ''
            for j in range(env.grid.shape[1])
        ]
        for i in range(env.grid.shape[0])
    ]))
Q = init_Q_zeros(env)
train_tabular(Q, env)
Q
plot_policy(env, Q)
def train_step_by_step(Q, env, episodes=1000, max_steps=100, discount_factor=0.9, lr=0.618, cb_episode=None):
    for episode in range(episodes):
        state = env.reset()
        done = False
        cb_step = None
        n_steps = 0
        if callable(cb_episode):
            cb_step = cb_episode(episode)
        while not done and n_steps < max_steps:
            action = Q.iloc[state].idxmax()
            next_state, reward, done = env.step(action)
            Q.loc[state, action] = (1-lr) * Q.loc[state, action] + lr * (reward + discount_factor * Q.loc[next_state].max())
            if callable(cb_step):
                cb_step(state, action, next_state, reward, Q)
            state = next_state
            n_steps += 1
Q = init_Q_zeros(env)
#train_step_by_step(Q, env)
Q
def cb_episode(episode):
    cum_reward = 0
    def cb_step(state, action, next_state, reward, Q):
        nonlocal cum_reward
        cum_reward += reward
        
        if episode == 0:
            print(state, action, next_state)
        if not episode % 50:
            print('Episode {0} reward:'.format(episode), cum_reward)
            
    return cb_step
Q = init_Q_zeros(env)
#train_step_by_step(Q, env, cb_episode=cb_episode)
def init_Q_random(env):
    return pd.DataFrame(np.random.rand(env.grid.size, len(env.ACTIONS)), columns=FrozenLake.ACTION_NAMES)

def cb_episode(episode):
    def cb_step(state, action, next_state, reward, Q):
        clear_output()
        display(Q)
    return cb_step
Q = init_Q_random(env)
#train_step_by_step(Q, env, cb_episode=cb_episode)
