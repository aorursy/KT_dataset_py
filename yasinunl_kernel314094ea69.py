# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -*- coding: utf-8 -*-
import pygame
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Hiper Parameters
WIDTH = 360
HEIGHT = 360
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#------------------------------- P L A Y E R ---------------------------------

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLUE)
        
        self.rect = self.image.get_rect()
        
        self.radius = 10
        pygame.draw.circle(self.image, RED , self.rect.center , self.radius)
        
        
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 1
         
        self.speedx = 0
    
    def update(self, action):
        self.speedx = 0
        keyState = pygame.key.get_pressed()
        
        if (action == 0 or keyState[pygame.K_LEFT]) and self.rect.left > 0 :
            self.speedx = -8
        elif (action == 2 or keyState[pygame.K_RIGHT]) and self.rect.right < HEIGHT:
            self.speedx = +8
        elif action == 1:
            self.speedx = 0
        self.rect.x += self.speedx


    def get_coordinates(self):
        return (self.rect.x, self.rect.y)

#-----------------------------------------------------------------------------
#------------------------------ E N E M Y ------------------------------------

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(BLUE)
        
        global player
        global enemy
        self.hits = False
        
        self.rect = self.image.get_rect()
        
        radius = 5
        pygame.draw.circle(self.image, WHITE, self.rect.center, radius)
        
        self.rect.centerx = random.randint(0 + self.rect.width, WIDTH )
        self.rect.bottom = random.randint(-150 , 0 )
        self.yspeed = 5

    def update(self):
        self.rect.y += self.yspeed
        if self.rect.y > HEIGHT:
            self.rect.bottom = random.randint(-150 , 0 )
            self.rect.centerx = random.randint(0 + self.rect.width, WIDTH )
        self.hits = pygame.sprite.spritecollide(player,enemy ,False,
                                            pygame.sprite.collide_circle)
        if self.hits:
            self.rect.bottom = random.randint(-150 , 0 )
            self.rect.centerx = random.randint(0 + self.rect.width, WIDTH )


    def get_Coordinates(self):
        return (self.rect.x,self.rect.y)


#-----------------------------------------------------------------------------
#-------------------------------- A G E N T ----------------------------------

class Agent:
    def __init__(self):
        self.state_size = 4
        self.action_size = 3
        self.learning_rate = 0.0001
        self.gamma = 0.95
        
        self.model = self.create_model()
        
        self.memory = deque(maxlen = 1000)
        
        self.epsilon = 1 
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(self.action_size, activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    
    def replay(self,batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory,batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            next_state = np.array(next_state) 
            state = np.array(state) 
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose = 0)
    
    
    def adaptiveE(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    
    def action(self, state):
        state = np.array(state)
        if np.random.rand() < self.epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(self.model.predict(state)[0])
    
    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))


#-----------------------------------------------------------------------------
#----------------------------------- E N V -----------------------------------

class ENV(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.reward = 0
        self.min_reward = 0
        self.max_reward = 0
        self.done = False
        self.agent = Agent()
        
        self.total_reward = []
        
        self.all_player_list = pygame.sprite.Group()
        
        global enemy
        global player
        
        player = Player()
        self.player = player
        self.e1 = Enemy()
        self.all_player_list.add(player)
        
        enemy = pygame.sprite.Group()
        self.enemy = enemy
        enemy.add(self.e1)
    
    def step(self, action):
        player.update(action)
        enemy.update()
        
        # Next State
        state_list = []
        
        next_e1_state = self.e1.get_Coordinates()
        next_player_state = player.get_coordinates()
        
        state_list.append(next_player_state[0])
        state_list.append(next_player_state[1])
        state_list.append(next_e1_state[0])
        state_list.append(next_e1_state[1])
        
        return [state_list]
    
    def initial_states(self):
        self.reward = 0
        self.done = False
        
        self.all_player_list = pygame.sprite.Group()
        global player
        player = Player()
        
        self.player = player 
        
        self.e1 = Enemy()
        self.all_player_list.add(self.player)
        
        enemy = pygame.sprite.Group()
        self.enemy = enemy
        
        enemy.add(self.e1)
        
        # State
        state_list = []
        
        next_e1_state = self.e1.get_Coordinates()
        next_player_state = player.get_coordinates()
        
        state_list.append(next_player_state[0])
        state_list.append(next_player_state[1])
        state_list.append(next_e1_state[0])
        state_list.append(next_e1_state[1])
        
        return [state_list]
    
    def run(self):
        pygame.init()
        clock = pygame.time.Clock()
        state = self.initial_states()
        running = True
        while running:
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            action = self.agent.action(state)
            next_state = self.step(action)
            
            # Hedefi yakalarsa +25, ıskalarsa -50 ve oyun biter
            for i in enemy.sprites():
                if i.hits:
                    self.reward = 25
                    i.hits = False
                if i.rect.y > HEIGHT - 5:
                    self.reward = -50
                    self.done = True
                    running = False
           
            
            self.agent.remember(state, action, self.reward, next_state, self.done)
            state = next_state
            
            self.agent.replay(24)
            self.agent.adaptiveE()
            
            
            
            if self.min_reward > self.reward:
                self.min_reward = self.reward
            if self.max_reward < self.reward:
                self.max_reward = self.reward
        
        self.total_reward.append(self.reward)
        pygame.quit()


#-----------------------------------------------------------------------------
#---------------------------------- M A I N ----------------------------------

if __name__ == "__main__":
    enemy = ""
    player = ""
    
    
    env = ENV()
    e = 0
    while True:
        e += 1
        env.run()
        print("Episodes: ", e)
        print("Reward: ", env.reward)
    
    
    print("Min. Reward: ", env.min_reward)
    print("Max. Reward: ", env.max_reward)
 





# -*- coding: utf-8 -*-
import pygame
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Hiper Parameters
WIDTH = 360
HEIGHT = 360
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

#------------------------------- P L A Y E R ---------------------------------

class Player(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLUE)
        
        self.rect = self.image.get_rect()
        
        self.radius = 10
        pygame.draw.circle(self.image, RED , self.rect.center , self.radius)
        
        
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 1
         
        self.speedx = 0
    
    def update(self, action):
        self.speedx = 0
        keyState = pygame.key.get_pressed()
        
        if (action == 0 or keyState[pygame.K_LEFT]) and self.rect.left > 0 :
            self.speedx = -8
        elif (action == 2 or keyState[pygame.K_RIGHT]) and self.rect.right < HEIGHT:
            self.speedx = +8
        elif action == 1:
            self.speedx = 0
        self.rect.x += self.speedx


    def get_coordinates(self):
        return (self.rect.x, self.rect.y)

#-----------------------------------------------------------------------------
#------------------------------ E N E M Y ------------------------------------

class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(BLUE)
        
        global player
        global enemy
        self.hits = False
        
        self.rect = self.image.get_rect()
        
        radius = 5
        pygame.draw.circle(self.image, WHITE, self.rect.center, radius)
        
        self.rect.centerx = random.randint(0 + self.rect.width, WIDTH )
        self.rect.bottom = random.randint(-150 , 0 )
        self.yspeed = 5

    def update(self):
        self.rect.y += self.yspeed
        if self.rect.y > HEIGHT:
            self.rect.bottom = random.randint(-150 , 0 )
            self.rect.centerx = random.randint(0 + self.rect.width, WIDTH )
        self.hits = pygame.sprite.spritecollide(player,enemy ,False,
                                            pygame.sprite.collide_circle)
        if self.hits:
            self.rect.bottom = random.randint(-150 , 0 )
            self.rect.centerx = random.randint(0 + self.rect.width, WIDTH )


    def get_Coordinates(self):
        return (self.rect.x,self.rect.y)


#-----------------------------------------------------------------------------
#-------------------------------- A G E N T ----------------------------------

class Agent:
    def __init__(self):
        self.state_size = 4
        self.action_size = 3
        self.learning_rate = 0.0001
        self.gamma = 0.95
        
        self.model = self.create_model()
        self.target_model = self.create_model()
        
        self.memory = deque(maxlen = 1000)
        
        self.epsilon = 1 
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(self.action_size, activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    
    def replay(self,batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        
        not_down_indicates = np.where(minibatch[:,4] == False)
        y = np.copy(minibatch[:,2])
        
        if len(not_down_indicates) < 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:,3]))
            predict_sprime_target = self.target_model.predict(
                np.vstack(minibatch[:,3]))
            
            y[not_down_indicates] += np.multiply(self.gamma,
                                                 predict_sprime_target[not_down_indicates,
                                                                       np.argmax(predict_sprime[
                                                                           not_down_indicates,:][0],
                                                                           axis = 1)][0])
            
            action = np.array(minibatch[:,1] , dtype = int)
            y_target = self.target_model.predict(np.vstack(minibatch[:,0]))
            y_target[range(batch_size),action] = y
            self.model.fit(np.vstack(minibatch[:,0]), y_target, epoch = 1, verbose = 0)

        
        
        
        
        
        
        
        
        
        # minibatch = random.sample(self.memory,batch_size)
        
        # for state, action, reward, next_state, done in minibatch:
        #     next_state = np.array(next_state) 
        #     state = np.array(state) 
        #     if done:
        #         target = reward
        #     else:
        #         target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        #     train_target = self.model.predict(state)
        #     train_target[0][action] = target
        #     self.model.fit(state, train_target, verbose = 0)
    
    
    def adaptiveE(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    
    def action(self, state):
        state = np.array(state)
        if np.random.rand() < self.epsilon:
            return random.randint(0, 2)
        else:
            return np.argmax(self.model.predict(state)[0])
    
    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

    def targetModelUpdate(self):
        self.target_model.set_weights(self.model.get_weights())


#-----------------------------------------------------------------------------
#----------------------------------- E N V -----------------------------------

class ENV(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.reward = 0
        self.total = 0
        self.min_reward = 0
        self.max_reward = 0
        self.done = False
        self.agent = Agent()
        
        self.total_reward = []
        
        self.all_player_list = pygame.sprite.Group()
        
        global enemy
        global player
        
        player = Player()
        self.player = player
        self.e1 = Enemy()
        self.all_player_list.add(player)
        
        enemy = pygame.sprite.Group()
        self.enemy = enemy
        enemy.add(self.e1)
    
    def step(self, action):
        player.update(action)
        enemy.update()
        
        state_list = []
        
        next_e1_state = self.e1.get_Coordinates()
        next_player_state = player.get_coordinates()
        
        state_list.append(next_player_state[0])
        state_list.append(next_player_state[1])
        state_list.append(next_e1_state[0])
        state_list.append(next_e1_state[1])
        
        return [state_list]
    
    def initial_states(self):
        self.reward = 0
        self.done = False
        self.total = 0
        self.all_player_list = pygame.sprite.Group()
        global player
        player = Player()
        
        self.player = player 
        
        self.e1 = Enemy()
        self.all_player_list.add(self.player)
        
        enemy = pygame.sprite.Group()
        self.enemy = enemy
        
        enemy.add(self.e1)
        
        state_list = []
        
        next_e1_state = self.e1.get_Coordinates()
        next_player_state = player.get_coordinates()
        
        state_list.append(next_player_state[0])
        state_list.append(next_player_state[1])
        state_list.append(next_e1_state[0])
        state_list.append(next_e1_state[1])
        
        return [state_list]
    
    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT)) # Window 

        clock = pygame.time.Clock()
        state = self.initial_states()
        running = True
        while running:
            clock.tick(FPS)
            self.reward = 0.1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            action = self.agent.action(state)
            next_state = self.step(action)
            
            for i in enemy.sprites():
                if i.hits:
                    self.reward = 25
                    i.hits = False
                if i.rect.y > HEIGHT - 5:
                    self.reward = -50
                    self.done = True
                    running = False
                    
            
            
            self.agent.remember(state, action, self.reward, next_state, self.done)
            state = next_state
            
            self.agent.replay(24)
            self.agent.adaptiveE()
            
            if self.done:
                self.agent.targetModelUpdate()
            
            if self.min_reward > self.reward:
                self.min_reward = self.reward
            if self.max_reward < self.reward:
                self.max_reward = self.reward
                
            self.total += self.reward    
            screen.fill(GREEN)
            self.all_player_list.draw(screen)
            enemy.draw(screen)
            pygame.display.flip()
            
            
        self.total_reward.append(self.reward)
        pygame.quit()


#-----------------------------------------------------------------------------
#---------------------------------- M A I N ----------------------------------

if __name__ == "__main__":
    enemy = ""
    player = ""
    
    
    env = ENV()
    e = 0
    while True:
        e += 1
        env.run()
        print("Episodes: ", e)
        print("Reward: ", env.total)
    
    
    print("Min. Reward: ", env.min_reward)
    print("Max. Reward: ", env.max_reward)







