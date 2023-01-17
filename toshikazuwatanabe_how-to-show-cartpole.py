!apt-get install python-opengl -y

!pip install pyvirtualdisplay
import gym

import os

import matplotlib.pyplot as plt

from pyvirtualdisplay import Display



display = Display(visible=0, size=(1400, 900))

display.start()

os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)
env = gym.make("CartPole-v1")



env.reset()

plt.imshow(env.render('rgb_array'))

plt.grid(False)
from matplotlib import animation , rc



fig = plt.figure()



frame = []

env.reset()



for i in range(1_000):

    action = env.action_space.sample()

    state, reward, done, info = env.step(action)

    img = plt.imshow(env.render('rgb_array'))

    frame.append([img])

    if done:

        break



an = animation.ArtistAnimation(fig, frame, interval=100, repeat_delay=1000, blit=True)

rc('animation', html='jshtml')

an