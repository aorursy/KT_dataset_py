%matplotlib inline



import gym

import itertools

import matplotlib

import numpy as np

import sys

import tensorflow as tf

import collections

import pandas as pd

from collections import namedtuple

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



import sklearn.pipeline

import sklearn.preprocessing



from sklearn.kernel_approximation import RBFSampler



matplotlib.style.use('ggplot')
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])



def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):

    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)

    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)

    X, Y = np.meshgrid(x, y)

    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))



    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,

                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)

    ax.set_xlabel('Position')

    ax.set_ylabel('Velocity')

    ax.set_zlabel('Value')

    ax.set_title("Mountain \"Cost To Go\" Function")

    fig.colorbar(surf)

    plt.show()





def plot_value_function(V, title="Value Function"):

    """

    Plots the value function as a surface plot.

    """

    min_x = min(k[0] for k in V.keys())

    max_x = max(k[0] for k in V.keys())

    min_y = min(k[1] for k in V.keys())

    max_y = max(k[1] for k in V.keys())



    x_range = np.arange(min_x, max_x + 1)

    y_range = np.arange(min_y, max_y + 1)

    X, Y = np.meshgrid(x_range, y_range)



    # Find value for all (x, y) coordinates

    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))

    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))



    def plot_surface(X, Y, Z, title):

        fig = plt.figure(figsize=(20, 10))

        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,

                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)

        ax.set_xlabel('Player Sum')

        ax.set_ylabel('Dealer Showing')

        ax.set_zlabel('Value')

        ax.set_title(title)

        ax.view_init(ax.elev, -120)

        fig.colorbar(surf)

        plt.show()



    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))

    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))







def plot_episode_stats(stats, smoothing_window=10, noshow=False):

    # Plot the episode length over time

    fig1 = plt.figure(figsize=(10,5))

    plt.plot(stats.episode_lengths)

    plt.xlabel("Episode")

    plt.ylabel("Episode Length")

    plt.title("Episode Length over Time")

    if noshow:

        plt.close(fig1)

    else:

        plt.show(fig1)



    # Plot the episode reward over time

    fig2 = plt.figure(figsize=(10,5))

    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    plt.plot(rewards_smoothed)

    plt.xlabel("Episode")

    plt.ylabel("Episode Reward (Smoothed)")

    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))

    if noshow:

        plt.close(fig2)

    else:

        plt.show(fig2)



    # Plot time steps and episode number

    fig3 = plt.figure(figsize=(10,5))

    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))

    plt.xlabel("Time Steps")

    plt.ylabel("Episode")

    plt.title("Episode per time step")

    if noshow:

        plt.close(fig3)

    else:

        plt.show(fig3)



    return fig1, fig2, fig3
env = gym.envs.make("MountainCarContinuous-v0")

env.observation_space.sample()
# Feature Preprocessing: Normalize to zero mean and unit variance

# We use a few samples from the observation space to do this

observation_examples = np.array([env.observation_space.sample() for x in range(10000)])

scaler = sklearn.preprocessing.StandardScaler()

scaler.fit(observation_examples)



# Used to converte a state to a featurizes represenation.

# We use RBF kernels with different variances to cover different parts of the space

featurizer = sklearn.pipeline.FeatureUnion([

        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),

        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),

        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),

        ("rbf4", RBFSampler(gamma=0.5, n_components=100))

        ])

featurizer.fit(scaler.transform(observation_examples))
def featurize_state(state):

    """

    Returns the featurized representation for a state.

    """

    scaled = scaler.transform([state])

    featurized = featurizer.transform(scaled)

    return featurized[0]
class PolicyEstimator():

    """

    Policy Function approximator. 

    """

    

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):

        with tf.variable_scope(scope):

            self.state = tf.placeholder(tf.float32, [400], "state")

            self.target = tf.placeholder(dtype=tf.float32, name="target")



            # This is just linear classifier

            self.mu = tf.contrib.layers.fully_connected(

                inputs=tf.expand_dims(self.state, 0),

                num_outputs=1,

                activation_fn=None,

                weights_initializer=tf.zeros_initializer)

            self.mu = tf.squeeze(self.mu)

            

            self.sigma = tf.contrib.layers.fully_connected(

                inputs=tf.expand_dims(self.state, 0),

                num_outputs=1,

                activation_fn=None,

                weights_initializer=tf.zeros_initializer)

            self.sigma = tf.squeeze(self.sigma)

            self.sigma = tf.nn.softplus(self.sigma) + 1e-5

            

            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

            self.action = self.normal_dist._sample_n(1)

            self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])



            # Loss and train op

            self.loss = -self.normal_dist.log_prob(self.action) * self.target

            # Add cross entropy cost to encourage exploration

            self.loss -= 1e-1 * self.normal_dist.entropy()

            

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.train_op = self.optimizer.minimize(

                self.loss, global_step=tf.contrib.framework.get_global_step())

    

    def predict(self, state, sess=None):

        sess = sess or tf.get_default_session()

        state = featurize_state(state)

        return sess.run(self.action, { self.state: state })



    def update(self, state, target, action, sess=None):

        sess = sess or tf.get_default_session()

        state = featurize_state(state)

        feed_dict = { self.state: state, self.target: target, self.action: action  }

        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss
class ValueEstimator():

    """

    Value Function approximator. 

    """

    

    def __init__(self, learning_rate=0.1, scope="value_estimator"):

        with tf.variable_scope(scope):

            self.state = tf.placeholder(tf.float32, [400], "state")

            self.target = tf.placeholder(dtype=tf.float32, name="target")



            # This is just linear classifier

            self.output_layer = tf.contrib.layers.fully_connected(

                inputs=tf.expand_dims(self.state, 0),

                num_outputs=1,

                activation_fn=None,

                weights_initializer=tf.zeros_initializer)



            self.value_estimate = tf.squeeze(self.output_layer)

            self.loss = tf.squared_difference(self.value_estimate, self.target)



            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            self.train_op = self.optimizer.minimize(

                self.loss, global_step=tf.contrib.framework.get_global_step())        

    

    def predict(self, state, sess=None):

        sess = sess or tf.get_default_session()

        state = featurize_state(state)

        return sess.run(self.value_estimate, { self.state: state })



    def update(self, state, target, sess=None):

        sess = sess or tf.get_default_session()

        state = featurize_state(state)

        feed_dict = { self.state: state, self.target: target }

        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss
def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):

    """

    Actor Critic Algorithm. Optimizes the policy 

    function approximator using policy gradient.

    

    Args:

        env: OpenAI environment.

        estimator_policy: Policy Function to be optimized 

        estimator_value: Value function approximator, used as a critic

        num_episodes: Number of episodes to run for

        discount_factor: Time-discount factor

    

    Returns:

        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.

    """



    # Keeps track of useful statistics

    stats = EpisodeStats(

        episode_lengths=np.zeros(num_episodes),

        episode_rewards=np.zeros(num_episodes))    

    

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    

    for i_episode in range(num_episodes):

        # Reset the environment and pick the fisrst action

        state = env.reset()

        

        episode = []

        

        # One step in the environment

        for t in itertools.count():

            

            # env.render()

            

            # Take a step

            action = estimator_policy.predict(state)

            next_state, reward, done, _ = env.step(action)

            

            # Keep track of the transition

            episode.append(Transition(

              state=state, action=action, reward=reward, next_state=next_state, done=done))

            

            # Update statistics

            stats.episode_rewards[i_episode] += reward

            stats.episode_lengths[i_episode] = t

            

            # Calculate TD Target

            value_next = estimator_value.predict(next_state)

            td_target = reward + discount_factor * value_next

            td_error = td_target - estimator_value.predict(state)

            

            # Update the value estimator

            estimator_value.update(state, td_target)

            

            # Update the policy estimator

            # using the td error as our advantage estimate

            estimator_policy.update(state, td_error, action)

            

            # Print out which step we're on, useful for debugging.

            print("\rStep {} @ Episode {}/{} ({})".format(

                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")



            if done:

                break

                

            state = next_state

    

    return stats
tf.reset_default_graph()



global_step = tf.Variable(0, name="global_step", trainable=False)

policy_estimator = PolicyEstimator(learning_rate=0.001)

value_estimator = ValueEstimator(learning_rate=0.1)



with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())

    # Note, due to randomness in the policy the number of episodes you need varies

    # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.

    stats = actor_critic(env, policy_estimator, value_estimator, 50, discount_factor=0.95)
plot_episode_stats(stats, smoothing_window=10)