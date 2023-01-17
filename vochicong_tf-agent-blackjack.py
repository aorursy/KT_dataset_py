# !which python

# !sudo apt install -y cuda-cublas-10-0  cuda-cusolver-10-0 cuda-cudart-10-0 cuda-cusparse-10-0

# !conda install -y -c anaconda cudatoolkit

!pip install tf-nightly-gpu tf-agents-nightly 'gym==0.10.11'

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

from tf_agents.drivers import dynamic_step_driver

from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.agents.dqn import q_network

from tf_agents.agents.dqn import dqn_agent

from tf_agents.drivers import dynamic_episode_driver

from tf_agents.metrics import tf_metrics

from tf_agents.policies import random_tf_policy

from tf_agents.environments import utils

import numpy as np

import tensorflow as tf

from tf_agents.environments import py_environment

from tf_agents.environments import tf_environment

from tf_agents.environments import tf_py_environment

from tf_agents.environments import suite_gym

from tf_agents.environments import time_step

from tf_agents.specs import array_spec



tf.compat.v1.enable_v2_behavior()

assert tf.executing_eagerly()

# tf.enable_eager_execution()



DEBUG = False

num_eval_episodes = 5  # @param





def plog(msg, *args):

    if DEBUG:

        print(msg.format(*args))





class BlackJackEnv(py_environment.PyEnvironment):

    # Simplified Blackjack

    ACT_HIT = 0

    ACT_STICK = 1

    LIMIT_SCORE = 21



    def __init__(self, state_len=2):

        print(f"state_len: {state_len}")

        self._batch_size = 1  # batch_size

        self._state_len = state_len

        self._action_spec = array_spec.BoundedArraySpec(

            shape=(), dtype=np.int32, name='action',

            minimum=self.ACT_HIT, maximum=self.ACT_STICK,

        )

        self._observation_spec = array_spec.BoundedArraySpec(

            shape=(self._state_len,), dtype=np.int32, minimum=0,

            name='observation'

        )

        self.reset()

        return



    def _state(self):

        if self._state_len == 1:

            return self._state_player_sum()

        if self._state_len == 2:

            return self._state_dealer1st_player_sum()

        return self._state_last_cards()



    def _state_dealer1st_player_sum(self):

        # Return the player current score

        state = [self._dealer_cards[0], np.sum(self._player_cards)]

        return np.array(state, dtype=np.int32)



    def _state_player_sum(self):

        # Return the player current score

        state = [np.sum(self._player_cards)]

        return np.array(state, dtype=np.int32)



    def _state_last_cards(self):

        # Full state includes 1st card of the dealer and all cards of player,

        # but this return only the last _state_len cards.

        state = [self._dealer_cards[0]] + self._player_cards

        if len(state) < self._state_len:

            state = np.pad(state, (0, self._state_len-len(state)),

                           'constant', constant_values=(0))

        return np.array(state[-self._state_len:], dtype=np.int32)



    def action_spec(self):

        return self._action_spec



    def observation_spec(self):

        return self._observation_spec



    def __reset(self):

        self._player_cards = [self._new_card(), self._new_card()]

        self._dealer_cards = [self._new_card()]

        self._episode_ended = False



    def _reset(self):

        self.__reset()

        # self._current_time_step = time_step.restart(self._state())

        # return self._current_time_step

        return time_step.restart(self._state())



    def _new_card(self):

        # Simplified Blackjack rule

        new_card = np.random.randint(1, 11+1)

        return new_card



    def _dealer_hit(self):

        while np.sum(self._dealer_cards) < 17:

            self._dealer_cards.append(self._new_card())

        return np.sum(self._dealer_cards)



    def _player_score(self):

        return np.sum(self._player_cards)



    def _terminate(self, reward):

        plog(

            "Player: {} -> {}. Dealer: {} -> {}. Reward: {}.",

            self._player_cards, np.sum(self._player_cards),

            self._dealer_cards, np.sum(self._dealer_cards),

            reward)

        self._episode_ended = True

        return time_step.termination(self._state(), reward)



    def _step(self, action):

        if self._episode_ended:

            return self.reset()  # don't forget to `return`



        if action == self.ACT_HIT:

            self._player_cards.append(self._new_card())

            if self._player_score() > self.LIMIT_SCORE:  # the player goes bust

                return self._terminate(-1)



            return time_step.transition(self._state(), reward=0, discount=1)



        # Afteward action == self.ACT_STICK

        dealer_score = self._dealer_hit()

        player_score = self._player_score()

        if dealer_score > self.LIMIT_SCORE or dealer_score < player_score:

            reward = 1

        elif dealer_score == player_score:

            reward = 0

        else:

            reward = -1

        return self._terminate(reward)



    @classmethod

    def tf_env(cls, state_len=2):

        return tf_py_environment.TFPyEnvironment(cls(state_len))





def print_spec(env):

    act_spec, ts_spec = env.action_spec(), env.time_step_spec()

    for x in (act_spec, ts_spec.observation, ts_spec.step_type,

              ts_spec.discount, ts_spec.reward):

        print(x)

    return





# TODO: validate_py_environment should check for a reset()

utils.validate_py_environment(BlackJackEnv())





def play_blackjack(env, n_max_cards=1):

    ts = env.reset()

    gain = ts.reward

    cards = []

    for _ in range(np.random.randint(n_max_cards+1)):

        if ts.is_last():

            break

        ts = env.step(tf.constant([BlackJackEnv.ACT_HIT]))

        cards += [ts.observation[0][0].numpy()]

        gain += ts.reward



    if not ts.is_last():

        ts = env.step(tf.constant([BlackJackEnv.ACT_STICK]))

        gain += ts.reward

    gain = gain.numpy()[0]

    return cards, gain





env = BlackJackEnv.tf_env()

gains = []

for _ in range(num_eval_episodes):

    _, gain = play_blackjack(env, 2)

    gains.append(gain)

mean_score1 = np.mean(gains)

mean_score1





def evaluate_policy(

        policy,

        num_episodes=num_eval_episodes,

        eval_env=BlackJackEnv.tf_env(),

):

    avg_return = tf_metrics.AverageReturnMetric()

    # n_episodes = tf_metrics.NumberOfEpisodes()

    # n_steps = tf_metrics.EnvironmentSteps()

    observers = [avg_return,

                 #  n_episodes, n_steps

                 ]

    driver = dynamic_episode_driver.DynamicEpisodeDriver(

        eval_env, policy, observers, num_episodes)

    final_step, policy_state = driver.run(num_episodes=num_episodes)

    # print('Number of Steps: ', n_steps.result().numpy())

    # print('Number of Episodes: ', n_episodes.result().numpy())

    # print('Average Return: ', avg_return.result().numpy())

    return driver, final_step, policy_state, avg_return.result().numpy()





DEBUG = False

env = BlackJackEnv.tf_env()

rand_policy = random_tf_policy.RandomTFPolicy(

    action_spec=env.action_spec(),

    time_step_spec=env.time_step_spec(),)
avg_returns = []

for n_episodes in range(1000, 6000, 1000):

    _, _, _, avg_return = evaluate_policy(rand_policy, num_episodes=n_episodes)

    print(f"n_episodes: {n_episodes}, avg_return: {avg_return}")

    avg_returns.append(avg_return)









plt.plot(avg_returns)



DEBUG = False

log_interval = 10  # @param

eval_interval = log_interval*5  # @param

num_iterations = 100_000  # @param

learning_rate = 1e-4  # @param

batch_size = 100*10  # @param

collect_steps_per_iteration = 100  # @param

initial_collect_steps = batch_size  # @param

num_eval_episodes = batch_size  # @param

replay_buffer_capacity = 10_000  # @param

fc_layer_params = (100, 90, 80, 70, 60, 50, 40, 30, 20)  # @param





class DqnAgent:

    def __init__(self, env):

        # Agent初期化

        self.env = env

        q_net = q_network.QNetwork(

            env.observation_spec(),

            env.action_spec(),

            fc_layer_params=fc_layer_params,

        )



        adam = tf.compat.v1.train.AdamOptimizer(

            learning_rate=learning_rate, beta1=0.8, epsilon=1)



        train_step_counter = tf.compat.v2.Variable(0)



        self.agent = dqn_agent.DqnAgent(

            env.time_step_spec(),

            env.action_spec(),

            q_network=q_net,

            optimizer=adam,

            td_errors_loss_fn=dqn_agent.element_wise_squared_loss,

            train_step_counter=train_step_counter,

        )

        self.agent.initialize()

        self._create_replay_buffer()



    # TODO: try different num_steps value

    def _create_replay_buffer(self, num_steps=2):

        # Replay Bufferの初期化。初期データ収集

        self.replay_buffer = buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(

            data_spec=self.agent.collect_data_spec,

            batch_size=self.env.batch_size,  # actually 1, env isn't batched

            max_length=replay_buffer_capacity

        )

        print(buffer.capacity.numpy(), buffer._batch_size)

        print(buffer.data_spec)

        self._collect_data(

            self.agent.collect_policy,

            initial_collect_steps)

        dataset = buffer.as_dataset(

            num_parallel_calls=3, num_steps=num_steps,

            sample_batch_size=batch_size,

        ).prefetch(batch_size)

        self.data_iterator = iter(dataset)



    def _collect_data(self, policy, n_steps):

        # Replay Bufferへのデータ追加

        dynamic_step_driver.DynamicStepDriver(

            self.env, policy, [self.replay_buffer.add_batch], n_steps

        ).run()

        return



    def train(self, num_iterations):

        _, _, _, avg_return = evaluate_policy(

            self.agent.policy, num_eval_episodes)

        avg_returns = [avg_return]

        for step in range(1, 1 + num_iterations):

            self._collect_data(self.agent.collect_policy,

                               collect_steps_per_iteration)

            experience, _ = next(self.data_iterator)

            train_loss = self.agent.train(experience)

            self._print_log(step, train_loss.loss, avg_returns)

        return avg_returns



    def _print_log(self, step, loss, avg_returns):

        if step % log_interval == 0:

            print(f'Step {step: >3}. Loss {loss}.')

        if step % eval_interval == 0:

            _, _, _, avg_return = evaluate_policy(

                self.agent.policy, num_eval_episodes)

            print(f'Step {step: >3}. AvgReturn {avg_return}.')

            avg_returns.append(avg_return)



def plot(avg_returns, num_iterations, eval_interval):

    steps = range(0, num_iterations + 1, eval_interval)

    plt.ylabel('Average Return')

    plt.xlabel('Step')

    plt.plot(steps, avg_returns)

    # plt.ylim(top=210)





# Set a `bad` _state_len and see that it can't learn

dqn = DqnAgent(BlackJackEnv.tf_env(state_len=2))



eval_interval = log_interval*5  # @param

num_iterations = eval_interval*10  # @param

avg_returns = dqn.train(num_iterations)



plot(avg_returns, num_iterations, eval_interval)

# assert flat_action_spec[0].shape.ndims <= 1
# (1,).ndim

# https://github.com/tensorflow/agents/blob/154b81176041071a84b72eb64d419d256dcc947a/tf_agents/agents/dqn/examples/v2/train_eval.py