!pip install 'kaggle-environments>=0.1.4'
from kaggle_environments import evaluate, make



env = make("connectx", debug=True)



env.render()
def my_agent(observation, configuration):

    

    from random import choice

          

    def check_enemy():

        if observation.board[24] == 2 and observation.board[17] != 1:

            return 3

        elif observation.board[23] == 2 and observation.board[16] != 1:

            return 2

        elif observation.board[25] == 2 and observation.board[18] != 1:

            return 4

        elif observation.board[22] == 2 and observation.board[15] != 1:

            return 1

        elif observation.board[26] == 2 and observation.board[19] != 1:

            return 5

        elif observation.board[21] == 2 and observation.board[14] != 1:

            return 0

        elif observation.board[27] == 2 and observation.board[20] != 1:

            return 6

        else:

            return -99

    

    result = check_enemy()

    

    if result != -99:

        return result

    

    elif observation.board[24] != 2 and observation.board[17] != 2 and observation.board[10] != 2 and observation.board[3] == 0:

        return 3

    elif observation.board[23] != 2 and observation.board[16] != 2 and observation.board[9] != 2 and observation.board[2] == 0:

        return 2

    elif observation.board[25] != 2 and observation.board[18] != 2 and observation.board[11] != 2 and observation.board[4] == 0:

        return 4

    elif observation.board[22] != 2 and observation.board[15] != 2 and observation.board[8] != 2 and observation.board[1] == 0:

        return 1

    elif observation.board[26] != 2 and observation.board[19] != 2 and observation.board[12] != 2 and observation.board[5] == 0:

        return 5

    elif observation.board[21] != 2 and observation.board[14] != 2 and observation.board[7] != 2 and observation.board[0] == 0:

        return 0

    elif observation.board[27] != 2 and observation.board[20] != 2 and observation.board[13] != 2 and observation.board[6] == 0:

        return 6

    else:

        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
env.reset()

# Play as the first agent against default "random" agent.

env.run([my_agent, "random"])

# env.run([my_agent, "negamax"])

env.render(mode="ipython", width=500, height=450)
trainer = env.train([None, "random"])



observation = trainer.reset()



env.configuration
def mean_reward(rewards):

    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)



# Run multiple episodes to estimate it's performance.

print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=100)))

print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=100)))

print("My Agent vs My Agent:", mean_reward(evaluate("connectx", [my_agent, my_agent], num_episodes=100)))

import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")