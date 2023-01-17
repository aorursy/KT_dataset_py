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
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)



env.render()
def my_agent(obs, config):

    # Your code here: Amend the agent!

    import random

    import numpy as np

    import time

    

    start = time.time()

    

    def drop_piece(grid, col, mark, config):

        next_grid = grid.copy()

        for row in range(config.rows-1, -1, -1):

            if next_grid[row][col]==0:

                next_grid[row][col] = mark

                break

                

        return next_grid

    

    def check_window(window, num_disc, piece, config):

        return (window.count(piece)==num_disc and window.count(0)==config.inarow - num_disc)

    

    def count_windows(grid, num_disc, piece, config):

        num_windows = 0

        

        # Horizontal

        for row in range(config.rows):

            for col in range(config.columns - config.inarow + 1):

                window = list(grid[row, col:(col+config.inarow)])

                if check_window(window, num_disc, piece, config):

                    num_windows += 1

                

        # Vertical

        for col in range(config.columns):

            for row in range(config.rows - config.inarow + 1):

                window = list(grid[row:(row+config.inarow), col])

                if check_window(window, num_disc, piece, config):

                    num_windows += 1

                    

        # Positive Diagonal

        for row in range(config.rows - config.inarow + 1):

            for col in range(config.columns - config.inarow + 1):

                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])

                if check_window(window, num_disc, piece, config):

                    num_windows += 1

        

        #Negative Diagonal

        for row in range(config.inarow-1, config.rows):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])

                if check_window(window, num_disc, piece, config):

                    num_windows += 1

        

        return num_windows

    

    def get_heuristic(grid, mark, config):

        

        A = 1000000

        B = 10

        C = 1

        D = -100

        E = -1000

        

        num_twos = count_windows(grid, 2, mark, config)

        num_threes = count_windows(grid, 3, mark, config)

        num_fours = count_windows(grid, 4, mark, config)

        

        num_twos_opp = count_windows(grid, 2, mark%2+1, config)

        num_threes_opp = count_windows(grid, 3, mark%2+1, config)

        num_fours_opp = count_windows(grid, 4, mark%2+1, config)

        

        score = A*num_fours + B*num_threes + C*num_twos + D*num_twos_opp + E*num_threes_opp

        return score

    

    def is_terminal_window(window, config):

        return window.count(1) == config.inarow or window.count(2) == config.inarow

    

    def is_terminal_node(grid, config):

        

        # Check for a tie

        if list(grid[0,:]).count(0)==0:

            return True

        

        # Horizontal

        for row in range(config.rows):

            for col in range(config.columns - config.inarow + 1):

                window = list(grid[row, col:(col+config.inarow)])

                if is_terminal_window(window, config):

                    return True

                

        # Vertical

        for col in range(config.columns):

            for row in range(config.rows - config.inarow + 1):

                window = list(grid[row:(row+config.inarow), col])

                if is_terminal_window(window, config):

                    return True

                    

        # Positive Diagonal

        for row in range(config.rows - config.inarow + 1):

            for col in range(config.columns - config.inarow + 1):

                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])

                if is_terminal_window(window, config):

                    return True

        

        #Negative Diagonal

        for row in range(config.inarow-1, config.rows):

            for col in range(config.columns-(config.inarow-1)):

                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])

                if is_terminal_window(window, config):

                        return True

        

        return False

    

    def minimax(node, depth, maximizingPlayer, mark, alpha, beta, config):

        

        # Base case

        if depth==0 or is_terminal_node(node, config):

            return get_heuristic(node, mark, config)

        

        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]

        

        # Recursive case

        if maximizingPlayer:

            value = -np.Inf

            for col in valid_moves:

                child = drop_piece(node, col, mark, config)

                value = max(value, minimax(child, depth-1, False, mark, alpha, beta, config))

                alpha = max(alpha, value)

                if alpha>=beta:

                    break

            return value

        

        else:

            value = np.Inf

            for col in valid_moves:

                child = drop_piece(node, col, mark%2+1, config)

                value = min(value, minimax(child, depth-1, True, mark, alpha, beta, config))

                beta = min(beta, value)

                if beta<=alpha:

                    break

            return value

        

    def score_move(grid, col, mark, depth, config):

        next_grid = drop_piece(grid, col, mark, config)

        alpha = -np.Inf

        beta = np.Inf

        score = minimax(next_grid, depth-1, False, mark, alpha, beta, config)

        return score



    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)

    

    depth = 3

    

    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, depth, config) for col in valid_moves]))

    

    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]

    

    end = time.time()

    print(end-start)

    return random.choice(max_cols)
env.reset()

env.run([my_agent, my_agent])

env.render(mode="ipython", width=500, height=450)
import numpy as np

def get_win_percentages(agent1, agent2, n_rounds=10):

    # Use default Connect Four setup

    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time          

    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)

    # Agent 2 goes first (roughly) half the time      

    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))

    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))

    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))

    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))

    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0, 0]))
# get_win_percentages(agent1=my_agent, agent2=my_agent)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")