# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# ConnectX environment was defined in v0.1.6

!pip install 'kaggle-environments==0.1.6'
from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
def my_agent(observation, configuration):

    import time

    start = time.time()

    import numpy as np

    from random import choice

    limit = configuration.timeout - 0.7

    rewards = np.zeros(configuration.columns)



    # OPENING BOOK

    # Play in the middle of the board on our first move

    if sum(observation.board) <= 3:

        return int(configuration.columns/2)



    # CHECK INSTANT WIN/LOSS

    # WIN: use is_win on a copy of this subgame by playing obs.mark in each column

    def check_instant_win(observation, configuration):

        # our turn

        mark = observation.mark

        win = -1

        # for all columns with an open space

        for i in [c for c in range(configuration.columns) if observation.board[c] == 0]:

            # make copy of board

            temp_copy = observation.board.copy()

            play(temp_copy, i, mark, configuration)

            # if that play yields a win for us

            if is_win(temp_copy, i, mark, configuration, has_played=True):

                win = i

        return win





    # LOSS: use is_win on a copy of this subgame by playing other mark in each column

    def check_instant_loss(observation, configuration):

        # figure out whose turn it is

        mark = observation.mark

        # change mark to the other player

        mark = (mark%2)+1



        loss = -1

        # for all columns with an open space

        for i in [c for c in range(configuration.columns) if observation.board[c] == 0]:

            # make a temporary copy of the board that got passed in

            temp_copy = observation.board.copy()

            play(temp_copy, i, mark, configuration)

            # if that yields a win for the other player

            if is_win(temp_copy, i, mark, configuration, has_played=True):

                loss = i

        return loss



    # Checks if placing a piece in the column wins the game

    def is_winning_move(column, board, config, mark):

        columns = config.columns

        rows = config.rows

        row = max([r for r in range(rows) if board[column + (r * columns)] == 0])

        board[column + (row * columns)] = mark

        if is_win(board, column, mark, config):

            board[column + (row * columns)] = 0

            return True

        board[column + (row * columns)] = 0

        return False



    # Checks if the opponent placing a piece in the column would win them the game

    def blocks_opponent_win(column, board, config, mark):

        reverse_mark = (mark % 2) + 1

        return is_winning_move(column, board, config, reverse_mark)



    # Always plays a winning move, always plays a force move--otherwise random

    def intelligent_rando(board, config, mark):



        options = [c for c in range(config.columns) if board[c] == 0]

        for c in options:

            if is_winning_move(c, board, config, mark):

                return c



        for c in options:

            if blocks_opponent_win(c, board, config, mark):

                return c



        if len(options)>0:

            return choice(options)

        else:

            return -1



    # So this function will make a copy of the game board it receives for every playable column (in iteration).  Then, for column i (where i is a playable column in obs.board), our agent plays in column i on board_copy.  Switch the mark to the other player, and then for every column j (where j is a playable column in board_copy), make a copy of the board_copy and play in column j on the copy of the copy.  If that play j yields a win for the opponent, we add i (the move we made) to a list of bad moves.  Return that list

    # check that a move we plan to make won't give the opponent an easy win

    def prevent_easy_win(obs, config):

        bad_moves = []

        

        # for each open column i

        for i in [c for c in range(config.columns) if obs.board[c] == 0]:

            mark = obs.mark

            board_copy = obs.board.copy()

            # 'play' in column i

            play(board_copy, i, mark, config)

            # change mark to opposing player

            mark = (mark%2)+1

            

            # see if the opponent can win easily b/c of our move

            for j in [c for c in range(config.columns) if board_copy[c] == 0]:

                # make another copy of board

                board_copy2 = board_copy.copy()

                # play for the opponent in each column

                play(board_copy2, j, mark, config)



                # if yes, add i to an array of nonplayables

                if is_win(board_copy2, j, mark, config, has_played=True):

                    bad_moves.append(i)

                    break

        return bad_moves





    # credit this to kaggle env

    def play(board, column, mark, config):

        columns = config.columns

        rows = config.rows

        row = max([r for r in range(rows) if board[column + (r * columns)] == 0])

        board[column + (row * columns)] = mark



    def is_win(board, column, mark, config, has_played=True):

        columns = config.columns

        rows = config.rows

        inarow = config.inarow - 1

        row = (

             min([r for r in range(rows) if board[column + (r * columns)] == mark])

             if has_played

             else max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])

             )



        def count(offset_row, offset_column):

            for i in range(1, inarow + 1):

                r = row + offset_row * i

                c = column + offset_column * i

                if (

                    r < 0

                   or r >= rows

                    or c < 0

                    or c >= columns

                    or board[c + (r * columns)] != mark

                ):

                    return i - 1

            return inarow



        return (

            count(1, 0) >= inarow  # vertical.

            or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.

            or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.

            or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.

        )

     # end credit









    # check for instant win and play it if there is one

    win = int(check_instant_win(observation, configuration))

    if win != -1:

        return win



    # check for instant loss and play it if there is one

    loss = int(check_instant_loss(observation, configuration))

    if loss != -1:

        return loss



    bad_moves = prevent_easy_win(observation, configuration)



    

    now = time.time()

    while now-start < limit:

        # for each column

        for i in [c for c in range(configuration.columns) if observation.board[c] == 0 and c not in bad_moves]:

            

            # copy the current board

            board_copy = observation.board.copy()



            # track whose turn it is to play

            to_move = observation.mark



            # play the first new move on this board state in ith column

            play(board_copy, i, to_move, configuration)



            # change to_move to the other player

            to_move = (to_move%2) + 1



            playing = True

            winner = 0

            # now we enter the randoms' game

            while playing==True:

                #print(board_copy)

                # get a move from random player

                move = intelligent_rando(board_copy, configuration, to_move)

                # check that we havent come up with an error from randotron

                if move==-1:

                    winner = -1

                    playing = False

                    continue





                # play that move on the board copy

                play(board_copy, move, to_move, configuration)

                # see if that move has yielded a winner

                if is_win(board_copy, move, to_move, configuration, has_played=True):

                    # set winner to the mark of player who just went

                    winner = to_move

                    playing = False

                    continue

                # make it the other random player's move

                to_move = (to_move%2) + 1



            # if the random continuing for our mc player won, increase reward by 1.0

            if winner == observation.mark:

                rewards[i] = rewards[i] + 1.0

            # if nobody won, increase reward by 0.5

            elif winner == -1:

                rewards[i] = rewards[i] + 0.5

            # if the other random won, increase reward by 0.0

            else:

                rewards[i] = rewards[i] + 0.0

        now = time.time()

    best = int(np.argmax(rewards))

    if best != -1 and observation.board[best]==0:

        return best

    else:

        return int(choice([i for i in range(configuration.columns) if observation.board[i] == 0]))
env = make("connectx", debug=True)
observation = env.state[0].observation

print(my_agent(observation, env.configuration))
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")