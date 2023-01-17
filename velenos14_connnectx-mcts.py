!pip install 'kaggle-environments>=0.1.6'

from kaggle_environments import evaluate, make, utils

import random

import math

import time
env = make("connectx", debug=True)

configuration = env.configuration

print(configuration) # just to check
def MCTS_agent(observation, configuration):

    import random

    import math

    import time

    global curent_state



    init_time = time.time()

    EMPTY = 0

    # this algo is a every_move algo. it will be run at every position which is given as input from the kaggle interface.

    # it cannot run forever to expand the search tree indefinitely. it has to stop after a fixed time, otherwise we are penalized by kaggle with a Timeout Error

    T_max = configuration.timeout - 0.34 # for kaggle timeout reasons

    Cp_default = 1



    def play(board, column, mark, config):

        columns = config.columns

        rows = config.rows

        row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])

        board[column + (row * columns)] = mark # this basically throws the stone in that column



    def is_win(board, column, mark, config):

        columns = config.columns

        rows = config.rows

        inarow = config.inarow - 1

        row = min([r for r in range(rows) if board[column + (r * columns)] == mark])



        def count(offset_row, offset_column):

            for i in range(1, inarow + 1):

                r = row + offset_row * i

                c = column + offset_column * i

                if (

                    r < 0 or r >= rows or c < 0 or c >= columns

                    or board[c + (r*columns)] != mark

                ):

                    return i-1

                return inarow



        return (

            count(1,0) >= inarow # vertical win check

            or (count(0,1) + count(0,-1)) >= inarow # horizontal win check

            or (count(-1,-1) + count(1,1)) >= inarow # top left diagonal win check

            or (count(-1, 1) + count(1, -1)) >= inarow # top right diagonal win check

        )



    def is_tie(board):

        return not(any(mark==EMPTY for mark in board))



    def check_finish_and_score(board, column, mark, config):

        if is_win(board, column, mark, config):

            return (True, 1)

        if is_tie(board):

            return (True, 0.5)

        else:

            return (False, None)



    def uct_score(node_total_score, node_total_visits, parent_total_visits, Cp=Cp_default):

        # UCB1 calculation. UCB1 applied to trees. Explore - Exploit dillema

        if node_total_visits == 0:

            return math.inf # from the UCB1 formula

        return node_total_score / node_total_visits + Cp * math.sqrt(

          2 * math.log(parent_total_visits) / node_total_visits)



    def opponent_mark(mark): # mark indicates which player is active: player1 or player2

        return 3 - mark



    def opponent_score(score): # to backprop scores on the trere

        return 1 - score



    def random_action(board, config): # returns random legal action from available columns

        return random.choice([c for c in range(config.columns) if board[c]==EMPTY])



    def default_policy_simulation(board, mark, config):

        # Run a random play sim. Start state is assumed to be non-terminal.

        # Returns score of the game for the player with the given mark.

        original_mark = mark

        board = board.copy()

        column = random_action(board, config)



        play(board, column, mark, config)

        is_finish, score = check_finish_and_score(board, column, mark, config)



        while not is_finish:

            mark = opponent_mark(mark)

            column = random_action(board, config)

            play(board, column, mark, config)

            is_finish, score = check_finish_and_score(board, column, mark, config)



        if mark == original_mark:

            return score

        return opponent_score(score)



    def find_action_taken_by_opponent(new_board, old_board, config):

        # Given a new board state and the old board state, find which move was made.

        # Used for recycling trees between moves.

        for i, piece in enumerate(new_board):

            if piece != old_board[i]:

                return i % config.columns

        return -1 # shouldn't get here!



    class State(): # A class to represent a Node in the game tree

        def __init__(self, board, mark, config, parent=None, is_terminal=False,

                    terminal_score=None, action_taken=None):

            self.board = board.copy()

            self.mark = mark

            self.config = config

            self.children = []

            self.parent = parent

            self.node_total_score = 0

            self.node_total_visits = 0

            self.available_moves = [c for c in range(config.columns) if board[c]==EMPTY]

            self.expandable_moves = self.available_moves.copy()

            self.is_terminal = is_terminal

            self.terminal_score = terminal_score

            self.action_taken = action_taken



        def is_expandable(self): # Checks if node has unexplored kids (i.e. the set of legal, unexplored moves is not empty)

            return (not self.is_terminal) and (len(self.expandable_moves)>0)



        def expand_and_simulate_child(self):

                # Expands a random move from legal unexplored moves and runs a sim of it

                # (Expansion + Sim. + Backprop) steps from MCTS algo

            column = random.choice(self.expandable_moves)

            child_board = self.board.copy()



            play(child_board, column, self.mark, self.config)

            is_terminal, terminal_score = check_finish_and_score(child_board, column, self.mark, self.config)



                # Expansion stage

            self.children.append(  State(child_board, opponent_mark(self.mark),

                    self.config, parent=self, is_terminal=is_terminal,

                    terminal_score=terminal_score, action_taken=column)  )



                # Simulation stage

            simulation_score = self.children[-1].simulate()



                # Backprop stage

            self.children[-1].backpropagate(simulation_score)



                # Set of legal, unexplored moves at state self is depleted of this one, with which we just worked with.

            self.expandable_moves.remove(column)



        def choose_strongest_child(self, Cp):

                # Chooses kid that maximizes UCB1 score (Selection stage in MCTS description)

            children_scores = [uct_score(kid.node_total_score, kid.node_total_visits,

                                self.node_total_visits, Cp) for kid in self.children]

            max_score = max(children_scores)

            best_kid_idx = children_scores.index(max_score)

            return self.children[best_kid_idx]



        def choose_play_child(self): # not used in any of other State class' methods because the Selection step is done based on the UCB1 scores of nodes.

                # Chooses kid with maximum total score

            children_scores = [kid.node_total_score for kid in self.children]

            max_score = max(children_scores)

            best_kid_idx = children_scores.index(max_score)

            return self.children[best_kid_idx]



        def tree_single_run(self):

                # Single iteration of the 4 stages of the MCTS algo

            if self.is_terminal:

                self.backpropagate(self.terminal_score)

                return

            if self.is_expandable():

                self.expand_and_simulate_child()

                return

            self.choose_strongest_child(Cp_default).tree_single_run()



        def simulate(self):

                # Runs a sim from the current state.

                # This methdod is used to sim a game after a move of the current player.

                # If a terminal state is reached, score belongs to player who made the move.

                # Else the score received from sim is the opponent's score and thus

                # the usage of the flipping function opponent_score()

            if self.is_terminal:

                return self.terminal_score

            return opponent_score(default_policy_simulation(self.board, self.mark, self.config))



        def backpropagate(self, simulation_score):

                # Backprops score and # of visits to parents

            self.node_total_score += simulation_score

            self.node_total_visits += 1

            if self.parent is not None:

                self.parent.backpropagate(opponent_score(simulation_score))



        def choose_child_via_action(self, action):

                # Choose kid given the action taken from the state.

                # Used for recycling the tree

            for kid in self.children:

                if kid.action_taken == action:

                    return kid

            return None



    board = observation.board

    mark = observation.mark



    # If current_state already exists, recycle it based on action taken by opponent

    try:

        current_state = current_state.choose_child_via_action(

                    find_action_taken_by_opponent(board, current_state(board), configuration))

        current_state.parent = None # Make current_state the root. Dereference parents and siblings



    except: # New game or error due to Kaggle. Shall not happen more than once.

        current_state = State(board, mark, configuration, parent=None,

                        is_terminal=False, terminal_score=None, action_taken=None)



    while time.time() - init_time <= T_max:

        current_state.tree_single_run()



    current_state = current_state.choose_play_child()

    # This whole, big fnct. MCTS_agent() returns the action which has to be taken from the current position

    # Mapping (obs,config) ---> action_taken is made at each call of the MCTS_agent() fnct.

    return current_state.action_taken

env.reset()

try:

    del current_state

except:

    pass



env.run([MCTS_agent, MCTS_agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
import inspect



submission_path = "submission.py"

        

def write_agent_to_file(function, file):

    with open(file, "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(MCTS_agent, submission_path)
# Note: Stdout replacement is a temporary workaround.

from kaggle_environments import make, evaluate, utils, agent

import sys

out = sys.stdout

try:

    submission = utils.read_file("/kaggle/working/submission.py")

    agent = agent.get_last_callable(submission)

finally:

    sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
