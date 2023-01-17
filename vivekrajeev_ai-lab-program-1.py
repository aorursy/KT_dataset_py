# Create a 3x3 tic tac toe board of "" strings for each value

board = [' '] * 9
# Create a function to display your board

def display_board(board):

    print('   |   |')

    print(' ' + board[7] + ' | ' + board[8] + ' | ' + board[9])

    print('   |   |')

    print('-----------')

    print('   |   |')

    print(' ' + board[4] + ' | ' + board[5] + ' | ' + board[6])

    print('   |   |')

    print('-----------')

    print('   |   |')

    print(' ' + board[1] + ' | ' + board[2] + ' | ' + board[3])

    print('   |   |')



test_board = ['#','X','O','X','O','X','O','X','O','X']    

display_board(test_board)
#Create a function to check if anyone won, Use marks "X" or "O"

def check_win(mark, board):

     return ((board[7] == mark and board[8] == mark and board[9] == mark) or 

    (board[4] == mark and board[5] == mark and board[6] == mark) or 

    (board[1] == mark and board[2] == mark and board[3] == mark) or 

    (board[7] == mark and board[4] == mark and board[1] == mark) or

    (board[8] == mark and board[5] == mark and board[2] == mark) or 

    (board[9] == mark and board[6] == mark and board[3] == mark) or 

    (board[7] == mark and board[5] == mark and board[3] == mark) or 

    (board[9] == mark and board[5] == mark and board[1] == mark)) 

check_win('X', test_board)
# Create a function to check its a Draw

def check_draw(board):

        return ' ' not in board
# Create a Function that makes a copy of the board

def board_copy(board):

    dupeBoard = []



    for i in board:

        dupeBoard.append(i)



    return dupeBoard
#Immediate move checker

def test_win_move(move, player_mark, board):

    bCopy = board_copy(board)

    bCopy[move] = player_mark

    return check_win(player_mark, bCopy)

#Strategy if others fail

def win_strategy(board):

    if board[5] == '':

        return 4

    for i in [1,3,7,9]:

        if board[i] == ' ':

            return i

    for i in [2,4,6,8]:

        if board[i] == ' ':

            return i

        
# Agents move

def get_agent_move(board):

    # Return agent move with your strategy

    move = self.get_agent_move()

    self.board[move] = self.mark

    self.display_board()
# Assemble the game

def tictactoe():

    ### Note you need to recreate your board again here if you wish to play the game more than once

    Playing = True

    while Playing:

            self.board = [" " for i in range(9)]

            # Choose Marks

            print('X or O')

            self.player_mark = input()

            self.mark = 'X' if self.player_mark == "O" else "O"



            #Choose start

            print("Want to go first [y,n]")

            flag = '1' if input() == 'y' else '0'



            Ingame = True

            while Ingame:



                if flag == '1':

                    self.player_moves()

                    mark = self.player_mark

                else:

                    self.agent_moves()

                    mark = self.mark

                

                #Check if someone won

                if self.check_win(mark, self.board):

                    Ingame = False

                    if mark == self.player_mark:

                        print("Player wins!!!")

                    else:

                        print("Agent wins")

                    break

                

                # Check if game Draws

                if self.check_draw():

                    print('Draw!!!')

                    break

                

                #Switch to comp or person moves

                flag = '0' if flag == '1' else '1'

            

            # Another game 

            print("Another game [y,n]?")

            if input() == 'n':

                Playing = False

            print('Thank you for playing!')



            b = TicTacToe_board()

            b.tictactoe()

    
# Play!!!

tictactoe(board)
class Tic_Tac_Toe:

    def __init__(self):

        pass