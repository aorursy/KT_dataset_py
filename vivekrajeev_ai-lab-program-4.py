def H_n(state, target):

    dist = 0

    for i in state:

        d1, d2 = state.index(i), target.index(i)

        x1, y1 = d1 % 3, d1 // 3

        x2, y2 = d2 % 3, d2 // 3

        dist += abs(x1-x2) + abs(y1-y2)

    return dist
def print_grid(src):

    state = src.copy()

    state[state.index(-1)] = ' '

    print(

        f"""

{state[0]} {state[1]} {state[2]}

{state[3]} {state[4]} {state[5]}

{state[6]} {state[7]} {state[8]}

        """

    )

    
def h_n(state, target):

    dist = 0

    for i in state:

        d1, d2 = state.index(i), target.index(i)

        x1, y1 = d1 % 3, d1 // 3

        x2, y2 = d2 % 3, d2 // 3

        dist += abs(x1-x2) + abs(y1-y2)

    return dist
def astar(src, target):

    states = [src]

    g = 0

    visited_states = set()

    while len(states):

        print(f"Level: {g}")

        moves = []

        for state in states:

            visited_states.add(tuple(state))

            print_grid(state)

            if state == target:

                print("Success")

                return

            moves += [move for move in possible_moves(state, visited_states) if move not in moves]

        costs = [g + h_n(move, target) for move in moves]

        states = [moves[i] for i in range(len(moves)) if costs[i] == min(costs)]

        g += 1

    print("Fail")
def possible_moves(state, visited_states):# Add inputs if more are required

    # Find index of empty spot and assign it to b

    b = state.index(-1);

    

    #'d' for down, 'u' for up, 'r' for right, 'l' for left - directions array

    d = []

                                    

    #Add all possible direction into directions array - Hint using if statements

    if b - 3 in range(9): 

        d.append('u')

    if b not in [0,3,6]: 

        d.append('l')

    if b not in [2,5,8]: 

        d.append('r')

    if b + 3 in range(9): 

        d.append('d')

    

    # If direction is possible then add state to move

    pos_moves = []

    

    # for all possible directions find the state if that move is played

    ### Jump to gen function to generate all possible moves in the given directions

    for move in d:

        pos_moves.append(gen(state, move, b))

    

    # return all possible moves only if the move not in visited_states

    return [move for move in pos_moves if tuple(move) not in visited_states]

def gen(state, m, b):

    # m(move) is direction to slide, b(blank) is index of empty spot

    # create a copy of current state to test the move

    temp = state.copy()                              

    

    # if move is to slide empty spot to the left and so on

    if m == 'u': 

        temp[b-3] , temp[b] = temp[b], temp[b-3]

    if m == 'l': 

        temp[b-1] , temp[b] = temp[b], temp[b-1]

    if m == 'r': 

        temp[b+1] , temp[b] = temp[b], temp[b+1]

    if m == 'd': 

        temp[b+3] , temp[b] = temp[b], temp[b+3]   

    

    # return new state with tested move to later check if "src == target"

    return temp
#Test 1

src = [1,2,3,-1,4,5,6,7,8]

target = [1,2,3,4,5,-1,6,7,8]         

       





astar(src, target) 
# Test 2

src = [1,2,3,-1,4,5,6,7,8] 

target=[1,2,3,6,4,5,-1,7,8]







astar(src, target)
# Test 3

src = [1,2,3,7,4,5,6,-1,8] 

target=[1,2,3,6,4,5,-1,7,8]







astar(src, target)