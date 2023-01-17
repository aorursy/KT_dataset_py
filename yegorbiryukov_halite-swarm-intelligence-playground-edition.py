# T# 1. Enable Internet in the Kernel (Settings side pane)



# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 

# !curl -X PURGE https://pypi.org/simple/kaggle-environments



# Halite environment was defined in v0.2.1

!pip install 'kaggle-environments>=0.2.1'
from kaggle_environments import evaluate, make



env = make("halite", debug=True)

env.render()
%%writefile submission.py

# for Debug previous line (%%writefile submission.py) should be commented out, uncomment to write submission.py



#FUNCTIONS###################################################

def set_map_and_average_halite(s_env):

    """

        set average amount of halite per halite source

        and map as two dimensional array of objects and set amounts of halite in each cell

    """

    s_env["map"] = []

    halite_sources_amount = 0

    halite_total_amount = 0

    for x in range(conf.size):

        s_env["map"].append([])

        for y in range(conf.size):

            s_env["map"][x].append({

                # value will be ID of owner

                "shipyard": None,

                # value will be ID of owner

                "ship": None,

                # value will be amount of halite

                "ship_cargo": None,

                # amount of halite

                "halite": s_env["obs"].halite[conf.size * y + x]

            })

            if s_env["map"][x][y]["halite"] > 0:

                halite_total_amount += s_env["map"][x][y]["halite"]

                halite_sources_amount += 1

    s_env["average_halite"] = halite_total_amount / halite_sources_amount



def update_map(s_env):

    """

        update locations of ships and shipyards on the map,

        get lists of coords of Swarm's units,

        get targets for "torpedoes"

    """

    global torpedo_targets

    torpedo_targets = []

    # arrays of (x, y) coords

    s_env["swarm_shipyards_coords"] = []

    s_env["swarm_ships_coords"] = []

    # place on the map locations of units of every player

    for player in range(len(s_env["obs"].players)):

        # set torpedo targets

        target_index = 0

        for i in range(len(torpedo_targets)):

            while (target_index < len(torpedo_targets) and

                    s_env["obs"].players[player][0] < torpedo_targets[target_index]["halite"]):

                target_index += 1

        torpedo_targets.insert(target_index, {

            "player": player,

            "shipyards": [],

            "halite": s_env["obs"].players[player][0]

        })

        # place on the map locations of every shipyard of the player

        shipyards = list(s_env["obs"].players[player][1].values())

        for shipyard in shipyards:

            x = shipyard % conf.size

            y = shipyard // conf.size

            # place shipyard on the map

            s_env["map"][x][y]["shipyard"] = player

            torpedo_targets[target_index]["shipyards"].append({"x": x, "y": y})

            if player == s_env["obs"].player:

                s_env["swarm_shipyards_coords"].append((x, y))

        # place on the map locations of every ship of the player

        ships = list(s_env["obs"].players[player][2].values())

        for ship in ships:

            x = ship[0] % conf.size

            y = ship[0] // conf.size

            # place ship on the map

            s_env["map"][x][y]["ship"] = player

            s_env["map"][x][y]["ship_cargo"] = ship[1]

            if player == s_env["obs"].player:

                s_env["swarm_ships_coords"].append((x, y))



def get_c(c):

    """ get coordinate, considering donut type of the map """

    return c % conf.size



def clear(x, y, player, game_map):

    """ check if cell is safe to move in """

    # if there is no shipyard, or there is player's shipyard

    # and there is no ship

    if ((game_map[x][y]["shipyard"] == player or game_map[x][y]["shipyard"] == None) and

            game_map[x][y]["ship"] == None):

        return True

    return False



def get_closest_coords(x_initial, y_initial, s_env, coords_list):

    """ get from coords_list x and y closest to x_initial and y_initial """

    closest_coords_index = 0

    min_distance = None

    for i in range(len(coords_list)):

        to_x = coords_list[i]["x"]

        to_y = coords_list[i]["y"]

        # choose x route

        if x_initial > to_x:

            normal_route_x = x_initial - to_x

            donut_route_x = conf.size - x_initial + to_x

        else:

            normal_route_x = to_x - x_initial

            donut_route_x = conf.size - to_x + x_initial

        # x distance to shipyard

        x_dist = donut_route_x if donut_route_x < normal_route_x else normal_route_x

        # choose y route

        if y_initial > to_y:

            normal_route_y = y_initial - to_y

            donut_route_y = conf.size - y_initial + to_y

        else:

            normal_route_y = to_y - y_initial

            donut_route_y = conf.size - to_y + y_initial

        # y distance to shipyard

        y_dist = donut_route_y if donut_route_y < normal_route_y else normal_route_y

        dist = x_dist + y_dist

        if min_distance == None or dist < min_distance:

            min_distance = dist

            closest_coords_index = i

    return coords_list[closest_coords_index]



def move_to_cell(to_x, to_y, x_initial, y_initial, actions, s_env, ship_index):

    """ move ship to cell with to_x and to_y coords, if possible """

    ship_id = s_env["ships_keys"][ship_index]

    ship_cargo = s_env["ships_values"][ship_index][1]

    

    # choose x route

    if x_initial > to_x:

        normal_route_x = x_initial - to_x

        donut_route_x = conf.size - x_initial + to_x

        # x direction to shipyard

        x_dir = "WEST"

    else:

        normal_route_x = to_x - x_initial

        donut_route_x = conf.size - to_x + x_initial

        x_dir = "EAST"

    # x distance to shipyard

    x_dist = donut_route_x if donut_route_x < normal_route_x else normal_route_x

    

    # choose y route

    if y_initial > to_y:

        normal_route_y = y_initial - to_y

        donut_route_y = conf.size - y_initial + to_y

        # y direction to shipyard

        y_dir = "NORTH"

    else:

        normal_route_y = to_y - y_initial

        donut_route_y = conf.size - to_y + y_initial

        y_dir = "SOUTH"

    # y distance to shipyard

    y_dist = donut_route_y if donut_route_y < normal_route_y else normal_route_y



    # get possible directions

    for direction in directions_list:

        if direction["direction"] == x_dir:

            x = direction["x"](x_initial)

        elif direction["direction"] == y_dir:

            y = direction["y"](y_initial)



    # choose direction

    # if this ship is not a "torpedo"

    if ships_data[s_env["ships_keys"][ship_index]]["target_coords"] == None:

        if x_dist > y_dist:

            if (clear(x, y_initial, s_env["obs"].player, s_env["map"]) and

                    not hostile_ship_near(x, y_initial, s_env["obs"].player, s_env["map"], ship_cargo)):

                s_env["map"][x_initial][y_initial]["ship"] = None

                s_env["map"][x][y_initial]["ship"] = s_env["obs"].player

                actions[ship_id] = x_dir

                return True

        else:

            if (clear(x_initial, y, s_env["obs"].player, s_env["map"]) and

                    not hostile_ship_near(x_initial, y, s_env["obs"].player, s_env["map"], ship_cargo)):

                s_env["map"][x_initial][y_initial]["ship"] = None

                s_env["map"][x_initial][y]["ship"] = s_env["obs"].player

                actions[ship_id] = y_dir

                return True

    # if this ship is a "torpedo"

    else:

        if x_dist > y_dist:

            if s_env["map"][x][y_initial]["ship"] == None:

                s_env["map"][x_initial][y_initial]["ship"] = None

                s_env["map"][x][y_initial]["ship"] = s_env["obs"].player

                actions[ship_id] = x_dir

                return True

        else:

            if s_env["map"][x_initial][y]["ship"] == None:

                s_env["map"][x_initial][y_initial]["ship"] = None

                s_env["map"][x_initial][y]["ship"] = s_env["obs"].player

                actions[ship_id] = y_dir

                return True

    return False



def return_to_shipyard(x_initial, y_initial, actions, s_env, ship_index):

    """ return to shipyard's coords """

    ship_id = s_env["ships_keys"][ship_index]

    ship_cargo = s_env["ships_values"][ship_index][1]

    # if ship is currently at shipyard's coords

    if x_initial == shipyard_coords["x"] and y_initial == shipyard_coords["y"]:

        # if there is no shipyard at shipyard's coords

        if (s_env["map"][x_initial][y_initial]["shipyard"] == None and

                (ship_cargo + s_env["swarm_halite"]) >= conf.convertCost):

            actions[ship_id] = "CONVERT"

            s_env["map"][x_initial][y_initial]["ship"] = None

            return True

        # if ship is going to move out from shipyard's coords

        else:

            global movement_tactics_index

            ships_data[ship_id]["moves_done"] = 0

            ships_data[ship_id]["ship_max_moves"] = 1

            ships_data[ship_id]["directions"] = movement_tactics[movement_tactics_index]["directions"]

            ships_data[ship_id]["directions_index"] = 0

            movement_tactics_index += 1

            if movement_tactics_index >= movement_tactics_amount:

                movement_tactics_index = 0

    else:

        # if ship has to return to shipyard's coords

        if ship_cargo >= s_env["return_threshold"]:

            return move_to_cell(

                shipyard_coords["x"], shipyard_coords["y"], x_initial, y_initial, actions, s_env, ship_index)

    return False



def move_ship(x_initial, y_initial, actions, s_env, ship_index):

    """ move the ship according to first acceptable tactic """

    if go_for_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index):

        return

    standard_patrol(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)



def go_for_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):

    """ ship will go to safe cell with enough halite, if it is found """

    # if current cell has enough halite

    if (s_env["map"][x_initial][y_initial]["halite"] > s_env["low_amount_of_halite"] and

            not hostile_ship_near(x_initial, y_initial, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):

        most_halite = s_env["map"][x_initial][y_initial]["halite"]

    else:

        # biggest amount of halite among scanned cells

        most_halite = s_env["low_amount_of_halite"]

    direction = None

    for d in range(len(directions_list)):

        x = directions_list[d]["x"](x_initial)

        y = directions_list[d]["y"](y_initial)

        # if cell is safe to move in

        if (clear(x, y, s_env["obs"].player, s_env["map"]) and

                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):

            # if current cell has more than biggest amount of halite

            if s_env["map"][x][y]["halite"] > most_halite:

                most_halite = s_env["map"][x][y]["halite"]

                direction = directions_list[d]["direction"]

                direction_x = x

                direction_y = y

    # if cell is safe to move in and has substantial amount of halite

    if most_halite > s_env["low_amount_of_halite"] and direction != None:

        actions[ship_id] = direction

        s_env["map"][x_initial][y_initial]["ship"] = None

        s_env["map"][direction_x][direction_y]["ship"] = s_env["obs"].player

        return True

    # if current cell has biggest amount of halite

    elif most_halite == s_env["map"][x_initial][y_initial]["halite"]:

        return True

    return False



def standard_patrol(x_initial, y_initial, ship_id, actions, s_env, ship_index):

    """ 

        ship will move in expanding circles clockwise or counterclockwise

        until reaching maximum radius, then radius will be minimal again

    """

    directions = ships_data[ship_id]["directions"]

    # set index of direction

    i = ships_data[ship_id]["directions_index"]

    for j in range(len(directions)):

        x = directions[i]["x"](x_initial)

        y = directions[i]["y"](y_initial)

        # if cell is ok to move in

        if (clear(x, y, s_env["obs"].player, s_env["map"]) and

                (s_env["map"][x][y]["shipyard"] == s_env["obs"].player or

                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1]))):

            ships_data[ship_id]["moves_done"] += 1

            # apply changes to game_map, to avoid collisions of player's ships next turn

            s_env["map"][x_initial][y_initial]["ship"] = None

            s_env["map"][x][y]["ship"] = s_env["obs"].player

            # if it was last move in this direction

            if ships_data[ship_id]["moves_done"] >= ships_data[ship_id]["ship_max_moves"]:

                ships_data[ship_id]["moves_done"] = 0

                ships_data[ship_id]["directions_index"] += 1

                # if it is last direction in a list

                if ships_data[ship_id]["directions_index"] >= len(directions):

                    ships_data[ship_id]["directions_index"] = 0

                    ships_data[ship_id]["ship_max_moves"] += 1

                    # if ship_max_moves reached maximum radius expansion

                    if ships_data[ship_id]["ship_max_moves"] > max_moves_amount:

                        ships_data[ship_id]["ship_max_moves"] = 3

            actions[ship_id] = directions[i]["direction"]

            break

        else:

            # loop through directions

            i += 1

            if i >= len(directions):

                i = 0



def get_directions(i0, i1, i2, i3):

    """ get list of directions in a certain sequence """

    return [directions_list[i0], directions_list[i1], directions_list[i2], directions_list[i3]]



def hostile_ship_near(x, y, player, m, cargo):

    """ check if hostile ship is in one move away from game_map[x][y] and has less or equal halite """

    # m = game map

    n = get_c(y - 1)

    e = get_c(x + 1)

    s = get_c(y + 1)

    w = get_c(x - 1)

    if (

            (m[x][n]["ship"] != player and m[x][n]["ship"] != None and m[x][n]["ship_cargo"] <= cargo) or

            (m[x][s]["ship"] != player and m[x][s]["ship"] != None and m[x][s]["ship_cargo"] <= cargo) or

            (m[e][y]["ship"] != player and m[e][y]["ship"] != None and m[e][y]["ship_cargo"] <= cargo) or

            (m[w][y]["ship"] != player and m[w][y]["ship"] != None and m[w][y]["ship_cargo"] <= cargo)

        ):

        return True

    return False



def spawn_ship(actions, s_env, ships_amount, i):

    """ spawn ship, if possible """

    if s_env["swarm_halite"] >= conf.spawnCost and ships_amount < s_env["ships_max_amount"]:

        x = s_env["swarm_shipyards_coords"][i][0]

        y = s_env["swarm_shipyards_coords"][i][1]

        # if there is currently no ship at shipyard

        if clear(x, y, s_env["obs"].player, s_env["map"]):

            s_env["swarm_halite"] -= conf.spawnCost

            actions[s_env["shipyards_keys"][i]] = "SPAWN"

            s_env["map"][x][y]["ship"] = s_env["obs"].player

            ships_amount += 1

        return True, ships_amount

    return False, ships_amount



def this_is_new_ship(s_env, i):

    """ add this ship to ships_data """

    global movement_tactics_index

    ships_data[s_env["ships_keys"][i]] = {

        "moves_done": 0,

        "ship_max_moves": 3,

        "directions": movement_tactics[movement_tactics_index]["directions"],

        # coords of target if ship is launched as a torpedo

        "target_coords": None,

        "directions_index": 0

    }

    movement_tactics_index += 1

    if movement_tactics_index >= movement_tactics_amount:

        movement_tactics_index = 0

        

def send_as_torpedo(x_initial, y_initial, actions, s_env, i):

    """

        send this ship as a torpedo to closest shipyard of the player that is one position ahead of the Swarm

        or one position below, if Swarm is the leader

    """

    if ships_data[s_env["ships_keys"][i]]["target_coords"] == None:

        # minimal ships amount to launch ship as a torpedo

        if len(s_env["ships_keys"]) > 10 and s_env["torpedoes_amount"] < torpedoes_max_amount:

            for j in range(len(torpedo_targets)):

                if torpedo_targets[j]["player"] == s_env["obs"].player:

                    target_index = j - 1 if j > 0 else j + 1

                    if len(torpedo_targets[target_index]["shipyards"]) > 0:

                        coords = get_closest_coords(x_initial, y_initial, s_env, torpedo_targets[target_index]["shipyards"])

                        ships_data[s_env["ships_keys"][i]]["target_coords"] = coords

                    else:

                        return False

        else:

            return False

    x = ships_data[s_env["ships_keys"][i]]["target_coords"]["x"]

    y = ships_data[s_env["ships_keys"][i]]["target_coords"]["y"]

    if move_to_cell(x, y, x_initial, y_initial, actions, s_env, i):

        return True

    else:

        return False



def proceed_as_torpedo(x_initial, y_initial, actions, s_env, i):

    """ proceed to the target, if ship is a torpedo """

    if ships_data[s_env["ships_keys"][i]]["target_coords"] != None:

        x = ships_data[s_env["ships_keys"][i]]["target_coords"]["x"]

        y = ships_data[s_env["ships_keys"][i]]["target_coords"]["y"]

        # if target coords reached and are empty

        if x == x_initial and y == y_initial:

            ships_data[s_env["ships_keys"][i]]["target_coords"] = None

        elif move_to_cell(x, y, x_initial, y_initial, actions, s_env, i):

            return True

    return False



def this_is_last_step(x, y, actions, s_env, i):

    """ actions of ship, if it is last step """

    if s_env["obs"].step == (conf.episodeSteps - 2) and s_env["ships_values"][i][1] >= conf.convertCost:

        actions[s_env["ships_keys"][i]] = "CONVERT"

        s_env["map"][x][y]["ship"] = None

        return True

    return False



def to_spawn_or_not_to_spawn(s_env):

    """ to spawn, or not to spawn, that is the question """

    # get ships_max_amount to decide whether to spawn new ships or not

    ships_max_amount = s_env["average_halite"] // 5

    # if ships_max_amount is less than minimal allowed amount of ships in the Swarm

    if ships_max_amount < ships_min_amount:

        ships_max_amount = ships_min_amount

    return ships_max_amount



def define_some_globals(observation, configuration):

    """ define some of the global variables """

    global conf

    global max_moves_amount

    global globals_not_defined

    conf = configuration

    max_moves_amount = 7

    # set coords of the shipyard

    start_ship_coords = list(observation.players[observation.player][2].values())[0][0]

    shipyard_coords["x"] = start_ship_coords % conf.size

    shipyard_coords["y"] = start_ship_coords // conf.size

    globals_not_defined = False



def adapt_environment(observation, configuration, s_env):

    """ adapt environment for the Swarm """

    s_env["obs"] = observation

    if globals_not_defined:

        define_some_globals(observation, configuration)

    set_map_and_average_halite(s_env)

    s_env["low_amount_of_halite"] = 4 if s_env["average_halite"] < 4 else s_env["average_halite"]

    s_env["return_threshold"] = s_env["average_halite"] * 2

    s_env["swarm_halite"] = s_env["obs"].players[s_env["obs"].player][0]

    update_map(s_env)

    s_env["ships_keys"] = list(s_env["obs"].players[s_env["obs"].player][2].keys())

    s_env["ships_values"] = list(s_env["obs"].players[s_env["obs"].player][2].values())

    s_env["shipyards_keys"] = list(s_env["obs"].players[s_env["obs"].player][1].keys())

    s_env["ships_max_amount"] = to_spawn_or_not_to_spawn(s_env)

    s_env["torpedoes_amount"] = 0

    

def actions_of_ships(actions, s_env):

    """ actions of every ship of the Swarm """

    # calculate amount of "torpedo" ships

    for i in range(len(s_env["ships_keys"])):

        if (s_env["ships_keys"][i] in ships_data and

                ships_data[s_env["ships_keys"][i]]["target_coords"] != None):

            s_env["torpedoes_amount"] += 1

    for i in range(len(s_env["swarm_ships_coords"])):

        x = s_env["swarm_ships_coords"][i][0]

        y = s_env["swarm_ships_coords"][i][1]

        # if this is a new ship

        if s_env["ships_keys"][i] not in ships_data:

            this_is_new_ship(s_env, i)

            # send this ship as a "torpedo", if possible

            if send_as_torpedo(x, y, actions, s_env, i):

                s_env["torpedoes_amount"] += 1

                continue

        # proceed to the target, if ship is a torpedo

        if proceed_as_torpedo(x, y, actions, s_env, i):

            continue

        # if it is last step

        if this_is_last_step(x, y, actions, s_env, i):

            continue

        # if ship has to return to shipyard

        if return_to_shipyard(x, y, actions, s_env, i):

            continue

        move_ship(x, y, actions, s_env, i)



def actions_of_shipyards(actions, s_env):

    """ actions of every shipyard of the Swarm """

    ships_amount = len(s_env["ships_keys"])

    # spawn ships from every shipyard, if possible

    for i in range(len(s_env["shipyards_keys"])):

        # spawn a ship

        ok, ships_amount = spawn_ship(actions, s_env, ships_amount, i)

        if not ok:

            break





#GLOBAL_VARIABLES#############################################

conf = None

# max amount of moves in one direction before turning

max_moves_amount = None

# object with ship ids and their data

ships_data = {}

# max amount of "torpedo" ships

torpedoes_max_amount = 1

# list of torpedo targets

torpedo_targets = []

# initial movement_tactics index

movement_tactics_index = 0

# minimum amount of ships that should be in the Swarm at any time

ships_min_amount = 3

# coords of the shipyard

shipyard_coords = {"x": None, "y": None}

# not all global variables are defined

globals_not_defined = True



# list of directions

directions_list = [

    {

        "direction": "NORTH",

        "x": lambda z: z,

        "y": lambda z: get_c(z - 1)

    },

    {

        "direction": "EAST",

        "x": lambda z: get_c(z + 1),

        "y": lambda z: z

    },

    {

        "direction": "SOUTH",

        "x": lambda z: z,

        "y": lambda z: get_c(z + 1)

    },

    {

        "direction": "WEST",

        "x": lambda z: get_c(z - 1),

        "y": lambda z: z

    }

]



# list of movement tactics

movement_tactics = [

    # N -> E -> S -> W

    {"directions": get_directions(0, 1, 2, 3)},

    # S -> E -> N -> W

    {"directions": get_directions(2, 1, 0, 3)},

    # N -> W -> S -> E

    {"directions": get_directions(0, 3, 2, 1)},

    # S -> W -> N -> E

    {"directions": get_directions(2, 3, 0, 1)},

    # E -> N -> W -> S

    {"directions": get_directions(1, 0, 3, 2)},

    # W -> S -> E -> N

    {"directions": get_directions(3, 2, 1, 0)},

    # E -> S -> W -> N

    {"directions": get_directions(1, 2, 3, 0)},

    # W -> N -> E -> S

    {"directions": get_directions(3, 0, 1, 2)}

]

movement_tactics_amount = len(movement_tactics)





#THE_SWARM####################################################

def swarm_agent(observation, configuration):

    """ RELEASE THE SWARM!!! """

    # s_env -> swarm environment

    s_env = {}

    actions = {}

    adapt_environment(observation, configuration, s_env)

    actions_of_ships(actions, s_env)

    actions_of_shipyards(actions, s_env)

    return actions
if "swarm_agent" in globals():

    # reset variables

    ships_data = {}

    movement_tactics_index = 0



    # Play as first position against random agent.

    trainer = env.train([None, "random"])



    observation = trainer.reset()



    while not env.done:

        my_action = swarm_agent(observation, env.configuration)

        print("Step: {0}, My Action: {1}".format(observation.step, my_action))

        observation, reward, done, info = trainer.step(my_action)

        # env.render(mode="ipython", width=100, height=90, header=False, controls=False)

    env.render()
def mean_reward(rewards):

    wins = 0

    ties = 0

    loses = 0

    for r in rewards:

        r0 = 0 if r[0] is None else r[0]

        r1 = 0 if r[1] is None else r[1]

        if r0 > r1:

            wins += 1

        elif r1 > r0:

            loses += 1

        else:

            ties += 1

    return f'wins={wins/len(rewards)}, ties={ties/len(rewards)}, loses={loses/len(rewards)}'



# Run multiple episodes to estimate its performance.

# Setup agentExec as LOCAL to run in memory (runs faster) without process isolation.

print("Swarm Agent vs Random Agent:", mean_reward(evaluate(

    "halite",

    ["submission.py", "random"],

    num_episodes=10, configuration={"agentExec": "LOCAL"}

)))
env.run(["submission.py", "submission.py"])

#env.run(["submission.py", "random"])

env.render(mode="ipython", width=800, height=600)