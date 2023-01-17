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
from kaggle_environments import make

from kaggle_environments.envs.halite.helpers import *



import json

import sys
%%writefile submission.py

import numpy as np

import pandas as pd

from random import choice

from kaggle_environments import make

from kaggle_environments.envs.halite.helpers import *

from copy import deepcopy



def get_distance(pos1,pos2,context):

    size = context['board'].configuration.size

    distance_east = min(abs(pos2[0]-pos1[0]),abs(pos2[0]-pos1[0]+size),abs(pos2[0]-pos1[0]-size))

    distance_north = min(abs(pos2[1]-pos1[1]),abs(pos2[1]-pos1[1]+size),abs(pos2[1]-pos1[1]-size))

    return distance_east + distance_north



def cell_safety_check(cell,context):

    board = context['board']

    me = context['board'].current_player

    

    return 'safe'

    

def move_to_verification(ship_position,heading,context):

    board = context['board']

    current_player = context['board'].current_player

    current_cell = board.cells[ship_position]

    current_ship = [ship for ship in board.ships.values() if ship.position == ship_position][0]

    

    if heading == 'north':

        heading_cell = current_cell.north

    elif heading == 'south':

        heading_cell = current_cell.south

    elif heading == 'east':

        heading_cell = current_cell.east

    elif heading == 'west':

        heading_cell = current_cell.west

    else:

        heading_cell = current_cell

        

    if (heading_cell.ship is None) and (heading_cell.shipyard is None):

        return 'empty'

    elif (heading_cell.ship is not None) and (heading_cell.ship.player_id == current_player.id):

        return 'friendly_ship'

    elif (heading_cell.ship is not None) and (heading_cell.ship.player_id != current_player.id):

        if heading_cell.ship.halite <= current_ship.halite:

            return 'danger_enemy_ship'

        else:

            return 'target_enemy_ship'

    elif (heading_cell.ship is None) and (heading_cell.shipyard.player_id == current_player.id):

        return 'friendly_shipyard'

    elif (heading_cell.ship is None) and (heading_cell.shipyard.player_id != current_player.id):

        return 'enemy_shipyard'

    else:

        pass

    return 'empty'

    

def move_to(ship_position,target_position,context):

    

    east_move = target_position[0] - ship_position[0]

    if abs(east_move) > context['board'].configuration.size / 2 + 1:

        east_move = -1 * east_move

        

    north_move = target_position[1] - ship_position[1]

    if abs(north_move) > context['board'].configuration.size / 2 + 1:

        north_move = -1 * north_move

    

    #首先对周边4个cell进行占用情况评估

    ver_list = [move_to_verification(ship_position,heading,context) for heading in ['east','south','west','north']]

    prob_list = np.array([(ver in ['target_enemy_ship','friendly_shipyard','empty','enemy_shipyard']) * 1 for ver in ver_list])

    if north_move > 0 and east_move == 0 and prob_list.sum() > 0:

        prob_list[3] = prob_list[3] * 20

        prob_list = prob_list / prob_list.sum()

        return np.random.choice([ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.NORTH],p=prob_list)

    elif north_move > 0 and east_move > 0 and prob_list.sum() > 0:

        prob_list[3] = prob_list[3] * 20

        prob_list[0] = prob_list[0] * 20

        prob_list = prob_list / prob_list.sum()

        return np.random.choice([ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.NORTH],p=prob_list)

    elif north_move > 0 and east_move < 0 and prob_list.sum() > 0:

        prob_list[3] = prob_list[3] * 20

        prob_list[1] = prob_list[1] * 20

        prob_list = prob_list / prob_list.sum()

        return np.random.choice([ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.NORTH],p=prob_list)

    elif north_move < 0 and east_move == 0 and prob_list.sum() > 0:

        prob_list[1] = prob_list[1] * 20

        prob_list = prob_list / prob_list.sum()

        return np.random.choice([ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.NORTH],p=prob_list)

    elif north_move < 0 and east_move > 0 and prob_list.sum() > 0:

        prob_list[1] = prob_list[1] * 20

        prob_list[0] = prob_list[0] * 20

        prob_list = prob_list / prob_list.sum()

        return np.random.choice([ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.NORTH],p=prob_list)

    elif north_move < 0 and east_move < 0 and prob_list.sum() > 0:

        prob_list[1] = prob_list[1] * 20

        prob_list[2] = prob_list[2] * 20

        prob_list = prob_list / prob_list.sum()

        return np.random.choice([ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.NORTH],p=prob_list)

    elif north_move == 0 and east_move == 0 and prob_list.sum() > 0:

        return None

    elif north_move == 0 and east_move > 0 and prob_list.sum() > 0:

        prob_list[0] = prob_list[0] * 20

        prob_list = prob_list / prob_list.sum()

        return np.random.choice([ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.NORTH],p=prob_list)

    elif north_move == 0 and east_move < 0 and prob_list.sum() > 0:

        prob_list[2] = prob_list[2] * 20

        prob_list = prob_list / prob_list.sum()

        return np.random.choice([ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST,ShipAction.NORTH],p=prob_list)

    else:

        return None

    return None





def shipyard_convert_decision(context):

    board = context['board']

    current_player = context['board'].current_player

    shipyard_building = False

    #如果当前没有船坞，设法建造一个

    if (len(current_player.shipyards)) == 0:

        #找到库存最多的飞船,将其转化为船坞

        cargos = np.array([ship.halite for ship in current_player.ships])

        for ship in current_player.ships:

            if ship.halite == cargos.max():

                ship.next_action = ShipAction.CONVERT

                shipyard_building = True

                return context

            

    #如果船只数/船坞>=6，那么造第二船坞

    num_shipyard = len(current_player.shipyards)

    num_ship = len(current_player.ships)

    if (num_ship >= num_shipyard * 6) and (shipyard_building == False) and (current_player.halite >= 2000):

        #找到距离和负载合适的飞船,将其转化为船坞

        base_ship_list = []

        for ship in current_player.ships:

            distance_to_shipyards = [get_distance(ship.position,shipyard.position,context) for shipyard in current_player.shipyards]

            if np.min(distance_to_shipyards) >= 4:

                base_ship_list.append(ship)

        if len(base_ship_list) >= 1:

            ship_cargos = [ship.halite for ship in base_ship_list]

            for ship in base_ship_list:

                if ship.halite == np.max(ship_cargos) and shipyard_building == False:

                    ship.next_action = ShipAction.CONVERT

                    shipyard_building = True

    return context



def find_nearest_halite(ship,frange,context):  

    board = context['board']

    nearby_cells = [cell for cell in board.cells.values() if get_distance(ship.position,cell.position,context) <= frange]

    nearby_empty_cells = [cell for cell in nearby_cells if cell.ship is None]

    nearby_empty_cell_halites = [cell.halite for cell in nearby_empty_cells]

    highest_halite_empty_cells = [cell for cell in nearby_empty_cells if cell.halite == np.max(nearby_empty_cell_halites)]

    target_cell = choice(highest_halite_empty_cells)

    return target_cell



def ship_mining_decision(context):

    board = context['board']

    current_player = context['board'].current_player

    for n,ship in enumerate(current_player.ships):

        current_cell = board.cells[ship.position[0],ship.position[1]]

        mining_cell = find_nearest_halite(ship,2,context)

        #如果飞船载货量大于1500，那么就地转为船坞

        if (ship.halite >= 1500) and (ship.next_action is None):

            ship.next_action = ShipAction.CONVERT

        #如果飞船载货量大于300，那么就回最近的船坞卸货

        elif ship.halite >= 300 and ship.next_action!=ShipAction.CONVERT:

            pos = ship.position

            shipyards = current_player.shipyards

            shipyard_distances = np.array([get_distance(pos,shipyard.position,context) for shipyard in shipyards])

            for shipyard in shipyards:

                if get_distance(pos,shipyard.position,context) == np.min(shipyard_distances):

                    ship.next_action = move_to(ship.position,shipyard.position,context)



        #如果竞赛接近结束，那么就回最近的船坞卸货

        elif (board.configuration.episode_steps - board.step <= 12) and (ship.next_action!=ShipAction.CONVERT):

            pos = ship.position

            shipyards = current_player.shipyards

            shipyard_distances = np.array([get_distance(pos,shipyard.position,context) for shipyard in shipyards])

            for shipyard in shipyards:

                if get_distance(pos,shipyard.position,context) == np.min(shipyard_distances):

                    ship.next_action = move_to(ship.position,shipyard.position,context)

        

        #如果附近有一个富矿比当前halite含量大150以上，那么就转移

        elif (mining_cell.halite*0.25 - current_cell.halite*0.25*2 > 50) and (ship.next_action!=ShipAction.CONVERT):

            ship.next_action = move_to(ship.position,mining_cell.position,context)

        elif current_cell.halite > 200 and (ship.next_action!=ShipAction.CONVERT):

            ship.next_action = None

        elif ship.next_action!=ShipAction.CONVERT:

            p = [0.2,0.2,0.2,0.2]

            p[n%4] += 0.2

            ship.next_action = np.random.choice([ShipAction.NORTH,ShipAction.SOUTH,ShipAction.WEST,ShipAction.EAST],p=p)

        else:

            pass



    return context



def get_ship_next_cell(ship,context):

    board = context['board']

    if ship.next_action == ShipAction.NORTH:

        return ship.cell.north

    elif ship.next_action == ShipAction.SOUTH:

        return ship.cell.south

    elif ship.next_action == ShipAction.EAST:

        return ship.cell.east

    elif ship.next_action == ShipAction.WEST:

        return ship.cell.west

    elif ship.next_action is None:

        return ship.cell

    else:

        return None

    

def self_collision_check(context):

    board = context['board']

    current_player = context['board'].current_player

    next_cells = []

    idle_ships = [ship for ship in current_player.ships if ship.next_action is None]

    moving_ships = [ship for ship in current_player.ships if ship.next_action is not None]

    next_cells = [get_ship_next_cell(ship,context) for ship in idle_ships]

    for ship in moving_ships:

        next_cell = get_ship_next_cell(ship,context)

        if next_cell in next_cells:

            ship.next_action = None

            next_cells.append(ship.cell)

        else:

            next_cells.append(next_cell)

    return context



def shipyard_collision_check(shipyard,context):

    board = context['board']

    current_player = context['board'].current_player

    next_cells = []

    if len(current_player.ships) == 0:

        return 'shipyard available'

    for ship in current_player.ships:

        next_cells.append(get_ship_next_cell(ship,context))

    if shipyard.cell in next_cells:

        return 'ship incoming'

    else:

        return 'shipyard available'

    return 'shipyard available'



def short_range_attack(context):

    board = context['board']

    for myship in board.current_player.ships:

        next_cells = [cell for cell in board.cells.values() if get_distance(myship.position,cell.position,context)==1]

        next_ships = [cell.ship for cell in next_cells if cell.ship is not None]

        enemy_ships = [ship for ship in next_ships if ship.player_id != board.current_player.id]

        target_ships = [ship for ship in enemy_ships if myship.halite < ship.halite]

        if len(target_ships) >= 1:

            origional_action = myship.next_action

            attack_move = move_to(myship.position,target_ships[0].position,context)

            myship.next_action = np.random.choice([origional_action,attack_move])

    return context



def shipyard_attack(context,attack_range,attack_player_id = None,attack_ship_number = 1):

    board = context['board']

    current_player = context['board'].current_player

    attackships = [ship for ship in current_player.ships if ship.halite <= 80]

    attacking_ship = 0

    if attack_player_id is None:

        enemy_shipyards = [shipyard for shipyard in board.shipyards.values() if shipyard.player_id != current_player.id]

    else:

        enemy_shipyards = [shipyard for shipyard in board.shipyards.values() if shipyard.player_id == attack_player_id]

    

    for ship in attackships:

        #选取在攻击范围内的敌方船坞

        enemy_shipyards_in_range = [shipyard for shipyard in enemy_shipyards if get_distance(ship.position,shipyard.position,context)<=attack_range]

        #计算所有攻击范围内敌方船坞的距离

        target_ranges = [get_distance(ship.position,shipyard.position,context) for shipyard in enemy_shipyards_in_range]

        #选取距离最近的敌方船坞

        target_shipyards = [shipyard for shipyard in enemy_shipyards_in_range if get_distance(ship.position,shipyard.position,context)==np.min(target_ranges)]

        #如果合适目标>=1，下达攻击指令

        if len(target_shipyards) >= 1 and attacking_ship < attack_ship_number:

            ship.next_action = move_to(ship.position,target_shipyards[0].position,context)

            attacking_ship += 1

    return context



def ship_building(context):

    board = context['board']

    current_player = context['board'].current_player

    num_shipyard = len(current_player.shipyards)

    num_ship = len(current_player.ships)

    ship_building = 'not allow'

    if num_shipyard == 0:

        return context

    

    #如果当前ship很少，那么补充ship

    if num_ship <= 1:

        ship_building = 'allow'

    elif board.step <= 7 and current_player.halite > 500:

        ship_building = 'allow'

    #如果储备资源太少，那么暂停造船

    elif current_player.halite <= 1000:

        ship_building = 'not allow'

    #如果船队规模较大，而且接近终局，那么停止造船

    elif num_ship >= 10 and (board.configuration.episode_steps - board.step <= 15):

        ship_building = 'not allow'

    #如果船队规模小于船坞承载能力，且有足够储备资金和剩余回合数，那么开始造船

    elif (num_ship <= num_shipyard * 6) and (board.configuration.episode_steps - board.step >= 20) and (current_player.halite >= 1500):

        ship_building = 'allow'

    else:

        return context

    

    for shipyard in current_player.shipyards:

        if ship_building == 'allow':

            pass

        else:

            break

        ship_incoming_check = shipyard_collision_check(shipyard,context)

        if (ship_incoming_check=='shipyard available') and (ship_building == 'allow'):

            shipyard.next_action = ShipyardAction.SPAWN

            ship_building = 'ongoing'

    return context



def get_target_player_id(context):

    board = context['board']

    current_player = context['board'].current_player

    df_player_rank = pd.DataFrame()

    df_player_rank['player_id'] = [player.id for player in board.players.values()]

    df_player_rank['player_halite'] = [player.halite for player in board.players.values()]

    df_player_rank['player_cargo'] = [len(player.shipyards) for player in board.players.values()]

    df_player_rank['player_ships'] = [len(player.ships) for player in board.players.values()]

    df_player_rank['player_shipyards'] = [len(player.shipyards) for player in board.players.values()]

    df_player_rank.sort_values(by = 'player_halite', ascending=False, inplace=True)

    df_player_rank.reset_index(inplace=True,drop=True)



    if df_player_rank.loc[0,'player_id'] == current_player.id:

        attack_player_id = df_player_rank.loc[1,'player_id']

    else:

        attack_player_id = df_player_rank.loc[0,'player_id']

    return attack_player_id



def ship_retreat(context):

    board = context['board']

    me = context['board'].current_player

    for ship in me.ships:

        ver_list = [move_to_verification(ship.position,heading,context) for heading in ['east','south','west','north']]

        #如果身边出现敌方攻击飞船，那么就往老家跑

        if 'danger_enemy_ship' in var_list:

            shipyard_distances = np.array([get_distance(ship.position,shipyard.position,context) for shipyard in me.shipyards])

            for shipyard in shipyards:

                if get_distance(ship.position,shipyard.position,context) == np.min(shipyard_distances):

                    ship.next_action = move_to(ship.position,shipyard.position,context)

                    if ship.next_action is None and ship.halite >= 200 and me.halite >= 500:

                        ship.next_action = ShipAction.CONVERT

        



def agent(obs,config):

    board = Board(obs,config)

    size = board.configuration.size

    max_steps = board.configuration.episode_steps

    current_player = board.current_player

    context = {}

    context['board'] = board

    context['current_player'] = board.current_player

    context = shipyard_convert_decision(context)

    context = ship_mining_decision(context)

    context = short_range_attack(context)

    

    attacks = round(len(current_player.ships)/5)

    if board.step <= max_steps / 4:

        attacks = round(len(current_player.ships)/5)

        context = shipyard_attack(context,attack_range = size/4, attack_player_id = None, attack_ship_number = attacks)

    elif board.step >= max_steps / 4 * 3:

        attack_player_id = get_target_player_id(context)

        attacks = round(len(current_player.ships)/3)

        context = shipyard_attack(context,attack_range = size/2, attack_player_id = attack_player_id, attack_ship_number = attacks)

    else:

        attacks = round(len(current_player.ships)/5)

        context = shipyard_attack(context,attack_range = size/3, attack_player_id = None, attack_ship_number = attacks)

        

    context = self_collision_check(context)

    context = self_collision_check(context)

    context = self_collision_check(context)

    context = ship_building(context)

    

    return context['board'].current_player.next_actions
BOARD_SIZE = 13

AGENT_COUNT = 2

env = make('halite',debug=True,configuration = {'size':BOARD_SIZE,'agent_count':2})
env.run(["/kaggle/working/submission.py", "/kaggle/working/submission.py"])

print("EXCELLENT SUBMISSION!" if env.toJSON()["statuses"] == ["DONE", "DONE"] else "MAYBE BAD SUBMISSION?")



# Play as the first agent against default "shortest" agent.

env.run(["/kaggle/working/submission.py", "random"])

env.render(mode="ipython", width=750, height=550)
BOARD_SIZE = 8

AGENT_COUNT = 2

env = make('halite',debug=True,configuration = {'size':BOARD_SIZE,'agent_count':3})

env.reset(AGENT_COUNT)

env.render(mode="ipython", width=200, height=160)

state = env.state[0]
board = Board(state.observation, env.configuration)

context = {}

context['board'] = board

current_player = board.current_player

cells = [ship.cell for ship in board.ships.values()]

acell = cells[0]

cells.append(acell)
acell in set(cells)
board.configuration.episode_steps