# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
CONFIG = {

    'coin_in_to_send_balloon': {

        'help': 'The number of Coin In used to send a new balloon',

        'default': 200,

        'value': 200},

    'spins_to_send_existing_balloon': {

         'help': 'If the user already got a balloon and it was not popped, how many spins to wait before sending again',

         'default': 10,

         'value' : 10},

    'min_eligible_bet': {

         'help': 'The minimal amount of coins the user needs to be in order to get progress towards the next balloon',

         'default':50,

         'value' : 50},

    'balloon_tickets_bk':{

         'help': 'How many tickets to put on the high reward balloon',

         'default':100,

         'value' : 100},

    'balloon_tickets_regular':{

         'help': 'How many tickets to put on the low reward balloon',

         'default':200,

         'value' : 200}

}





import plotly.graph_objects as go

import random

import logging



APP_NAME = 'WinZoneTicketsBalloon'

DEBUG = True

NUM_USERS = 100

MIN_BALANCE = 100

MAX_BALANCE = 1000

STEPS_BALANCE = 100

MIN_BET = 100

MAX_BET = 200

STEPS_BET = 20

ODDS_OF_USER_POPPING_BALLOON = 0.5





class MyLogic:

    class Game:

        current_balloon = None

        spins_since_balloon_is_in_queue = 0

        total_coin_in = 0

        balloon_list = []

        total_tickets = 0

        total_spins = 0

    game = dict()

    debug = DEBUG



    def get_param_from_config(self, param):

        config_param = CONFIG[param]

        if config_param['value'] is None:

            return config_param['default']

        return config_param['value']





    coin_in_to_send_balloon = None

    spins_to_send_existing_balloon = None

    min_eligible_bet = None

    balloon_tickets_bk = None

    balloon_tickets_regular = None



    def runme(self, user_id, bet):

        # Parsing arguments to variables

        self.coin_in_to_send_balloon = self.get_param_from_config('coin_in_to_send_balloon')

        self.spins_to_send_existing_balloon = self.get_param_from_config('spins_to_send_existing_balloon')

        self.min_eligible_bet = self.get_param_from_config('min_eligible_bet')

        self.balloon_tickets_bk = self.get_param_from_config('balloon_tickets_bk')

        self.balloon_tickets_regular = self.get_param_from_config('balloon_tickets_regular')



        if user_id not in self.game:

            self.game[user_id] = self.Game()



        self.game[user_id].total_spins += 1



        # If user already has an active Balloon

        if self.game[user_id].current_balloon is not None:

            self.game[user_id].spins_since_balloon_is_in_queue += 1

            # See if it's time to resend the balloon

            if self.game[user_id].spins_since_balloon_is_in_queue >= self.spins_to_send_existing_balloon:

                self.game[user_id].spins_since_balloon_is_in_queue = 0

                self.send_balloon(self.game, self.game.current_balloon)

            return



        # Get User Bet

        current_bet = bet

        if current_bet is None:

          raise Exception("no total bet on spin")





        # Make sure user makes the minimum bet required

        if current_bet < self.min_eligible_bet:

          logging.debug("Not minimum bet")

          return



        # Update Total Coin In

        self.game[user_id].total_coin_in += current_bet



        # Check to see if balloon needs to be sent

        if self.game[user_id].total_coin_in >= self.coin_in_to_send_balloon:

          new_balloon = self.get_new_balloon(self.game[user_id])  # Create Balloon

          self.game[user_id].current_balloon = new_balloon

          self.send_balloon(new_balloon)  # Send balloon to the client

          self.game[user_id].balloon_list.append(new_balloon)  # Add balloon to log

          self.game[user_id].total_coin_in = 0

          self.game[user_id].spins_since_balloon_is_in_queue = 0



    def get_new_balloon(self, game):

      balloon = {}

      if self.is_balloon_bk(game):  # Condition for bk balloon

          balloon['tickets'] = self.balloon_tickets_bk  # High Prize Balloon

      else:

          balloon['tickets'] = self.balloon_tickets_regular  # First Balloon

      return balloon



    def is_balloon_bk(self, game):

      return False  # The condition of checking a user's balance to see if it's worth sending the highest balloon



    def send_balloon(self, balloon):

      return  # Code to send balloon data to client



    def pop_balloon(self, user_id):

        if user_id in self.game and self.game[user_id].current_balloon is not None:

            self.game[user_id].total_tickets += self.game[user_id].current_balloon['tickets']

            self.game[user_id].current_balloon = None



    def get_data(self):

        return self.game



    # add = payload.get('add', 0)

    #return 456+add





#class WinZoneFreePlay(LsBase):



#    def __init__(self):

#        super().__init__()

#        #self.my_field = {'value': 123}



#    def getData(self):

#        self.log.debug(F"Entering {APP_NAME} init")



#    # User Spins:

#    def run(self):

#        self.log.debug(F"Entering {APP_NAME} run")

#        self.my_field['value'] = MyLogic().runme(self.request['updateRequest']['data']['totalBet'])





#    def writeBack(self):

#        self.log.debug(F"Entering {APP_NAME} writeBack")

#        return self.my_field





# if __name__ == "__main__":

#    main()



class User:

    id = 0

    balance = 0



    def __init__(self, user_id):

        self.id = user_id

        self.balance = random.randrange(MIN_BALANCE, MAX_BALANCE, STEPS_BALANCE)





output = None





def main():

    logic = MyLogic()

    for i in range(1, NUM_USERS):

        user = User(i)

        while user.balance > 0:

            bet = random.randrange(MIN_BET, MAX_BET, STEPS_BET)

            if bet <= user.balance:

                user.balance -= bet

                logic.runme(user.id, bet)

            else:

                user.balance = 0

        if random.uniform(0, 1) <= ODDS_OF_USER_POPPING_BALLOON:

            logic.pop_balloon(user.id)

    output = logic.get_data()

    #MyLogic().runme(self.request['updateRequest']['data']['totalBet'])

    users = []

    tickets = []

    balloons = []

    total_spins = []

    for user in output:

        users.append(user)

        tickets.append(output[user].total_tickets)

        balloons.append(len(output[user].balloon_list))

        total_spins.append(output[user].total_spins)



    fig = go.Figure(data=[go.Table(header=dict(values=['User #', 'Total Tickets', 'Balloon #', 'Total Spins']),

                     cells=dict(values=[users, tickets, balloons, total_spins]))

                         ])

    fig.show()



main()


