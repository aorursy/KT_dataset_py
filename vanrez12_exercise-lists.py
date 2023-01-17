from learntools.core import binder; binder.bind(globals())

from learntools.python.ex4 import *

print('Setup complete.')
# Return the second element of the given list. If the list has no second

# element, return None



def select_second(L):

    if len(L)<2:

        return None

    return (L[1])



q1.check()
q1.hint()

q1.solution()
#Given a list of teams, where each team is a list of names, return the 2nd player (captain)

#from the last listed team

def losing_team_captain(teams):

    return teams[-1][1]



q2.check()
#q2.hint()

#q2.solution()
racers=[]

def purple_shell(racers):

    #Given a list of racers, set the first place racer (at the front of the list) to last

    #place and vice versa.

    last=racers[-1]

    racers[-1]=racers[0]

    racers[0]=last

    return 

    

r = ["Mario", "Bowser", "Luigi"]

purple_shell(r)

r



q3.check()
#q3.hint()

#q3.solution()
a = [1, 2, 3]

b = [1, [2, 3]]

c = []

d = [1, 2, 3][1:]



# Put your predictions in the list below. Lengths should contain 4 numbers, the

# first being the length of a, the second being the length of b and so on.

lengths = [3,2,0,2]



q4.check()
# line below provides some explanation

q4.solution()
arrivals=[]

def fashionably_late(arrivals, name):

    if len(arrivals)%2>0:

        if ((arrivals.index(name))>(len(arrivals)/2)) and ((arrivals.index(name))!=(len(arrivals)-1)):

            return True

        return False

    else :

        if ((arrivals.index(name))>=((len(arrivals)/2))) and ((arrivals.index(name))!=(len(arrivals)-1)):

            return True

        return False
party_attendees = ['Adela', 'Fleda', 'Owen', 'May', 'Mona', 'Gilbert', 'Ford']



print(fashionably_late(party_attendees, 'May'))

print(fashionably_late(party_attendees, 'Mona'))

print(fashionably_late(party_attendees, 'Gilbert'))

print(fashionably_late(party_attendees, 'Ford'))

party_attendees_2 = ['Paul', 'John', 'Ringo', 'George']

print(fashionably_late(party_attendees_2, 'Paul'))

print(fashionably_late(party_attendees_2, 'John'))

print(fashionably_late(party_attendees_2, 'Ringo'))

print(fashionably_late(party_attendees_2, 'George'))
q5.check()
#q5.hint()

#q5.solution()