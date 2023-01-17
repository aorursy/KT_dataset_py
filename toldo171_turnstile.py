import numpy as np
def update_arrays(_persons, _indice, _updateTime, _next_index):

    #removing the first entry in each array

    _persons.pop(_indice)

    direction.pop(_indice)

    time.pop(_indice)

    #The other person, who couldn't take the turnstile, has its time value incremented

    if _updateTime == True:

        for i in range(_next_index):

            time[i] += 1
def getTimes(time, direction):



    wasUsedPreviousSecond = False

    wasUsedAsExit = False

    persons = [i for i in range(len(time))]

    array = np.zeros(len(time))

    

    for i in range (len(persons)):

        

        #checking if there is a queue at the turnstile. If not, updates the time array and boolean value

        if time[0] > i:

            temp_time0 = time[0]

            wasUsedPreviousSecond = False

            for j in range(len(time)):

                time[j] -= temp_time0 - i

          

        print('------------------------------------------')

        print('begin iteration   ', i)

        print('time =            ', time)

        print('direction =       ', direction)

        print('persons =         ', persons)

        print('array =           ', array)

        print('Used previously = ', wasUsedPreviousSecond)

        print('Used EXIT =       ', wasUsedAsExit)

        print('------------------------------------------')

        

        #-------------------------------------------------------------------------------

        #if there is only one element left, he can pass the turnstile

        if len(time) == 1:

            #person in first position on the array can go through

            array[i] = persons[0]

        #-------------------------------------------------------------------------------

        #if time[0] != time[1], person[0] can go through the turnstile

        elif time[0] != time[1]:

            #person in first position on the array can go through

            array[i] = persons[0]



            #Updating booleans

            wasUsedPreviousSecond = True

            if direction[0] == 0:

                wasUsedAsExit = False

            else:

                wasUsedAsExit = True

            update_arrays(_persons = persons, _indice = 0, _updateTime = False, _next_index = 1)

        #-------------------------------------------------------------------------------

        #else, at least 2 persons are waiting. If the turnstile has not been used the second before, exactly 2 persons are waiting.

        elif wasUsedPreviousSecond == False:

            #direction = 1 (EXIT) has priority. It's either the first or second element in the array, since noone has used the turnstile the second before

            if direction[0] == 1:

                array[i] = persons[0]

                wasUsedPreviousSecond = True

                wasUsedAsExit = True

                update_arrays(_persons = persons, _indice = 0, _updateTime = True, _next_index = 1)

            else:

                array[i] = persons[1]

                wasUsedPreviousSecond = True

                wasUsedAsExit = True

                update_arrays(_persons = persons, _indice = 1, _updateTime = True, _next_index = 1)

        #-------------------------------------------------------------------------------

        #else, there can be more than two persons waiting. The priority is EXIT is it was EXIT last second, else it is ENTER

        elif wasUsedPreviousSecond == True:

            #if last was ENTER/EXIT, we check that there are still persons who wants to ENTER/EXIT

            if wasUsedAsExit in direction:

                next_index = direction.index(wasUsedAsExit)

                #We check if the next EXIT/ENTER guy is waiting or still not there

                if time[next_index] == i:

                    array[i] = persons[next_index]

                    update_arrays(_persons = persons, _indice = next_index, _updateTime = True, _next_index = next_index)

                else:

                    array[i] = persons[0]

                    update_arrays(_persons = persons, _indice = 0, _updateTime = False, _next_index = 1)

                    wasUsedAsExit = not wasUsedAsExit #Switch boolean

            else:

                array[i] = persons[0]

                update_arrays(_persons = persons, _indice = 0, _updateTime = False, _next_index = 1)

                wasUsedAsExit = not wasUsedAsExit #Switch boolean



    return array
time = [0,0,2,4,4,5]

direction = [0,1,1,1,0,0]
getTimes(time, direction)
time = [0,0,1,1,2,2,3,3,10,20,20,21]

direction = [0,1,0,1,0,1,0,1,1,0,1,1]
getTimes(time, direction)