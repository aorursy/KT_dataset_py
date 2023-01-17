import sys

import collections



def get_day_of_week(idx):

    return (idx % 7) if (idx % 7) else 7



print("id,nextvisit")

sys.stdin.readline()

for line in sys.stdin:

    id, visits = line.split(", ")

    id = int(id)



    days = visits.split()



    if int(days[-1]) < 920:

        print("{}, {}".format(id, 0))

        continue



    counter = collections.Counter()

    for day in days:

        day = int(day)

        if day < 764:

            continue

        counter.update([get_day_of_week(day)])



    print("{}, {}".format(id, counter.most_common(1)[0][0]))