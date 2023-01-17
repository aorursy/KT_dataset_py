from scipy.spatial import distance

#Generally we have 3 types of distance formulas: euclidan, manhattan, hamming distance. 

#The easiest way to understand the difference is to create functions for them instead of using .euclidean(),  .cityblock()  .hamming()





def euclidean_distance(pt1, pt2):

  distance = 0

  for i in range(len(pt1)):

    distance += (pt1[i] - pt2[i]) ** 2

  return distance ** 0.5



def manhattan_distance(pt1, pt2):

  distance = 0

  for i in range(len(pt1)):

    distance += abs(pt1[i] - pt2[i])

  return distance



def hamming_distance(pt1, pt2):

  distance = 0

  for i in range(len(pt1)):

    if pt1[i] != pt2[i]:

      distance += 1

  return distance



#Euc. distance and cityblock (manhattan) distance is very easy and common, but for me humming distance is most interesting

#as you can use it to compare lists, for example of words.



print(euclidean_distance([1, 2], [4, 0]))

print(manhattan_distance([1, 2], [4, 0]))

print(hamming_distance(['a', 'b', 'c'], ['c', 'b', 'a']))
