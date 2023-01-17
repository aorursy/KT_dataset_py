# Analysis of first algorithm : 

def sort_reviews(data):
    new_list = [] # 1 step
    for j in range(len(data)): # n steps
        new_list.append((data[j][0], data[j][11])) # 3n steps
    for j in range(1, len(new_list)):  # n steps
        current = new_list[j] # 2n steps
        k = j # n steps
        try: 
            while k > 0 and new_list[k - 1][1] > current[1]: # 4n steps
                new_list[k] = new_list[k - 1] # n^2 steps
                k -= 1 # 2n^2 steps
        except TypeError: 
            pass # n steps
        new_list[k] = current # 2n steps

    new_list.reverse() # 1 step
# Analysis of the second algorithm:

def most_listings_host(data):
    current_most = data[1][14] # 3 steps
    most_index = 0 # 1 step

    for i in range(1, len(data)): # n steps
        if data[i][14] > current_most: # 2n steps
            current_most = data[i][14] # 3n steps
            most_index = i # n steps

    return data[most_index] #n steps

import csv


# ['id', 'name', 'host_id', 'host_name', 'neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'room_type',
# 'price', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count',
# 'availability_365']

# Loading data into a list of lists, since csv file contains all strings only, converted ints and floats to their
# correct types before appending.

def read_file():
    with open('../input/assg1csv/AB_NYC_2019.csv') as csv_file:
        whole_file = csv.reader(csv_file, delimiter=",")
        raw_data = []
        for row in whole_file:
            newRow = []
            for each in row:
                if each.isdigit():
                    newRow.append(int(each))
                else:
                    try:
                        newRow.append(float(each))
                    except ValueError:
                        newRow.append(each)
            raw_data.append(newRow)

        data = raw_data[:int(len(raw_data) / 10)]

    return data


# The first algorithm will sort the given data by number of reviews. Note that it can be sorted by any integer-based
# column by changing index from number 11 to any index.


def sort_reviews(data):
    new_list = []
    for j in range(len(data)):
        new_list.append((data[j][0], data[j][11]))
    for j in range(1, len(new_list)):  # Insertion sort algorithm to sort the list by price
        current = new_list[j]
        k = j
        try:  # error checking : first nested array is the description of columns
            while k > 0 and new_list[k - 1][1] > current[1]:
                new_list[k] = new_list[k - 1]
                k -= 1
        except TypeError:
            pass
        new_list[k] = current

    new_list.reverse()  # Reversed the list, since original list was sorted in ascending order.

    print("Top 5 number of reviews can be found at the following indexes:")
    print("Index {} with the value {}".format(new_list[1][0], new_list[1][1]))
    print("Index {} with the value {}".format(new_list[2][0], new_list[2][1]))
    print("Index {} with the value {}".format(new_list[3][0], new_list[3][1]))
    print("Index {} with the value {}".format(new_list[4][0], new_list[4][1]))
    print("Index {} with the value {}".format(new_list[5][0], new_list[5][1]))


# This is the same algorithm as above - however, it returns a list of sorted listings by reviews.


def sorted_array_by_reviews(data):
    volatile_data = data  # create a temporary list so that original data array isn't interfered with.
    for i in range(len(volatile_data)):  # Insertion sort algorithm to sort the list by price
        current = volatile_data[i]
        j = i
        try:  # error checking : first nested array is the description of columns
            while j > 0 and volatile_data[j - 1][11] > current[11]:
                volatile_data[j] = volatile_data[j - 1]
                j -= 1
        except TypeError:
            pass
        volatile_data[j] = current

    volatile_data.reverse()  # Reversed the list, since original list was sorted in ascending order.

    return volatile_data


# The second algorithm will give the listing with the most listing per host. Like the previous algorithm, the category
#  can be any integer based column, changed by simply changing the index number 14 to any index with of integer type.


def most_listings_host(data):
    current_most = data[1][14]
    most_index = 0

    try:
        for i in range(1, len(data)):
            if data[i][14] > current_most:
                current_most = data[i][14]
                most_index = i
    except TypeError:
        pass

    return data[most_index]


def main():
    data = read_file()
    sort_reviews(data)
    sorted_array_by_reviews(data)
    print(most_listings_host(data))


if __name__ == '__main__':
    main()

