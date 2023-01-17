#Q1.



def find_second_to_last(list):

    if list.head is None or list.head.next is None: 

        return None

    current = list.head

    next = list.head.next

    while next.next is not None: 

        current = next

        next = next.next

    return current
#Q2.



def count_nodes(list):

    if list.current is None:

        return 0

    count = 1

    original = list.current

    current = original

    while current.next != original:

        count += 1

    return count
#Q3.



def same_circular_list(a, b):

    current = a

    while current.next is not b:

        if current.next is a:

            return False

        current = current.next

    return True
#Q4.



#from positional_list import PositionalList



def list_to_positional_list(list_):

    pos_list = PositionalList()

    for element in list_:

        pos_list.add_last(element)

    return pos_list
#Q5.



def max(self):

    return max(number for number in self)