def nested_change(item, func):

    if isinstance(item, list):

        return [nested_change(x, func) for x in item]

    return func(item)
listA=[1,2,0.6,"asd",True,74]

ListB=nested_change(listA, str)

ListB