

class node():

    def __init__(self,data=None):

        self.data=data

        self.next=None

class slinkedlist():

    def __init__(self):

        self.head=None

        self.tail=None

    def add(self,item):

        if not isinstance(item, node):

            item=node(item)

        if self.head is None:

            self.head=item

        else:

            self.tail.next=item

        self.tail=item

    def out(self):

        curr=self.head

        while curr is not None:

            print (curr.data)

            curr=curr.next

        return curr

    def search(self ,val):

        curr = self.head

        while curr is not None:

            

            if curr.data == val:

                print ("val exists")

            curr = curr.next

l_o=slinkedlist()

N_o1=node(12)

N_o2=node("hi")

for i in [N_o1,N_o2]:

    l_o.add(i)

    l_o.out()

    l_o.search(212)

            
class node():

    def __init__(self,data):

        self.prev=None

        self.data=data

        self.next=None

class doublelinkedlist():

    def __init__(self):

        self.head=None

    

    def add(self,value):

        item=node(value)

        item.next=self.head

        if self.head is not None:

            self.head.prev=item

        self.head=item

    def out(self,node):

        while node is not None:

            print(node.data)

            last=node

            node=node.next

    
dl=doublelinkedlist()

dl.add(4)

dl.add(7)

dl.out(dl.head)
class stack():

    def __init__(self):

        self.stac=[]

    def add(self,val):

        if val not in self.stac:

            self.stac.append(val)

        else:

            pass

    def out(self):

        print(self.stac[-1])

        

    def remove(self):

        if len(self.stac) == 0:

            print ("empty")

        else:

            return self.stac.pop()

        
s=stack()

s.add(5)

s.add(6)

s.out()

s.add("int")

s.out()

print(s.remove())
class queue():

    def __init__(self):

        self.que=[]

    def add(self,value):

        if value not in self.que:

            self.que.insert(0,value)

        else:

            pass

    def remove (self):

        if len(self.que) > 0:

            return self.que.pop()

        else:

            print("empty")

        

        
q=queue()

q.add(6)

q.add(7)

print (q.remove())

print(q.out())

class node:

    def __init__(self,data):

        self.right=None

        self.data=data

        self.left=None

    def insert (self,data):

        if self.data:

            if data < self.data:

                if self.left is None:

                    self.left=node(data)

                else:

                    self.left.insert(data)

            elif data > self.data:

                if self.right is None:

                    self.right=node(data)

                else:

                    self.right.insert(data)

        else:

            self.data=data

    def out(self):

        if self.left:

            self.left.out()

        print (self.data)

        if self.right:

            self.right.out()

x=node(5)

x.insert(7)

x.insert(89)

x.insert(2)

x.out()