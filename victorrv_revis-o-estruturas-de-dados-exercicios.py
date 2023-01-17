class Node:
    def __init__(self):
        self.val = None
        self.next = None
        self.last = None
    
    def nodeAppend(self,val):
        node = self
        previous = None
        while node.val != None:
            previous = node
            node = node.next
            
        node.val = val
        node.last = previous
        
        nodenew = Node()
        node.next = nodenew
        nodenew.last = node
        
    def count(self, val):
        node = self
        n = 0
        while node.val != None:
            if node.val == val:
                n += 1
            node = node.next
        print(n)
        
    def traverse(self):
        node = self
        while node.val != None:
            print(node.val)
            node = node.next
    
    def traverseBack(self):
        node = self
        while node.val != None:
            node = node.next
        node = node.last
        while node.last != None:
            print(node.val)
            node = node.last
        print(node.val)
            
    def removeDuplicates(self):
        elements = []
        previous = None
        node = self
        while node.val != None:
            if node.val in elements:
                previous.next = node.next
                node.next = None
                node = previous
            else:
                elements.append(node.val)
            previous = node
            node = node.next
            
    def get_kth_element(self,k):
        node = self
        i = 0
        while node.val != None and i < k:
            node = node.next
            i += 1
        if node != None and i == k:
            print(node.val)
        else:
            print('not found')
    
    def remove_kth_element(self,k):
        node = self
        i = 0
        newnode = self
        while node.val != None and i < k:
            node = node.next
            i += 1
        if node != None and i == k:
            node.val = None
            if k != 0:
                node.last.next = node.next
            else:
                newnode = node.next
            node.next.last = node.last
            node.next = None
            node.last = None
        else:
            print('not found')
        return newnode

node = Node()
node.nodeAppend(1)
node.nodeAppend(2)
node.nodeAppend(1)
node.nodeAppend(2)
node.nodeAppend(3)
node.traverse()
node.traverseBack()
node.count(2)
node = node.remove_kth_element(0)
node.traverse()
print('Navegando entre os nós:')
node.traverse()
print()
print('Removendo os duplicados...')
node.removeDuplicates()
print()
print('Verificando se foram removidos:')
node.traverse()
print()
print('Encontrando o elemento da lista baseado num índice:')
node.get_kth_element(0)
print()