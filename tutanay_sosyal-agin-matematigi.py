import networkx as nx

import random as rd 

import matplotlib.pyplot as plt
G_symmetric = nx.Graph()

G_symmetric.add_edge('Gamze Ceylan','Merve Elif Saraç')

G_symmetric.add_edge('Gamze Ceylan','Hasan Koçak')

G_symmetric.add_edge('Gamze Ceylan','Fatma Bozyiğit')

G_symmetric.add_edge('Gamze Ceylan','Otomata')

G_symmetric.add_edge('Merve Elif Saraç','Hasan Koçak')

G_symmetric.add_edge('Merve Elif Saraç','Fatma Bozyiğit')

G_symmetric.add_edge('Merve Elif Saraç','Otomata')

G_symmetric.add_edge('Otomata','Hasan Koçak')

nx.draw_networkx(G_symmetric)
G_asymmetric = nx.DiGraph()

G_asymmetric.add_edge('A','B')

G_asymmetric.add_edge('A','D')

G_asymmetric.add_edge('C','A')

G_asymmetric.add_edge('D','E')

nx.spring_layout(G_asymmetric)

nx.draw_networkx(G_asymmetric)
G_weighted = nx.Graph()

G_weighted.add_edge('Gamze Ceylan','Merve Elif Saraç', weight=25)

G_weighted.add_edge('Gamze Ceylan','Hasan Koçak', weight=8)

G_weighted.add_edge('Gamze Ceylan','Fatma Bozyiğit', weight=11)

G_weighted.add_edge('Gamze Ceylan','Otomata', weight=1)

G_weighted.add_edge('Merve Elif Saraç','Hasan Koçak', weight=4)

G_weighted.add_edge('Merve Elif Saraç','Fatma Bozyiğit',weight=7)

G_weighted.add_edge('Merve Elif Saraç','Otomata', weight=1)

G_weighted.add_edge('Otomata','Hasan Koçak',weight=1)

nx.spring_layout(G_weighted)

nx.draw_networkx(G_weighted)
#En kısa Yolu hesaplamak için kendimize şehirler oluşturduk

G3 = nx.Graph()

cities = ['Aydın', 'Ankara', 'Istanbul', 'Bursa', 'Muğla', 'İzmir', 'Balıkesir']

for city in cities:

    G3.add_node(city)

nx.draw(G3, with_labels=1)

plt.show()


import random as rd

while(G3.number_of_edges() == 10):

    #random.choise Oluşturduğumuz  diziden boş olmayan rastgele bir öğe döndürür.



    cities[0] = rd.choice(G3.nodes())

    cities[1] = rd.choice(G3.nodes())

i=0;

indis=[0,1,2,3,4,5,6]

for i in indis:

    #Burada şehir sayısı uzunluğunda iki adet döngü oluşturduk 

    for j in indis:

        wt = rd.randint(20, 2000)

        if cities[i]!= cities[j] and G3.has_edge(cities[i],cities[j]) == 0:

            G3.add_edge(cities[i], cities[j], weight = wt)      

#Dizideki her şehri birbiriyle karşılaştırdık eğer iki şehir birbirine eşit değilse ve aralarında bir bağ yoksa iki şehir arasında random atadığımız değer uzunluğunda bir bağ oluşturduk.



        j=j+1

    i=i+1;

pos = nx.circular_layout(G3)

# G( NetworkX grafiği veya düğüm listesi ) -

nx.draw(G3, pos, with_labels=1)

nx.draw_networkx_edge_labels(G3, pos)

plt.show()

#Oluşturduğumuz grafiği çizdirdik
print (nx.dijkstra_path(G3,'Istanbul','Balıkesir'))

#İki şehir arası en kısa mesafeyi gitmemiz için sırayla gitmemiz gereken şehirleri gösteriyor.

print(nx.dijkstra_path_length(G3,'Istanbul','Balıkesir'))

#İki şehir arası en kısa mesafenin uzunluğunu gösteriyor.
#Bu örnek en kısa yolu bulur ve bu örnek TSP örneğine benzer.Gezgin Satıcı Örneği.Otomatayla olan ilişkisi
