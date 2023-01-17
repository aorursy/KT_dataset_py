import matplotlib.pyplot as plt
labels = ['Frogs', 'Hogs', 'Dogs']
sizes = [15, 30, 45]
#colors = ['yellowgreen', 'mediumpurple', 'lightskyblue', 'lightcoral'] 
#explode = (0, 0.1, 0, 0)    # proportion with which to offset each wedge

plt.pie(sizes,              # data
        #explode=explode,    # offset parameters 
        labels=labels,      # slice labels
        #colors=colors,      # array of colours
        autopct='%1.1f%%',  # print the values inside the wedges
        shadow=True,        # enable shadow
        #startangle=70       # starting angle
        )
plt.axis('equal')

plt.show()