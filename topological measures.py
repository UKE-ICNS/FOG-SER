import matplotlib
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#create directional graph using adjecency matrix a

#matrix from arnaud with all points plus 2 included
A = np.array([[0,1,0,0,0,0,0,0,0,1,1,1], [1,0,1,1,1,0,0,0,0,1,0,0], [1,1,0,1,1,0,0,0,0,0,0,0], 
              [0,1,1,0,1,1,1,1,0,1,1,1], [0,-1,-1,-1,0,0,0,0,-1,0,-1,-1], [0,1,0,1,1,0,1,1,1,0,1,1], 
              [0,-1,0,-1,-1,-1,0,-1,-1,0,-1,-1], [0,0,0,0,-1,-1,-1,-1,-1,-1,-1,0], [0,0,0,0,-1,0,-1,-1,-1,0,-1,0],
              [0,1,0,1,0,1,0,1,1,1,1,0],[0,0,0,0,-1,1,-1,-1,-1,0,0,0],[0,0,0,0,0,0,0,0,1,1,0,0]]) #directed FOG network 
              #(LC,PRF,CNF,PPN,SNr,STN,GPi,GPe,Str,Ctx,SNc,Th) 

A_pd = np.array([[0,1,0,0,0,0,0,0,0,1,1,1], [1,0,1,1,1,0,0,0,0,1,0,0], [1,1,0,1,1,0,0,0,0,0,0,0], 
              [0,1,1,0,1,1,1,1,0,1,1,1], [0,-1,-1,-1,0,0,0,0,-1,0,-1,-1], [0,1,0,1,1,0,1,1,1,0,1,1], 
              [0,-1,0,-1,-1,-1,0,-1,-1,0,-1,-1], [0,0,0,0,-1,-1,-1,-1,-1,-1,-1,0], [0,0,0,0,-1,0,-1,-1,-1,0,-1,0],
              [0,1,0,1,0,1,0,1,1,1,1,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1,0,0]]) #directed FOG network 
              #(LC,PRF,CNF,PPN,SNr,STN,GPi,GPe,Str,Ctx,SNc,Th) 

G = nx.from_numpy_matrix(A,create_using=nx.DiGraph)
G_pd = nx.from_numpy_matrix(A_pd,create_using=nx.DiGraph)

#label graph
labeldict = {}
labeldict[0] = "LC"
labeldict[1] = "PRF"
labeldict[2] = "CNF"
labeldict[3] = "PPN"
labeldict[4] = "SNr"
labeldict[5] = "STN"
labeldict[6] = "GPi"
labeldict[7] = "GPe"
labeldict[8] = "Str"
labeldict[9] = "Ctx"
labeldict[10] = "SNc"
labeldict[11] = "Th"

labels_ig = ['LC', 'PRF', 'CNF', 'PPN', 'SNr', 'STN', 'GPi', 'GPe', 'Str', 'Ctx', 'SNc','Th']

#%% plotting utility

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

weight_dict = nx.get_edge_attributes(G, 'weight')

fig = plt.figure(figsize=(15,12)) 
# And a data frame with characteristics for your nodes
color = ['#07b4b9ff', '#07b4b9ff', '#07b4b9ff', '#07b4b9ff', '#005753ff', '#005753ff',
        '#005753ff', '#005753ff', '#005753ff', '#005753ff', '#005753ff', '#005753ff']
pos = nx.circular_layout(G)
nodes = nx.draw_networkx_nodes(G, pos, alpha=1, node_color=color, node_size=6000)
edges = nx.draw_networkx_edges(G, pos,node_size=6000,edgelist = list(getKeysByValue(weight_dict,-1)),edge_color='#1e2f97ff',width=3.5, arrowsize=30)
edges = nx.draw_networkx_edges(G, pos,node_size=6000,edgelist = list(getKeysByValue(weight_dict,1)),edge_color='#c25518ff',width=3.5, arrowsize=30)
labels = nx.draw_networkx_labels(G, pos,labels=labeldict,font_size=35,font_color='w',font_weight='bold')
nodes.set_zorder(0) 
#plt.title('FOG network',fontsize=36)
plt.box(False)
fig.tight_layout()
#plt.savefig(f'Animations/new_network_1801.pdf', dpi=600,transparent=True)
plt.show()

#%% calculate the centrality measures to define hubs in the network

#betweenness centrality
bc = nx.betweenness_centrality(G)
#print('Betweenness centrality:', bc)

inverse_dict = [(value, key) for key, value in bc.items()]
max_bc = max(inverse_dict)[1]
max_bc_val = bc[max_bc]
mean_bc = np.mean(list(bc.values()))
std_bc = np.std(list(bc.values()))

print('Node with maximal bc:', max_bc) #PRF
print('Value of maximal bc:', max_bc_val) #PRF
print('Value of mean bc:', mean_bc) 
print('Value of bc std:', std_bc) 
print('Value of mean cc:', nx.average_clustering(G))
print('Value of average shortest path length:', nx.average_shortest_path_length(G))

#betweenness centrality for pd
bc_pd = nx.betweenness_centrality(G_pd)
#print('Betweenness centrality, PD:', bc_pd)

inverse_dict_pd = [(value, key) for key, value in bc_pd.items()]
max_bc_pd = max(inverse_dict_pd)[1]
max_bc_pd_val = bc_pd[max_bc_pd]
mean_bc_pd = np.mean(list(bc_pd.values()))
std_bc_pd = np.std(list(bc_pd.values()))

print('Node with maximal bc, PD:', max_bc_pd) #PPN
print('Value of maximal bc, PD:', max_bc_pd_val) #PPN
print('Value of mean bc, PD:', mean_bc_pd)
print('Value of bc std, PD:', std_bc_pd) 
print('Value of mean cc, PD:', nx.average_clustering(G_pd))
print('Value of average shortest path length, PD:', 
    nx.average_shortest_path_length(G_pd))

#Absolute difference in centrality
bc_dif = {y: np.abs(bc[y] - bc_pd[y]) for y in bc if y in bc_pd}
bc_dif_sortednodes = sorted(bc_dif, key=bc_dif.get, reverse=True) #from max to min
print('Differece:', bc_dif)
print('Differece, sorted:', bc_dif_sortednodes)

#%% plots
import matplotlib.pylab as plt
from matplotlib import pyplot

matplotlib.rcParams.update({'font.size': 40})

#distances plot
plot_state = ['Healthy','Parkinsonian']

cc = [nx.average_clustering(G), nx.average_clustering(G_pd)]
betc = [mean_bc, mean_bc_pd]
maxbc = [max_bc_val, max_bc_pd_val]
sp = [nx.average_shortest_path_length(G), nx.average_shortest_path_length(G_pd)]
sdbc = [std_bc, std_bc_pd]

fig = plt.figure(figsize=(10,10))
plt.bar(plot_state, cc, color='#d35f5f', width=0.5)
plt.ylim(0)
plt.title('Mean clustering coefficient')
fig.tight_layout()
#plt.savefig(f'Animations/cc.pdf', dpi=600,transparent=True)
plt.show()

fig = plt.figure(figsize=(10,10))
plt.bar(plot_state, betc, color='#d35f5f', width=0.5)
plt.ylim(0)
plt.title('Mean betweenness centrality')
fig.tight_layout()
#plt.savefig(f'Animations/bc.pdf', dpi=600,transparent=True)
plt.show()

fig = plt.figure(figsize=(10,10))
plt.bar(plot_state, maxbc, color='#d35f5f', width=0.5)
plt.ylim(0)
plt.title('Maximal betweenness centrality')
fig.tight_layout()
#plt.savefig(f'Animations/maxbc.pdf', dpi=600,transparent=True)
plt.show()

fig = plt.figure(figsize=(10,10))
plt.bar(plot_state, sp, color='#d35f5f', width=0.5)
plt.ylim(0)
plt.title('Average shortest path length')
fig.tight_layout()
#plt.savefig(f'Animations/sp.pdf', dpi=600,transparent=True)
plt.show()

fig = plt.figure(figsize=(10,10))
plt.bar(plot_state, sdbc, color='#d35f5f', width=0.5)
plt.ylim(0)
plt.title('Std of the betweenness centrality')
fig.tight_layout()
#plt.savefig(f'Animations/sdbc.pdf', dpi=600,transparent=True)
plt.show()

#btweenness centrality by nodes
x = np.arange(len(labels_ig))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(30,10))
rects1 = ax.bar(x - width/2, list(bc.values()), width, label='Healthy', color='#98c1d9')
rects2 = ax.bar(x + width/2, list(bc_pd.values()), width, label='PD',color='#ee6c4d')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Betweenness centrality')
ax.set_xticks(x)
ax.set_xticklabels(labels_ig)
ax.legend(loc='upper right')
fig.tight_layout()
plt.show()

#%% detect modules and modularity with leiden method
import leidenalg as la
import igraph as ig

G1 = ig.Graph.Weighted_Adjacency(A.tolist())
G1.vs['label'] = labels_ig

G1_pd = ig.Graph.Weighted_Adjacency(A_pd.tolist())
G1_pd.vs['label'] = labels_ig

partition = la.find_partition(G1, la.ModularityVertexPartition)
partition_pd = la.find_partition(G1_pd, la.ModularityVertexPartition) 

#plot healhy
ig.plot(partition, edge_color=['black'], edge_width=3, vertex_size=50)

##UNCOMMENT TO PLOT FOR THE PD STATE
#plot pd
#ig.plot(partition_pd, edge_color=['black'], edge_width=3, vertex_size=50)
