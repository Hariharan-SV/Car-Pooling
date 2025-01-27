# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 23:01:18 2020

@author: DELL
"""

import networkx as nx
import matplotlib.pyplot as plt
import pylab
from itertools import combinations

def printGraph(G1:nx.DiGraph,fileName:str):
    pos=nx.spring_layout(G1)
    values = [val_map.get(node, 1.85) for node in G1.nodes()]
    node_labels = {node:node for node in G1.nodes()}
    red_edges = [('C','F'),('G','A'),('G','H'),('B','H'),('E','J'),('E','F'),('H','B')]
    edge_labels=dict([((u,v,),d['weight']) for u,v,d in G1.edges(data=True)])
    nx.draw_networkx_labels(G1, pos, labels=node_labels)
    nx.draw_networkx_edge_labels(G1,pos,edge_labels=edge_labels)
    edge_colors = ['black' if not edge in red_edges else 'red' for edge in G1.edges()]
    nx.draw(G1,pos,node_color = values, node_size=1200,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
    pylab.savefig(fileName+'.png')
    plt.show()
    plt.draw()

def combinationSum(candidates, target):
  result = []
  unique={}
  candidates = list(set(candidates))
  solve(candidates,target,result,unique)
  return result

def solve(candidates,target,result,unique,i = 0,current=[]):
  if target == 0:
     temp = [i for i in current]
     temp1 = temp
     temp.sort()
     temp = tuple(temp)
     if temp not in unique:
        unique[temp] = 1
        result.append(temp1)
     return
  if target <0:
     return
  for x in range(i,len(candidates)):
     current.append(candidates[x])
     solve(candidates,target-candidates[x],result,unique,i,current)
     current.pop(len(current)-1)

G = nx.DiGraph()

G.add_edges_from([('G','T'),('T','G'),('S','E'),('E','S')], weight=1)
G.add_edges_from([('D','A'),('A','D'),('B','D'),('D','B'),('D','E'),('E','D'),('G','A'),('A','G')], weight=2)
G.add_edges_from([('B','C'),('C','B')], weight=3)
G.add_edges_from([('C','F'),('F','C')], weight=4)
G.add_edges_from([('G','H'),('H','G')], weight=4)
G.add_edges_from([('G','I'),('J','E'),('E','J'),('D','F'),('F','D')], weight=4)
G.add_edges_from([('I','J'),('J','I'),('B','H'),('H','B')], weight=6)

no_of_cars = 10
seats_per_car = 4
user_requests = {'S':2,'J':1,'I':1,'F':1,'C':1,'H':1}

val_map = {'A':0.49,'D': 2.5714285714285714,'T':2.7,'S':2.7}

values = [val_map.get(node, 1.85) for node in G.nodes()]
edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
red_edges = [('C','F'),('G','A'),('G','H'),('B','H'),('E','J'),('E','F'),('H','B')]
edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]

printGraph(G,'MainGraph')

G1 = G.subgraph(['A','B','C','E','F','G','H','I','J','S','T'])

components = [G1.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
for idx,g in enumerate(components,start=1):
    print(f"Component {idx}: Nodes: {g.nodes()} Edges: {g.edges()}")

printGraph(G1,'DisjointGraph')

print("No of disjoint components ",len(components))

sum = 0
for i in user_requests.keys():
    sum = sum + user_requests[i]
min_no_of_cars = sum // seats_per_car + 1

print("Minimum number of cars ",min_no_of_cars)

node_combination=[]
possiblities = []

for i in range(len(user_requests.keys())//min_no_of_cars,len(user_requests.keys())):
    node_combination.append(combinations(user_requests.keys(),i))

for paths in node_combination:
    for i in list(paths):
        temp = list(user_requests.keys())
        for j in list(i):
            temp.remove(j)
        possiblities.append([list(i),list(temp)])

"""
# max_no_of_cars = len(user_requests.keys())
tripArray = combinationSum(list(range(1,max_no_of_cars)),max_no_of_cars)
print(tripArray)
current_possiblities = []
for trips in tripArray:
    print("Trip value : ",trips)
    temp = list(user_requests.keys())
    cur = []
    for i in trips:
        print("Selecting ",i," passenger trip")
        value = temp[:i]
        cur.append(value)
        for item in value:
            temp.remove(item)
        if cur!=[] and cur not in current_possiblities:
            current_possiblities.append(cur)
"""

trip_array = []
trip_detail = {}
for prediction in possiblities:
    travelled = 0
    seats = 0
    current_trip = []
    for path in prediction:
        trip_detail = {"path":path}
        cur = []
        for idx in range(len(path)):
            if seats > 4:
                trip_detail["distance"] = -1
                break
            if(idx+1 != len(path)):
                next_stop = path[idx+1]
            else:
                next_stop = 'D'
            output = nx.bidirectional_dijkstra(G,path[idx],next_stop)
            travelled = travelled + output[0]
            seats = seats + 1
            trip_detail["distance"] = travelled
        current_trip.append(trip_detail)
        travelled = 0
        seats = 0
    trip_array.append(current_trip)

for trips in trip_array:
    idx=0
    for trip in trips:
        print(idx+1,trip["path"]," to office = "+str(trip["distance"]))
        idx+=1


