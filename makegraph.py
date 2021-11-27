#This file creates a graph (same as Trial does)
#helper file
from graph_tool.all import *
from numpy.random import randint
from numpy.random import random
from numpy.random import *
import numpy as np


def add_ages(g,distribution):
    path1 = 'Age_Dists/'
    path2 = 'Dist.csv'
    #generate array containing amount of people per age group
    agedistribution = np.genfromtxt(path1+distribution+path2, delimiter=',')[:,1]
    #visit vertices in random order
    #print(agedistribution)
    vs = list(g.vertices())
    shuffle(vs)
    #we will fill all age groups one by one
    a = 1 #current agegroup
    counter = 0 #counter for how many people have been placed in current group
    
    for v in vs:
        if(a > 20):
            #safety measure
            return
        if counter < agedistribution[a-1]:
            #if current age group hasn't gotten enough people assigned, add another, move on to next vertex
            g.vp.age[v] = a
            #agegrouplists[a-1].append(v)
            counter = counter + 1
        else:
            #current age group is full, move to next age group with at least one person
            a = a + 1
            while(agedistribution[a-1] == 0):
                if(a < 20):
                    a = a + 1
                else:
                    #in case number for last age group is 0
                    return
            g.vp.age[v] = a
            #agegrouplists[a-1].append(v)
            counter = 1

# SET UP GRAPH
def make_graph(size,distribution):
    #size = 100
    
    g = Graph(directed=False)
    
    #definition of vertex properties
    global S          # White color
   
    global I        # Black color
    
    global R     # Grey color (will not actually be drawn)
     
    global Sv        # Blue color
    
    global Iv        #Dark blue color
    
    global D        #DEATH (red)
    
    #timestamp list, basically [1,2,...]
        
    #lists to track the number of people of each state
    
    #all definitions of vertice edges
    state = g.new_vertex_property("vector<double>")
    age=g.new_vertex_property("int")
    removed = g.new_vertex_property("bool")
    
    g.vp.state = state
    g.vp.age = age
    g.vp.removed = removed

    # insert random vertices (nodes)
    g.add_vertex(size)
    
    # insert some random links
    for s,t in zip(randint(0, size, 2*size), randint(0, size, 2*size)):
        g.add_edge(g.vertex(s), g.vertex(t))
    
    #makes all nodes suceptible
    for i in g.vertices():
        v = g.vertex(i) 
        
        g.vp.state[v] = S
    
    #this line is to randomly initialize one vertex as infected
    state[randint(0,size)]= I
    #assign ages to vertices following given age distribution
    add_ages(g,distribution)
    return g

#definition of vertex properties
S = [1, 1, 1, 1]          # SUSCEPTIBLE               (white)

I = [0, 0, 0, 1]          # INFECTED                  (black)

R = [0.5, 0.5, 0.5, 1.]   # RECOVERED                 (grey, will not be drawn)

Sv = [0, 0, 1, 1]         # SUSCEPTIBLE + VACCINATED  (blue)

Iv = [0.2,0.2,0.5,1]      # INFECTED + VACCINATED     (dark blue)

D = [0.8, 0, 0, 0.6]      # DEAD                      (red)

'''agegroups=20
agegrouplists = [[] for i in range(1, agegroups+1)]
tlist = list()
TIMEUNIT = 20
tags = list()'''


#probabilities


#death probability list for different age groups: https://pubmed.ncbi.nlm.nih.gov/32674819/
drlist = [.003,.003,.003,.003,.003,.003,.005,.005,.011,.011,.03,.03,.095,.095,.228,.228,.296,.296,.296,0.296]
#same VACCINATED
vac_drlist = [.001,.001,.001,.001,.001,.001,.002,.002,.005,0.005,0.015,0.015,.047,.047,.114,.114,.148,.148,.148,.148]


#vaccine availability by age group:
vacc_av = [0,0,0,0,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
#vaccine acceptance rate by age group:
vacc_ac = [0,0,0,0,0.5,0.5,0.5,0.5,0.6,0.6,0.6,0.6,0.7,0.7,0.7,0.7,0.8,0.8,0.9,0.9]

#infection probability for different age groups
inflist = [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8]
#same VACCINATED
vac_inflist = [.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,.20,0.20]

#recovery probability for different age groups
reclist = [0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.32,0.32,0.28,0.28,0.22,0.22,0.15,0.15,0.10,0.10,0.08,0.08]
#same VACCINATED
vac_reclist = [0.39,0.39,0.39,0.39,0.39,0.39,0.39,0.39,0.35,0.35,0.31,0.31,0.26,0.26,0.20,0.20,0.14,0.14,0.13,0.13]

#economy contribution

economy = [0,0,0,0.025,0.025,0.05,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0.05,0.025,0.025,0,0,0]


#g = make_graph()

def graph_to_matrix(g):
    matrix = np.zeros([g.num_vertices(),6])
    for v in g.vertices():
        if g.vp.state[v] == S: 
            matrix[int(v),0] = 1
        if g.vp.state[v] == I: 
            matrix[int(v),1] = 1
        if g.vp.state[v] == R: 
            matrix[int(v),2] = 1
        if g.vp.state[v] == Sv: 
            matrix[int(v),3] = 1
        if g.vp.state[v] == Iv: 
            matrix[int(v),4] = 1
        if g.vp.state[v] == D: 
            matrix[int(v),5] = 1
    return matrix

def graph_get():
    return g

def update_state(g,action):
    
    #newly_infected.a = False
    
    #g.vp.removed.a = False
    
    vs = list(g.vertices())

    shuffle(vs)

    for v in vs:

        if g.vp.state[v] == I: #infected
            
            if random() < reclist[g.vp.age[v]-1] : #r: recovery rate
                g.vp.state[v] = R
                g.clear_vertex(v) #clear all connections
                
            elif random() < drlist[g.vp.age[v]-1]:
                g.vp.state[v] = D       #dead
                g.clear_vertex(v)  #clear all connections
                
        if g.vp.state[v] == Iv: #infected and vaccinated
            
            if random() < vac_reclist[g.vp.age[v]-1]:
                g.vp.state[v] = R
                g.clear_vertex(v)  #clear all connections
                
            elif random() < vac_drlist[g.vp.age[v]-1]:
                g.vp.state[v] = D       #vac dead
                g.clear_vertex(v)  #clear all connections

        elif g.vp.state[v] == S: #susceptible
            #economy_value += economy[age[v]-1]
            if random() < action[g.vp.age[v]-1]*vacc_ac[g.vp.age[v]-1]:
                #action models vaccine availability
                g.vp.state[v] = Sv #if vaccine is available and accepted by individual, vaccinate
            
            else: 
                ns = list(v.out_neighbors())
                
                i = 0

                for w in ns: #iterate through all neighbors, each has infection probability

                    if (g.vp.state[w] == I or g.vp.state[w] == Iv) and random()<inflist[g.vp.age[v]-1]:
                    #infection rate dependent on age
                        
                        i = 1 
                        
                if i:#if infected by at least one neighbor, change status
                    g.vp.state[v] = I
    
        elif g.vp.state[v] == Sv:
            #economy_value += economy[age[v]-1]
            ns = list(v.out_neighbors())

            i = 0

            for w in ns:
                
                if (g.vp.state[w] == I or g.vp.state[w] == Iv) and random()<vac_inflist[g.vp.age[v]-1]:
                    
                    i = 1
            
            if i:
                g.vp.state[v] = I
                        

        #if g.vp.state[v] == R:
            #economy_value += economy[age[v]-1]
            #g.vp.removed[v] = True
            
       
    #economy_list.append(economy_value)

    # Filter out the recovered vertices

    g.set_vertex_filter(g.vp.removed, inverted=True)