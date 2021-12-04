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
    
    #graph creation
    #first statement gives you a fully randomized graph while the second one gives you a REGULAR graph with all nodes having 5 degrees
    if 1 ==0:
        size = 100
        g = Graph(directed=False)
    
        # insert random vertices (nodes)
        g.add_vertex(size)
    
        # insert some random links
        for s,t in zip(randint(0, size, 2*size), randint(0, size, 2*size)):
            g.add_edge(g.vertex(s), g.vertex(t))
    else:
         g=graph_tool.generation.price_network(100, m=5, c=None, gamma=1, directed=False,seed_graph=None)
    
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
    emax = g.new_vertex_property("int")
    g.vp.state = state
    g.vp.age = age
    g.vp.removed = removed
    g.vp.emax = emax
    
    
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
drlist = np.array([.003,.003,.003,.003,.003,.003,.005,.005,.011,.011,.03,.03,.095,.095,.228,.228,.296,.296,.296,0.296])
#same VACCINATED https://www.nejm.org/doi/pdf/10.1056/NEJMc2113864?articleTools=true
vac_drlist = np.array([0,0,0,0,0,0,0,0,.005*.12,.011*.12,.011*.10,.03*.10,.03*.10,.095*.10,.095*.10,.228*.10,.228*.10,.296*.10,.296*.10,.296*.10,0.296*.10])

#vaccine availability by age group:
vacc_av = [0,0,0,0,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
#vaccine acceptance rate by age group:
#https://www.ecdc.europa.eu/sites/default/files/documents/Interim-analysis-of-COVID-19-vaccine-effectiveness.pdf
vacc_ac = [0,0,0.57,.57,0.57,0.57,0.57,0.57,0.57,0.57,0.517,0.517,0.613,0.613,0.774,0.774,0.834,0.834,0.834,0.834]

#infection probability for different age groups https://www.nature.com/articles/s41591-020-0962-9#Sec13
inflist = np.array([0.4,0.38,0.38,0.79,0.79,0.86,0.86,0.8,0.8,0.82,0.82,0.88,0.88,0.74,0.74,0.74,0.74,0.74,0.74,0.74])
#same VACCINATED https://pubmed.ncbi.nlm.nih.gov/34202324/
vac_inflist = inflist*0.2

#recovery is now only dependent on death

#economy contribution

economy = [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]


#g = make_graph()
def extract_economy(g):
    #emax is maximal economy value calculated over all people (even dead)
    emax = 0
    #ecurr is current value calculated over people who are NOT dead, NOT infected and NOT infected vaccinated
    ecurr = 0
    for v in g.vertices():
        emax += economy[g.vp.age[v]-1]
        if g.vp.state[v]!= D:
            if g.vp.state[v]!= I and g.vp.state[v]!=Iv:
                ecurr += economy[g.vp.age[v]-1]
    return ecurr/emax

def graph_to_matrix(g):
    matrix = np.zeros([g.num_vertices(),7])
    for v in g.vertices():
        if g.vp.state[v] == S: 
            matrix[int(v),0] = 1
            matrix[int(v),6] = economy[g.vp.age[v]-1]
        if g.vp.state[v] == I: 
            matrix[int(v),1] = 1
            matrix[int(v),6] = 0
        if g.vp.state[v] == R: 
            matrix[int(v),2] = 1
            matrix[int(v),6] = economy[g.vp.age[v]-1]
        if g.vp.state[v] == Sv: 
            matrix[int(v),3] = 1
            matrix[int(v),6] = economy[g.vp.age[v]-1]
        if g.vp.state[v] == Iv: 
            matrix[int(v),4] = 1
            matrix[int(v),6] = 0
        if g.vp.state[v] == D: 
            matrix[int(v),5] = 1
            matrix[int(v),6] = 0
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
            '''
            if random() < reclist[g.vp.age[v]-1] : #r: recovery rate
                g.vp.state[v] = R
                g.clear_vertex(v) #clear all connections
            '''    
            if random() < drlist[g.vp.age[v]-1]:
                g.vp.state[v] = D       #dead
                g.clear_vertex(v)  #clear all connections
                
            else: 
                g.vp.state[v] = R
                
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



