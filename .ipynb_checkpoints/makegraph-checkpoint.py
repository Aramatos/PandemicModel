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
            agegrouplists[a-1].append(v)
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
            agegrouplists[a-1].append(v)
            counter = 1

# SET UP GRAPH
def make_graph():
    size = 100
    
    g = Graph(directed=False)
    
    #definition of vertex properties
    global S          # White color
   
    global I        # Black color
    
    global R     # Grey color (will not actually be drawn)
     
    global V        # Blue color
    
    global D        #DEATH (red)
    
    #timestamp list, basically [1,2,...]
    global tlist 
    #TIMEUNIT defines how many time units it should take until graph plotting
    global TIMEUNIT
    for i in range(TIMEUNIT):
        tlist.append(i+1)
    
    #lists to track the number of people of each state
    global sval
    global ival
    global rval
    global vval
    global dval
    global economy_list
    global scount
    global icount
    global vcount
    global rcount
    global dcount
    global ccount
    
    global tags
    #initialize tag as empty string
    tag =""
    #all definitions of vertice edges
    state = g.new_vertex_property("vector<double>")
    age=g.new_vertex_property("int")
    vac=g.new_vertex_property("bool")
    removed = g.new_vertex_property("bool")
    newly_infected = g.new_vertex_property("bool")
    
    g.vp.state = state
    g.vp.age = age
    g.vp.vac = vac
    g.vp.removed = removed 
    g.vp.newly_infected = newly_infected
    global agegroups
    global agegrouplists 

    # insert random vertices (nodes)
    g.add_vertex(size)
    
    # insert some random links
    for s,t in zip(randint(0, size, 2*size), randint(0, size, 2*size)):
        g.add_edge(g.vertex(s), g.vertex(t))
    
    global scount
    global vcount
    #makes all nodes suceptible
    for i in g.vertices():
        v = g.vertex(i) 
        
        g.vp.state[v] = S
        scount+=1
        vac[v]=False;
        
        '''if random() < vac_prob:
        
            vac[v]=True;
            vcount+=1
            scount-=1
            state[v] = V'''
    
    #this line is to randomlyinitialize one vertex as infected
    state[randint(0,size)]= I
    icount += 1
    #assign ages to vertices following given age distribution
    add_ages(g,'Japan')
    return g
    #print(graph_tool.spectral.adjacency(g,weight=None, vindex=None, operator=True)*np.identity(99))

#definition of vertex properties
S = [1, 1, 1, 1]           # White color

I = [0, 0, 0, 1]           # Black color

R = [0.5, 0.5, 0.5, 1.]    # Grey color (will not actually be drawn)

V = [0, 0, 1, 1]           # Blue color

D = [0.8, 0, 0, 0.6]       #DEATH (red)
agegroups=20
agegrouplists = [[] for i in range(1, agegroups+1)]
tlist = list()
TIMEUNIT = 20
tags = list()
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
#lists to track the number of people of each state
sval=list()
ival=list()
rval=list()
vval=list()
dval=list()
economy_list=list()
scount=0
icount=0
vcount=0
rcount=0
dcount=0
ccount=0
g = make_graph()


def graph_get():
    return g

