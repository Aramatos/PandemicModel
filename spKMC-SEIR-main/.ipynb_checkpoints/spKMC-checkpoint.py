# =======================sp-KMC-SEIR algorithms=================================================
'''
Vasiliauskaite, Vaiva, Nino Antulov-Fantulin, and Dirk Helbing. "Some Challenges in Monitoring Epidemics." arXiv preprint arXiv:2105.08384 (2021).

Extension of previous work:
Böttcher, Lucas, and Nino Antulov-Fantulin. "Unifying continuous, discrete, and hybrid susceptible-infected-recovered processes on networks." Physical Review Research 2.3 (2020): 033121.

Tolić, Dijana, Kaj-Kolja Kleineberg, and Nino Antulov-Fantulin. "Simulating SIR processes on networks using weighted shortest paths." Scientific reports 8.1 (2018): 1-10.
''''
#===============================================================================================
    
import graph_tool.all as gt
import numpy as np
def extract_edge_SEIR_weights(g__, node_recovery_weights, node_incubation_weights, edge_transmission_weights):    
    edge_recovery_weights = node_recovery_weights[g__.get_edges()[:,0]]
    edge_recovery_weights = edge_recovery_weights.reshape((g__.get_edges().shape[0], ))
    
    edge_incubation_weights = node_incubation_weights[g__.get_edges()[:,0]]
    edge_incubation_weights = edge_incubation_weights.reshape((g__.get_edges().shape[0], ))
    
    edge_weights_filter = edge_transmission_weights+edge_incubation_weights
    edge_weights_filter[edge_transmission_weights>=edge_recovery_weights] = np.Inf
    
    return edge_weights_filter
def spKMC_SEIR_full_state(g_base__, epi_params__, source__, debug__=0): 
    assert g_base__.is_directed() == 1
    
    g__ = gt.Graph(directed=True)
    edge_weight_prop = g__.new_edge_property("double")
    g__.add_edge_list(g_base__.get_edges().copy())
    
    node_incubation_weights = np.random.exponential(1/epi_params__["alpha"], [g__.get_vertices().shape[0],1])
    edge_transmission_weights = np.random.exponential(1/epi_params__["beta"], [g__.get_edges().shape[0],])
    node_recovery_weights = np.random.exponential(1/epi_params__["gamma"], [g__.get_vertices().shape[0],1])

    edge_weights_filter = extract_edge_SEIR_weights(g__,node_recovery_weights,node_incubation_weights,edge_transmission_weights)
    edge_weight_prop.get_array()[:] = edge_weights_filter
    isdirected = g__.is_directed()
    
    if type(source__)==int: source__ = [source__]
        
    if type(source__)==list:
        if len(source__)==1:
            dist_tmp = gt.shortest_distance(g__, source=g__.vertex(source__[0]), weights=edge_weight_prop,directed = isdirected)
            dist_tmp = np.asarray(dist_tmp.get_array()[:]).reshape(g__.get_vertices().shape[0],1)
        else:
            print("## Multiple source initial condition ##")
            dist_list = [gt.shortest_distance(g__, source=g__.vertex(x), weights = edge_weight_prop, directed = isdirected)\
                         for x in source__]
            dist_np_list = [np.asarray(x.get_array()[:]).reshape(g__.get_vertices().shape[0],1) for x in dist_list]
            epi_geodesic_matrix = np.concatenate(dist_np_list, axis = 1)
            dist_tmp = epi_geodesic_matrix.min(axis = 1).reshape(g__.get_vertices().shape[0],1)

        dynamics_hidden_state__ = {"epidemic_geodesics": dist_tmp, "node_recovery": node_recovery_weights,\
                                 "node_incubation": node_incubation_weights}
    
        return dynamics_hidden_state__  
    else:
        return None

def extract_SEIR_state(dynamics_hidden_state__, time__):
    # ===== micro-state extraction (state of each node) at timestamp =======
    bound_time_tmp = dynamics_hidden_state__["epidemic_geodesics"].copy()
    S_state = time__ < bound_time_tmp
    
    bound_time_tmp += dynamics_hidden_state__["node_incubation"]
    E_state = (~S_state) & (time__ < bound_time_tmp)
    
    I_state = bound_time_tmp <= time__
    bound_time_tmp += dynamics_hidden_state__["node_recovery"]
    I_state = I_state & (time__ < bound_time_tmp)
    R_state = time__ >= bound_time_tmp
    assert np.sum(S_state&E_state&I_state&R_state) == 0
    assert np.sum(S_state|E_state|I_state|R_state) == bound_time_tmp.shape[0]
    assert np.sum(S_state)+np.sum(E_state)+np.sum(I_state)+np.sum(R_state) == bound_time_tmp.shape[0]
    return {"S":S_state, "E": E_state, "I": I_state, "R": R_state}

def event_driven_state_extraction(dynamics_hidden_state__, T__):
    # ======macro-state curve (e.g. S(t), E(t),I(t), R(t)) event-driven extraction in range (0,T) in O(N+T)=======
    node_exposed_time_array = dynamics_hidden_state__["epidemic_geodesics"].copy()
    node_infected_time_array = node_exposed_time_array + dynamics_hidden_state__["node_incubation"]
    node_recoved_time_array = node_infected_time_array + dynamics_hidden_state__["node_recovery"]
    #integer ceiling operation ....
    node_exposed_time_array = np.ceil(node_exposed_time_array)
    node_infected_time_array = np.ceil(node_infected_time_array)
    node_recoved_time_array = np.ceil(node_recoved_time_array)
    N = len(node_exposed_time_array)
    # =================exposed state ==================
    event_E_time_array = np.zeros(T__)
    time_int_exposed_events = node_exposed_time_array[node_exposed_time_array<T__].astype(int)
    np.add.at(event_E_time_array, time_int_exposed_events, +1)
    time_int_infected_events = node_infected_time_array[node_infected_time_array<T__].astype(int)
    np.add.at(event_E_time_array, time_int_infected_events, -1)
    E_time_array = np.cumsum(event_E_time_array)
    #  ================= infected state ================
    time_int_recovered_events = node_recoved_time_array[node_recoved_time_array<T__].astype(int)
    event_I_time_array = np.zeros(T__)
    np.add.at(event_I_time_array, time_int_infected_events, +1)
    np.add.at(event_I_time_array, time_int_recovered_events, -1)
    I_time_array = np.cumsum(event_I_time_array)
    #  =================recovered state ==================
    event_R_time_array = np.zeros(T__)
    np.add.at(event_R_time_array, time_int_recovered_events, +1)
    R_time_array = np.cumsum(event_R_time_array)
    #  ================= susceptible state ==================
    event_S_time_array = np.zeros(T__)
    np.add.at(event_S_time_array, time_int_exposed_events, -1)
    S_time_array = np.cumsum(event_S_time_array)+N
    return {"S":S_time_array, "E": E_time_array, "I": I_time_array, "R": R_time_array}
