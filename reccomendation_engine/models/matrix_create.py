from reccomendation_engine.convertor import num_users, num_items, total_nodes, interactions
import numpy as np
import math

# Edge matrix with global indexing
def make_global_edge_mat():
    edge_mat = []
    for i in interactions:
        edge_mat.append((i[0],i[1]+num_users))
    return edge_mat

# Function for making adjacency matrix
def make_adj_mat():
    edge_index = make_global_edge_mat()
    
    adj = np.zeros((total_nodes, total_nodes))
    # print(edge_index)
    for i in edge_index:
        u = i[0]
        v = i[1]
        adj[u][v] = 1
        adj[v][u] = 1
    return adj

# Function to make the degree matrix from adjacency matrix
def make_deg_mat(adj):
    deg = np.zeros((total_nodes, total_nodes))
    for index, i in enumerate(adj):
        count = 0
        for j in i:
            if(j == 1):
                count = count + 1
        deg[index][index] = count

    deg_1_2 = np.zeros((total_nodes, total_nodes))
    for i, row in enumerate(deg):
        num = math.sqrt(deg[i][i])
        deg_1_2[i][i] = round(1/num, 4)
    return deg_1_2

def make_n_adj_mat():
    adj = make_adj_mat()
    deg_1_2 = make_deg_mat(adj)

    temp_mat = np.matmul(deg_1_2, adj)
    n_adj = np.matmul(temp_mat, deg_1_2)

    return n_adj

n_adj = make_n_adj_mat()