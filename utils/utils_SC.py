import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import random
import pandas as pd
import unittest
import pydot
import math
import matplotlib.pyplot as plt
import igraph as ig
import lingam
import networkx as nx
import seaborn as sns
import xlrd
from multiprocessing import Pool
import multiprocess as mp
from datetime import datetime
import argparse 
from tqdm import tqdm
from functools import partial  
import scipy 
from scipy.optimize import minimize
from scipy import stats 
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit
from scipy.special import softmax
import torch
import torch.linalg as linalg
import torch.optim as optim
from torch.optim import lr_scheduler

#=========================================
# Data Generating Functions for Simulation 
#=========================================

##(NOTEARS - Zheng et al. 2018)
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X

def simulate_mixed_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified mixed types of noises.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (list str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale, v_type):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if v_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif v_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif v_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    for i in range(d):
        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type[i] == 'gauss':
                # make 1/d X'X = true cov
                X[:, i] = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W[:, i])
                return X[:, i]
            else:
                raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j], sem_type[j])
    return X

def sim_mixed_sem(W, n_c, n_f, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified mixed types of continuous and binary noises
    with exogenous categorical features.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n_f (int): num of features 
        n (int): num of samples, n=inf mimics population risk
        sem_type (list str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale, v_type):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if v_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif v_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif v_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    for i in range(d):
        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type[i] == 'gauss':
                # make 1/d X'X = true cov
                X[:, i] = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W[:, i])
                return X[:, i]
            else:
                raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])

    #number of variables for each feature 
    reduced = [x - 1 for x in n_c]
    num_ones = n_f - len(reduced)
    ones = [1] * num_ones
    c = reduced + ones
    #Indices in X corresponding to features
    w_indices = []
    start = 0
    for length in c:
        end = start + length
        w_indices.append(list(range(start, end)))
        start = end
    
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        ##Gaurantee Onehot encoding
        for k in range(len(n_c)):
            cate_samp = X[:,w_indices[k]]
            for i, row in enumerate(cate_samp):
                ones_indices = [j for j, val in enumerate(row) if val == 1]
                if len(ones_indices) > 1:
                    chosen_index = random.choice(ones_indices)
                    cate_samp[i] = [1 if j == chosen_index else 0 for j in range(len(row))]
            X[:,w_indices[k]] = cate_samp
 
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j], sem_type[j])
    return X


def simulate_mixed_cat(W, d, n, n_c, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise and exogenous multi-class.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [p, p] weighted adj matrix, where k = num responses, p = k - num of cat features (exclude baseline)
        n (int): num of samples, n=inf mimics population risk
        n_c (list of float): num of classes for each categorical feature (nominal)
        sem_type (list str): gauss, exp, gumbel, uniform, logistic, poisson, multiclass
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def softmax(x):
        """Compute the softmax"""
        exp_x = np.exp(x - np.max(x, axis=1,keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def _simulate_single_equation(X, w, scale, v_type, n_classes):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if v_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif v_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif v_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif v_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        elif v_type == 'multiclass':
            M = X @ w 
            zeros = np.zeros(n).reshape((n,1))
            Scores = np.hstack((M, zeros)) #scores (last level baseline has no coefficients and score = 0)
            Z = softmax(Scores) # Apply softmax (include baseline as exponent 0)
            # Sample class labels based on probabilities
            x = np.array([np.random.choice(n_classes, p=prob) for prob in Z])  
            x = x + 1 
        else:
            raise ValueError('unknown sem type')
        return x

    p = W.shape[0] #num of predictors 
    #noise_type = sem_type[sum(n_c):] 
    #for j in range(len(n_c)):
         #noise_type = ['multiclass'] + noise_type  # data type for each feature 

    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    
    #submatrix for non-categorical features
    rows = list(range(np.sum(n_c)-len(n_c),p))
    columns = list(range(np.sum(n_c)-len(n_c),p))
    I_R = np.eye(p)
    Row = I_R[:,rows]
    I_C = np.eye(p)
    Col = I_C[:,columns]
    B = Row.T @ W @ Col #square matrix
    if not is_dag(B):
        raise ValueError('W must be a DAG')
        
    for i in range(d):
        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type[i] == 'gauss':
                # make 1/d X'X = true cov
                w_ind = i - 2 * len(n_c) + sum(n_c)
                X[:, i] = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W[:, w_ind])
                return X[:, i]
            else:
                raise ValueError('population risk not available')

    #Collapse W to [d,d] matrix: Z 
    non_pred = [x - 1 for x in n_c] #number of predictors for categorical features
    Mat_res = np.zeros(d*p).reshape([d,p])
    v = 0 
    for i in range(len(non_pred)):
        col_abs_max_indices = np.argmax(np.abs(W[v:v+non_pred[i], :]), axis=0) + v
        Mat_res[i,:] = W[col_abs_max_indices, range(p)]
        v += non_pred[i]
    Mat_res[(len(non_pred)):,:] = W[v:,:]
    Mat_pred = np.zeros(d*d).reshape([d,d])
    a = 0
    for j in range(len(non_pred)):
        row_abs_max_indices = np.argmax(np.abs(Mat_res[:, a:a+non_pred[j]]), axis=1) + a 
        Mat_pred[:,j] = Mat_res[np.arange(d), row_abs_max_indices]
        a += non_pred[j]   
    Mat_pred[:,(len(non_pred)):] = Mat_res[:,a:]
    Z = Mat_pred
    
    G = ig.Graph.Weighted_Adjacency(Z.tolist())
    ordered_vertices = G.topological_sorting() #causal order
    assert len(ordered_vertices) == d
    
    X = np.zeros([n,d])
    X_pred = np.zeros([n,p])

    #number of variables for each feature 
    reduced = [x - 1 for x in n_c]
    num_ones = d - len(reduced)
    ones = [1] * num_ones
    c = reduced + ones

    #W indices corresponding to X indices
    w_indices = []
    start = 0
    for length in c:
        end = start + length
        w_indices.append(list(range(start, end)))
        start = end
    
    for j in ordered_vertices:
        x_parents = G.neighbors(j, mode=ig.IN) #parents in X 

        #parents in W 
        w_parents = [] 
        for k in range(len(x_parents)):
            w_parents = w_parents + w_indices[x_parents[k]]

        if sem_type[j] == 'multiclass':
            x = _simulate_single_equation(X_pred[:, w_parents],
                                          W[np.ix_(w_parents, w_indices[j])], 
                                          scale_vec[j], sem_type[j],c[j] + 1)
            X[:, j] = x  
            x_zero_based = x - 1
            one_hot = np.eye(c[j] + 1)[x_zero_based] # One-hot encoding
            X_pred[:,w_indices[j]] = np.delete(one_hot, c[j], axis=1) #remove baseline  
        else:
            x = _simulate_single_equation(X_pred[:, w_parents], W[w_parents, w_indices[j]], scale_vec[j], sem_type[j],c[j])
            X[:, j] = x
            X_pred[:,w_indices[j]] = x.reshape(-1, 1)
        
    return X

def cate_dag_constraints(n_c, k):
    """    
    Args:
    n_c (list of float): number of classes for each categorical feature 
    k (float): number of total responses 

    Returns:
    W_c (np.array): [k-c,k] basis matrix for row constraints, where c = number of categories
    R (np.array): [k,m] basis matrix for row constraints, where m = number of edge constraints
    C (np.array): [k,m] basis matrix for column constraints
    """
    p = k - len(n_c)
    rows = list(range(np.sum(n_c)-len(n_c),p))
    columns = list(range(np.sum(n_c),k))

    n_predictors = [x - 1 for x in n_c]
    Row = []
    Col = []
    idx = 0 
    offset = 0
    for xi, yi in zip(n_c, n_predictors):
        for _ in range(yi):
            Row.extend([idx] * xi)
            Col.extend(range(offset, offset + xi))
            idx += 1
        offset += xi
    Row.extend(rows)
    Col.extend(columns)
     #constraint matrix W_c and Basis matrices for constraints: A_R, A_C
    W_c = np.zeros((k - len(n_c)) * k).reshape([(k - len(n_c)),k])
    I_R = np.eye(k-len(n_c)) 
    I_C = np.eye(k)
    R = np.c_[I_R[0]]
    C = np.c_[I_C[0]]
    for i in range(len(Row)):
        W_c[Row[i],Col[i]] = 1
        R = np.c_[R,I_R[Row[i]]]
        C = np.c_[C,I_C[Col[i]]]
    R = np.delete(R, 0, axis=1)
    C = np.delete(C, 0, axis=1)   

    return W_c, R, C

## Find the constraint matrix
def find_constraint (Mat, m):
    """Generate the constraint matrix with specified number of inactive-edge constraints
        randomly select them from a set of all inactive edges
    
    Args:
    Mat (np.array): [d,d] adjacency matrix
    m (int): number of edge constraints

    Returns:
    W_c (np.array): [d,d] constraint matrix {0,1}^d*d
    A_R (np.array): [d,m] matrix containing basis vectors for row constraints, m = number of edge constraints
    A_C (np.array): [d,m] matrix containing basis vectors for column constraints
    W_L (np.array): [d,d] constraint matrix for DirectLiNGAM algorithm
    """
    y = Mat.shape
    d = y[0]
    for i in range(d): #remove diagonal entries
        Mat[i,i] = 1
    Row = [] 
    Col = []
    for i in range(d):
        for j in range(d):
            if Mat[i,j] == 0: #(row,column) index for absence
                Row.append(i)
                Col.append(j)
    seq_row = range(0,len(Row)) 
    samp_idx = random.sample(seq_row,m) #randomly select m indexes for rows and columns
    samp_idx = sorted(list(samp_idx))
   
    #get constraints row and column indexes
    Row_idx = []
    Col_idx = []
    for i in range(len(samp_idx)): 
        Row_idx.append(Row[samp_idx[i]])
        Col_idx.append(Col[samp_idx[i]])
    
    #constraint matrix W_c and Basis matrices for constraints: A_R, A_C
    W_c = np.zeros(d * d).reshape([d,d])
    I = np.eye(d) 
    A_R = np.c_[I[0]]
    A_C = np.c_[I[0]]
    for i in range(len(Row_idx)):
        W_c[Row_idx[i],Col_idx[i]] = 1
        A_R = np.c_[A_R,I[Row_idx[i]]]
        A_C = np.c_[A_C,I[Col_idx[i]]]
    A_R = np.delete(A_R, 0, axis=1)
    A_C = np.delete(A_C, 0, axis=1)

    #constraint matrix W_L for DirectLiNGAM
    W_L = np.zeros(d * d).reshape([d,d]) 
    W_L[W_c == 1] = 0 
    W_L[W_c == 0] = -1
    
    return W_c, A_R, A_C, W_L

def NS_constraint (Mat, m):
    """Generate the NS-constraint matrix with specified number of inactive-edge constraints
        Assign the last feature as the outcome of interest 
    
    Args:
    Mat (np.array): [d,d] adjacency matrix
    m (int): number of edge constraints

    Returns:
    W_c (np.array): [d,d] constraint matrix {0,1}^d*d
    A_R (np.array): [d,m] matrix containing basis vectors for row constraints, m = number of edge constraints
    A_C (np.array): [d,m] matrix containing basis vectors for column constraints
    W_L (np.array): [d,d] constraint matrix for DirectLiNGAM algorithm
    """
    
    y = Mat.shape
    d = y[0]
    for i in range(d): #remove diagonal entries
        Mat[i,i] = 1
    Mat[(d-1),:] = 1 #remove edges pointing from outcome
    Row = [] 
    Col = []
    for i in range(d):
        for j in range(d):
            if Mat[i,j] == 0: #(row,column) index for absence
                Row.append(i)
                Col.append(j)
    seq_row = range(0,len(Row)) 
    samp_idx = random.sample(seq_row,m) #randomly select m indexes for rows and columns
    samp_idx = sorted(list(samp_idx))
   
    #get constraints row and column indexes
    Row_idx = []
    Col_idx = []
    for i in range(len(samp_idx)): 
        Row_idx.append(Row[samp_idx[i]])
        Col_idx.append(Col[samp_idx[i]])

    #add constraints row and columns for outcome
    for i in range(d):
        Row_idx.append(d-1)
        Col_idx.append(i)
  
    #constraint matrix W_c and Basis matrices for constraints: A_R, A_C
    W_c = np.zeros(d * d).reshape([d,d])
    I = np.eye(d) 
    A_R = np.c_[I[0]]
    A_C = np.c_[I[0]]
    for i in range(len(Row_idx)):
        W_c[Row_idx[i],Col_idx[i]] = 1
        A_R = np.c_[A_R,I[Row_idx[i]]]
        A_C = np.c_[A_C,I[Col_idx[i]]]
    A_R = np.delete(A_R, 0, axis=1)
    A_C = np.delete(A_C, 0, axis=1)

    #constraint matrix W_L for DirectLiNGAM
    W_L = np.zeros(d * d).reshape([d,d]) 
    W_L[W_c == 1] = 0 
    W_L[W_c == 0] = -1
    
    return W_c, A_R, A_C, W_L

def non_descendant_constraint (nodes,d1,d2,NS):
    """Generate the constraint matrix with specified inactive-edge (non-descendant) constraints
    
    Args:
    nodes (list): non-descendant node index
    d1 (int): number of predictor nodes
    d2 (int): number of response nodes
    NS: True/False

    Returns:
    W_c (np.array): [d1,d2] constraint matrix {0,1}^d*d
    A_R (np.array): [d1,m] matrix containing basis vectors for row constraints, m = number of edge constraints
    A_C (np.array): [d2,m] matrix containing basis vectors for column constraints
    W_L (np.array): [d1,d2] constraint matrix for DirectLiNGAM algorithm
    """
    Mat = np.zeros(d1*d2).reshape([d1,d2])
    for i in range(len(nodes)): 
        Mat[:,nodes[i]] = 1
    if NS == True:
        Mat[(d2-1):,] = 1 #assign last feature as the outcome

    Row_idx = [] 
    Col_idx = []
    for i in range(d1):
        for j in range(d2):
            if Mat[i,j] == 1: #(row,column) index for absence
                Row_idx.append(i)
                Col_idx.append(j)
    
    #constraint matrix W_c and Basis matrices for constraints: A_R, A_C
    W_c = np.zeros(d1 * d2).reshape([d1,d2])
    I_R = np.eye(d1) 
    I_C = np.eye(d2) 
    A_R = np.c_[I_R[0]]
    A_C = np.c_[I_C[0]]
    for i in range(len(Row_idx)):
        W_c[Row_idx[i],Col_idx[i]] = 1
        A_R = np.c_[A_R,I_R[Row_idx[i]]]
        A_C = np.c_[A_C,I_C[Col_idx[i]]]
    A_R = np.delete(A_R, 0, axis=1)
    A_C = np.delete(A_C, 0, axis=1)

    #constraint matrix W_L for DirectLiNGAM
    W_L = np.zeros(d1 * d2).reshape([d1,d2]) 
    W_L[W_c == 1] = 0 
    W_L[W_c == 0] = -1
    
    return W_c, A_R, A_C, W_L

def pc_convert_adj(Mat):
    """convert weighted adj matrix to a binary adj matrix for FCI algorithms
    Args: 
        Mat[i <- j] (np.array): [d,d] estimated graph
    Return:
        X: [i -> j] (np.array): [d,d] estimated adj matrix 
        undirected_edges e.g.(-1,-1)
    """
    d = Mat.shape[0]
    X = np.zeros((d,d))
    for i in range(d):
        for j in range(i,d):
            if Mat[j,i] == 1 and Mat[i,j] == -1:
                X[j,i] = 1
            elif Mat[j,i] == -1 and Mat[i,j] == 1:
                X[i,j] = 1
            elif Mat[j,i] == -1 and Mat[i,j] == -1:
                X[i,j] = -1
                X[j,i] = -1
            elif Mat[j,i] == 1 and Mat[i,j] == 1:
                X[i,j] = 1
                X[j,i] = 1
    for i in range(d): 
        X[i,i] = 0 #loops
    return X.T

def fci_convert_adj(Mat):
    """convert weighted adj matrix to a binary adj matrix for FCI algorithms
    Args: 
        Mat[i <- j] (np.array): [d,d] estimated graph
    Return:
        X[i -> j] (np.array): [d,d] estimated adj matrix
    """
    d = Mat.shape[0]
    X = np.zeros((d,d))
    for i in range(d):
        for j in range(i,d):
            if Mat[j,i] == 1 and Mat[i,j] == -1:
                X[j,i] = 1
            elif Mat[j,i] == -1 and Mat[i,j] == 1:
                X[i,j] = 1
            elif Mat[j,i] == 1 and Mat[i,j] == 2:
                X[i,j] = 0 
            elif Mat[j,i] == 2 and Mat[i,j] == 1:
                X[i,j] = 0 
            elif Mat[j,i] == 1 and Mat[i,j] == 1:
                X[i,j] = 1
                X[j,i] = 1
    
    for i in range(d): 
        X[i,i] = 0 #loops
    return X.T

def collapse_graph(W,p,d,n_c):
    """
    Args:
        W (np.array): [d, d] weighted adj matrix
        p (int): num of predictors
        p (int): num of true features 
        n_c (list of float): number of classes for each categorical feature (nominal)
    Return:
        Mat_pred (np.array): [d, d] estimated DAG
    """
    non_pred = [x - 1 for x in n_c] #number of predictors for categorical features
    Mat_res = np.zeros(d*p).reshape([d,p])
    v = 0 
    for i in range(len(non_pred)):
        col_abs_max_indices = np.argmax(np.abs(W[v:v+non_pred[i], :]), axis=0) + v
        Mat_res[i,:] = W[col_abs_max_indices, range(p)]
        v += non_pred[i]
    Mat_res[(len(non_pred)):,:] = W[v:,:]
    Mat_pred = np.zeros(d*d).reshape([d,d])
    a = 0
    for j in range(len(non_pred)):
        row_abs_max_indices = np.argmax(np.abs(Mat_res[:, a:a+non_pred[j]]), axis=1) + a 
        Mat_pred[:,j] = Mat_res[np.arange(d), row_abs_max_indices]
        a += non_pred[j]   
    Mat_pred[:,(len(non_pred)):] = Mat_res[:,a:]
    return Mat_pred

def CCF_edge(X, W_c, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+20, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.
    Args:
        X (tensor): [n, d] sample matrix
        W_c (np.array): [d, d] inactive constraint matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    n, d = X.shape
    #X = torch.as_tensor(X,dtype=torch.float32)
    import torch
    import torch.linalg as linalg    

    def _loss(W):
        """Evaluate value of l2 loss"""
        M = torch.mm(X,W)
        R = X - M
        loss = 0.5 / X.shape[0] * torch.sum(R ** 2)
        return loss

    def _h(W):
        """Evaluate value of acyclicity and inactive constraint."""
        E = linalg.matrix_exp(W * W)  # (Zheng et al. 2018)
        G = torch.abs(W) * W_c #(Wang et al. 2024)
        h = torch.trace(E) - d + torch.norm(G, p=1)
        return h

    def _func(W):
        """Evaluate value of augmented Lagrangian"""
        loss = _loss(W)
        h = _h(W)
        obj = loss + 0.5 * rho * h ** 2 + alpha * h + lambda1 * W.sum()
        return obj

    """initial values"""
    init_w = torch.zeros([d,d], dtype=torch.float32)
    w_est  = init_w.clone().requires_grad_(True)
    rho, alpha, h = 1.0, 0.0,  float('inf')
    
    if loss_type == 'l2':
        X = X - X.mean(dim=0, keepdim=True)
        
    """Optimization procedure"""
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            w_new = w_est.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([w_new], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            for step in range(100):
                obj = _func(w_new)
                optimizer.zero_grad()
                obj.backward(retain_graph=True) 
                optimizer.step()
  
            w_new = w_new.detach().clone().requires_grad_(True)
            h_new = _h(w_new)
            if h_new > 0.25 * h:
                rho = rho*10
            else:
                break
        w_est, h = w_new, h_new
        alpha = alpha + rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = w_est.detach()
    W_est[W_est.abs() < w_threshold] = 0
    
    return W_est

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size

def count_accuracy2(B_true, B) -> tuple:
    """Compute FDR, TPR, and SHD for a matrix B.
        
        Args:
        G_true: ground truth graph
        G: predicted graph
        
        Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        shd: undirected extra + undirected missing + reverse
        """
#     B_true = nx.to_numpy_array(G_true) != 0
#     B = nx.to_numpy_array(G) != 0
    d = B.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    B_lower = np.tril(B + B.T)
    pred_lower = np.flatnonzero(B_lower)
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return fdr, tpr, fpr, shd, pred_size

def plot_pc(mt, labels_name=None, file_name=None):     
    G = nx.DiGraph(mt)
    d = mt.shape[0]
    pos = nx.circular_layout(G)

    #Undirected edges in G
    undirected_edges = []
    for i in range(d):
        for j in range(i,d):
            if mt[j,i] == -1 and mt[i,j] == -1:
                undirected_edges.append((i,j))

    # Label setup
    labels = {i: labels_name[i] if labels_name is not None else i for i in range(d)}

    # Draw directed edges (excluding the ones to be undirected)
    if undirected_edges is None:
        undirected_edges = []

    directed_edges = [e for e in G.edges() if e not in undirected_edges and (e[1], e[0]) not in undirected_edges]
    nx.draw_networkx_edges(G, pos, edgelist=directed_edges, arrowstyle='->', arrows=True, arrowsize=15, width=0.5)

    # Draw undirected edges
    nx.draw_networkx_edges(G, pos, edgelist=undirected_edges, arrows=False, style='solid', width=0.8)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color='#A0CBE2', node_size=200, linewidths=0.25)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.axis('off')
    
    if labels_name is not None and file_name:
        plt.savefig(file_name + '.pdf')
    plt.show()