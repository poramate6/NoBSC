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

from utils_SC import *
from utils import *


def notears_con(X, W_c, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+20, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h1(W) = 0 and h2(W) = 0 using augmented Lagrangian.
    Args:
        X (np.ndarray): [n, d] sample matrix
        W_c (np.array): [d,d] constraint matrix, 1 if inactive edge and 0 otherwise
        A_R (np.array): [d,m] matrix containing basis vectors for row constraints, m = number of inactive-edge constraints
        A_C (np.array): [d,m] matrix containing basis vectors for column constraints
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        NS_criteria (str): Marginal, Conditional
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        B_est (np.ndarray): [d, d] estimated DAG
    """
    d = X.shape[1]

    def generate_basis_mat (W):
        """Generate the constraint matrix with specified inactive-edge (non-descendant) constraints
        Args:
            W (np.array): [d,d] constraint matrix {0,1}^d*d

        Returns:
            A_R (np.array): [d1,m] matrix containing basis vectors for row constraints, m = number of edge constraints
            A_C (np.array): [d2,m] matrix containing basis vectors for column constraints
        """
        d = W.shape[0]
        Row_idx = [] 
        Col_idx = []
        for i in range(d):
            for j in range(d):
                if W[i,j] == 1: #(row,column) index for absence
                    Row_idx.append(i)
                    Col_idx.append(j)
    
        #constraint Basis matrices for constraints: A_R, A_C
        I_R = np.eye(d) 
        I_C = np.eye(d) 
        A_R = np.c_[I_R[0]]
        A_C = np.c_[I_C[0]]
        for i in range(len(Row_idx)):
            A_R = np.c_[A_R,I_R[Row_idx[i]]]
            A_C = np.c_[A_C,I_C[Col_idx[i]]]
        A_R = np.delete(A_R, 0, axis=1)
        A_C = np.delete(A_C, 0, axis=1)
    
        return A_R, A_C
        
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W  #[w_ij] weight x_i => x_j
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h1(W):
        """Evaluate value and gradient of acyclicity constraint for DAGness"""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h1 = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h1 = E.T * W * 2
        return h1, G_h1
        
    def _h2(W, R, C):
        """Evaluate value and gradient of inactive-edge constraints"""
        B = R.T @ W @ C
        Y = B * B 
        h2 = np.trace(Y)
        I_mat = np.eye(m)
        G_h2 = 2 * (R @ (I_mat * B) @ C.T)
        return h2, G_h2

    def _adj(w):
        """Convert doubled variables ([2 * d^2,] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d]) # w[:d * d] = first half, w[d * d:] = second half => reshape by row

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h1, G_h1 = _h1(W)
        h2, G_h2 = _h2(W, A_R, A_C)
        obj = loss + 0.5 * rho * (h1 * h1 + h2* h2) + alpha1 * h1 + alpha2 * h2 + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h1 + alpha1) * G_h1 + (rho * h2 + alpha2) * G_h2
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    A_R, A_C = generate_basis_mat(W_c) #basis matrices for constraints
    m = A_R.shape[1] #number of all edge constraints

    """initial values"""
    w_est, rho, alpha1, alpha2, h1, h2 = np.zeros(2 * d * d), 1.0, 0.0, 0.0, np.inf, np.inf  # double w_est into (w_pos, w_neg)
    
    """Assign the optimized boundary for each weight"""
    #Sequence of (min, max) pairs for each element in x. None is used to specify no bound
    bnds = [(0, 0) if i == j else (None, None) for _ in range(2) for i in range(d) for j in range(d)] # DAG
    C_bnds = np.array(W_c).flatten().tolist() * 2
    for i in range(len(C_bnds)):
        if C_bnds[i] == 1.0:
            bnds[i] = (0,0)

    
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
        
    """Optimization procedure"""
    for _ in range(max_iter):
        w_new, h1_new, h2_new = None, None, None
        while rho < rho_max:
            #Method L-BFGS-B uses the L-BFGS-B (Limited-memory BFGS) algorithm for bound constrained minimization
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            #sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True)
            w_new = sol.x
            h1_new, _ = _h1(_adj(w_new))
            h2_new, _ = _h2(_adj(w_new), A_R, A_C)
            if h1_new > 0.25 * h1 or h2_new > 0.25 * h2:
                rho *= 10
            else:
                break
        w_est, h1, h2 = w_new, h1_new, h2_new
        alpha1 += rho * h1
        alpha2 += rho * h2
        h = [h2, h1]
        if all(x < h_tol for x in h) or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    
    return W_est