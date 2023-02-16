import jax.numpy as np
import jax.random as random
from jax.scipy.linalg import cholesky
# from jax.ops import index, index_update
from jax import jit, vmap

from functools import partial

# routines for RKHS interpolation

@jit
def softplus(x):
    # [-inf, inf]->[0,inf]
    return np.logaddexp(x, 0.)

@jit
def exp_quadratic(x1, x2, params):    
    return params[0]*np.exp(-params[1] * np.sum((x1 - x2)**2))

@partial(jit, static_argnums=(0))
def cov_map(cov_func, xs, xs2 = None):
    # xs and xs2 are stacked along the leading dimension
    if xs2 is None:
        return vmap(lambda x:  vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T

@partial(jit, static_argnums=(1))   
def cov_inv_matr(X_t, kernel_fun):
    # inverse covariance matrix of kernel evaluated on nodes X_t (using exp_quadratic)
    # note eps acts as gaussian noise parameter
    
    eps = 1e-7 # nugget for X_t covariance, i.e. Gaussian noise for point jitter
    Nt = X_t.shape[0] # number of nodes for RKHS matrix interpolation
    train_cov =  cov_map(kernel_fun, X_t) + np.eye(Nt) * eps
    
    chol = cholesky(train_cov, lower=True)
    c = np.linalg.inv(chol); 
    return np.dot(c.T,c)

@partial(jit, static_argnums=(3))
def vec_interp(X_t, T_t, cov_inv, kernel_fun, x):
    '''RKHS-interpolate elements of T_t, given at X_t to x without coregination
    
       INPUT: 
        X_t         -- Nt x d matrix of nodes
        T_t         -- Nt x p matrix of T_t (each T_i is a row-vector). 
                       For T_i representing triu-part of a matrix, p = d*(d+1)/2
        x           -- 1 x d interpolation point
        cov_inv     -- inverted covariance matrix of X_t, X_t
        kernel_fun  -- kernel function for computing cross-covariance
        
       OUTPUT:
        T           -- 1 x p vector
    '''
    
    
    cross_cov = cov_map(kernel_fun, x, X_t)
    return np.dot(cross_cov, cov_inv @ T_t)

batch_vec_interp = jit(vmap(vec_interp, in_axes = (None, None, None, None, 0), out_axes = 0),
                        static_argnums = (3))


@partial(jit, static_argnums=(3))
def matr_interp(X_t, T_t, cov_inv, kernel_fun, x):
    """  
    Interpolate matrix-valued function given by data {X_t, T_matr} at x
    
    Input: x        - [1 x d] is a point of R^d, where we need to compute D
    
           X_t      - [Nt x d] is matrix of nodes for RKHS interpolation 
                      (nodes are stacked along leading dimension of X_t)
                      
           T_t   - [Nt x d x d] array of Diffusion Tensors at X_t, 
                      (diff tensors are stacked along leading dimension of T_matr)

           cov_inv  - inverted covariance matrix of X_t, X_t
           
           kernel_params  - kernel parameters -- needed to compute cross-cov matrix. 
                       Alternatively, pass cross-cov matrix
           
    Output: [d x d] matrix D(x), which iterpolates T to x
    
    Note: 
        Forward shuffle of T reshapes it into array of columns
            T_1 = T_t.transpose((2,1,0)).reshape(d*d,Nt).T
            
        Backward shuffle restores T:
                  T_1.reshape(Nt,d,d).transpose((0,2,1))
                  
        After shuffled D(x) is computed, it is backward shuffled by 
                  D_1.reshape(d,d).T
                  
    Observations: 
        If trained on psd, somehow this returns psd! -- so no need for training on logm
        If points far from X_t -- it returns mean of T_matr
         """
    Nt = X_t.shape[0] # number of nodes for RKHS matrix interpolation
    # Reshuffle T_matr into one matrix of Nt column-vectors, each representing T_1[i] 
    # (consider eliminating this variable: no need to create this huge matrix)
    T1 = T_t.transpose((2,1,0)).reshape(d*d,Nt).T 
    
    #Tmean = T1.mean(axis = 0)
    Tmean = 0.
    
    cross_cov = cov_map(kernel_fun, x, X_t)
    
    D = np.dot(cross_cov, cov_inv @ (T1-Tmean)) + Tmean
    
    return D.reshape(d,d).T



@partial(jit, static_argnums=(3))
def triu_interp(X_t, T_t, cov_inv, kernel_fun, x):
    '''RKHS-interpolate elements of T_t, given at X_t to x without coregination
    
       INPUT: 
        X_t         -- Nt x d matrix of nodes
        T_t         -- Nt x d*(d+1)/2 matrix elements of Nt matrices T_i: 
                       each row is a flattened triu part of a symmetric matrix T_i
        x           -- 1 x d interpolation point
        cov_inv     -- inverted covariance matrix of X_t, X_t
        kernel_fun  -- kernel function for computing cross-covariance
        
       OUTPUT:
        T           -- symmetric matrix, obtained by interpolating T_t and reshaping
        
        NOTE: test that index_update is performed in-place 
    '''
    
    # interpolate
    cross_cov = cov_map(kernel_fun, x, X_t)
    t =  np.dot(cross_cov, cov_inv @ T_t).flatten()
    
    return flat2sym(t,X_t.shape[1])
    
    # reshape to symmetric in this order: diagonal, triu^, tril^
    # Note: 1. cannot outsource this code, because no way infer d       
    #          from array size in jax due to type casting issue
    #       2. hope that with jit-compilation, index_update is performed in-place!
#     d = X_t.shape[1]
#     ind_d = np.diag_indices(d)
#     ind_u = np.triu_indices(d, k=1, m=None)
#     ind_l = np.tril_indices(d, k=-1, m=None)
#     inds = np.hstack((ind_d[0],ind_u[0],ind_l[0])), np.hstack((ind_d[1],ind_u[1],ind_l[1]))
#     return index_update(np.zeros((d,d)), index[inds], np.hstack((t[:d],t[d:],t[d:])))



batch_triu_interp = jit(vmap(triu_interp, in_axes = (None, None, None, None, 0), out_axes = 0),
                        static_argnums = (3))


# @partial(jit, static_argnums = (1,))
# def triu2sym(t,d):
#     # this should have been named flat2sym. It takes a flat matrix and returns a symmetric one
#     ind_d = np.diag_indices(d)
#     ind_u = np.triu_indices(d, k=1, m=None)
#     ind_l = np.tril_indices(d, k=-1, m=None)
#     inds = np.hstack((ind_d[0],ind_u[0],ind_l[0])), np.hstack((ind_d[1],ind_u[1],ind_l[1]))
#     return index_update(np.zeros((d,d)), index[inds], np.hstack((t[:d],t[d:],t[d:])))

@partial(jit, static_argnums = (1,))
def flat2sym(t,d):
    # Takes a flat matrix and returns a symmetric one
    ind_d = np.diag_indices(d)
    ind_u = np.triu_indices(d, k=1, m=None)
    inds = np.hstack((ind_d[0],ind_u[0],ind_u[1])), np.hstack((ind_d[1],ind_u[1],ind_u[0]))
    return index_update(np.zeros((d,d)), index[inds], np.hstack((t[:d],t[d:],t[d:])))

@jit
def sym2flat(t):
#  strip diag + uppper triangular part of t and return it as a flat array
    d = t.shape[0]
    ind_d = np.diag_indices(d)
    ind_u = np.triu_indices(d, k=1, m=None)
    return np.array([*t[ind_d], *t[ind_u]])