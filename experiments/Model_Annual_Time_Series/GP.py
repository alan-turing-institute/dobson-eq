import jax.numpy as np
import jax.random as random
from jax.scipy.linalg import cholesky
from jax import jit, vmap
from functools import partial


sq_exp = jit(lambda x, y, l, v: v*np.exp(-.5*np.power((x-y)/l,2)))

# compute covariance matrix for xs, xs2
@partial(jit, static_argnums=(0))
def cov_map(cov_func, xs, xs2 = None):
    # xs and xs2 are stacked along the leading dimension
    if xs2 is None:
        return vmap(lambda x:  vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
        return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T

# squared exponential kernel with diagonal noise term
def cov_matr(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):
    """
    Return covariance matrix of size X x Z
    """    
    return cov_map(partial(sq_exp, l = length, v = var), X, Z) + \
                (noise*include_noise + jitter) * np.eye(X.shape[0], Z.shape[0])

@partial(jit, static_argnums=(1))   
def cov_inv_matr(X_t, kernel_fun, observation_noise, eps = 1e-7):
    # inverse covariance matrix of kernel evaluated on nodes X_t
    # eps is our nugget to avoid non-psd covariance
    
    Nt = X_t.shape[0] # number of nodes for RKHS matrix interpolation
    train_cov =  cov_map(kernel_fun, X_t) + np.eye(Nt) * eps + np.eye(X_t.size)*observation_noise
    
    chol = cholesky(train_cov, lower=True)
    c = np.linalg.inv(chol); 
    return np.dot(c.T,c)

def predictGP(x, y, t, kern, observation_noise):
    # prediction from a GP with kernel function kern(x,y)
    # x, y - training data
    # t - test points
    # noise - observation noise
    # kern - kernel fun of (x,y), e.g. partial(sq_exp, l = length, v = var)
    
    # returns mean and covariance matrix of GP posterior, evaluated at t
    
    cov_inv = cov_inv_matr(x, kern, observation_noise) 
    
    cross_cov = cov_map(kern, t, x)
    
    K = cross_cov @ cov_inv
    
    # GP posterior mean at t
    posterior_mean = K @ y
    
    # GP posterior covariance at t
    posterior_cov = cov_map(kern, t) - K @ cross_cov.T
 
    return posterior_mean, posterior_cov


def uncrt(key, posterior_mean, posterior_cov):
    # simple representation of uncertainty: draw
    # Gaussian random noise with posterior variance around posterior mean
    # E.G. 
    #    predictions = uncrt(....)
    #    percentiles = onp.percentile(predictions, [5.0, 95.0], axis=0)
    #    ax.fill_between(X_test, percentiles[0, :], percentiles[1, :], color="lightblue")
    
    sigma_noise = np.sqrt(np.clip(np.diag(posterior_cov), a_min=0.0)) \
    * random.normal(key, posterior_mean.shape)
    
    return posterior_mean+sigma_noise