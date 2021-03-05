import sys, os
import numpy as np
from pylab import *
from sklearn.decomposition import TruncatedSVD
from os.path import expanduser
home = expanduser("~")

def pca_1D(var1_yearseries_domain_1D):
    # Compute SVD of covariance matrix
    var1_yearseries_domain_1D_T = np.transpose(var1_yearseries_domain_1D)  

    # 50 components truncation
    u, Sigma, v = compute_eof_modes(var1_yearseries_domain_1D_T,50)
    # compute squared covariance fraction
    s = np.diag(Sigma)
    varnorm = np.power(s,2)/np.sum(np.power(s,2))

    return u, v, varnorm 

def mca_1D(var1_yearseries_domain_1D,var2_yearseries_domain_1D):
    # Compute SVD of covariance matrix
    var1_yearseries_domain_1D_T = np.transpose(var1_yearseries_domain_1D)  
    var1_var2_yearseries_covar_1D = np.dot(var1_yearseries_domain_1D_T,var2_yearseries_domain_1D)
    var1_var2_yearseries_covar_1D = np.divide(var1_var2_yearseries_covar_1D,shape(var2_yearseries_domain_1D)[0]-1)

    # 50 components truncation
    u, Sigma, v = compute_eof_modes(var1_var2_yearseries_covar_1D,50)
    # compute squared covariance fraction
    s = np.diag(Sigma)
    varnorm = np.power(s,2)/np.sum(np.power(s,2))

    return u, v, varnorm 

def compute_eof_modes(var_2D,PC_comp):
    nComp = min(min(var_2D.shape)-1,PC_comp)
    var_pca = TruncatedSVD(n_components=nComp)
    var_pca.fit(var_2D)
    v_T = var_pca.components_
    v = np.transpose(v_T)
    u = var_pca.transform(var_2D)
    Sigma = var_pca.singular_values_
    inv_s = np.linalg.inv(np.diag(Sigma)) 
    u = np.dot(u,inv_s)

    return u, Sigma, v

def return_output(u_2D,u_fulldomain_1D,u,var_yearseries_fulldomain_1D):

    # Compute time series
    var_yearseries_full_1D = np.transpose(var_yearseries_fulldomain_1D)
    var_modes_timeseries_1D = np.dot(u_fulldomain_1D,var_yearseries_full_1D)
    std_var_modes = np.transpose(np.std(var_modes_timeseries_1D,axis=1))

    # Compute spatial patterns
    var_modes_patterns = []
    var_modes_timeseries = []
    for t in np.arange(0,shape(std_var_modes)[0],1):        
        var_modes_patterns_frame = np.multiply(np.squeeze(u_2D[t,:,:]), std_var_modes[t])
        var_modes_timeseries_frame = np.divide(np.squeeze(var_modes_timeseries_1D[t]), std_var_modes[t])
        var_modes_patterns.append(var_modes_patterns_frame)
        var_modes_timeseries.append(var_modes_timeseries_frame)
    var_modes_patterns = np.asarray(var_modes_patterns); var_modes_timeseries = np.asarray(var_modes_timeseries)
     
    # Adapt sign convention
    var_modes_norm = np.squeeze(np.sum(u,axis=0))
    nComp_var = shape(u)[1]
    for k in np.arange(0,nComp_var,1):
        if var_modes_norm[k]<0:
            var_modes_timeseries[k,:] = - var_modes_timeseries[k,:]
            var_modes_patterns[k,:,:] = - var_modes_patterns[k,:,:]
   
    return var_modes_timeseries,var_modes_patterns
