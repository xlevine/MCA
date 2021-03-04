import sys, os
import numpy as np
from pylab import *
from os.path import expanduser
home = expanduser("~")

def convert_varglobal2D_to_vardomain1D(var_yearseries_2D,weight_area,mask_region,lon_2d,lat_2d):
    nvar = shape(var_yearseries_2D)
    ntime = nvar[0]
    len_lat = nvar[1]
    len_lon = nvar[2]

    # substract time-mean
    var_yearseries_2D = remove_timemean_yearseries_2D(var_yearseries_2D)
    
    # weight by surface area
    weight_yearseries_2D_l = []
    for t in np.arange(0,ntime,1):
        weight_yearseries_2D_l.append(weight_area)
    weight_yearseries_2D = np.asarray(weight_yearseries_2D_l)
    var_yearseries_2D = np.multiply(var_yearseries_2D,weight_yearseries_2D)

    [var_yearseries_domain_1D, var_yearseries_fulldomain_1D, var_total_mask] = convert_global2D_to_domain1D(var_yearseries_2D,mask_region)

    return var_yearseries_domain_1D, var_yearseries_fulldomain_1D, var_total_mask

def convert_vardomain1D_to_varglobal2D(var_domain_1D,var_total_mask,weight_area):
    var_fulldomain_1D_l = []
    for t in np.arange(0,shape(var_domain_1D)[1],1):
        var_fulldomain_1D_frame = np.zeros(shape(weight_area)[0]*shape(weight_area)[1])
        var_fulldomain_1D_frame[np.where(var_total_mask!=0)]=np.squeeze(var_domain_1D[:,t])
        var_fulldomain_1D_l.append(var_fulldomain_1D_frame)
    var_fulldomain_1D = np.asarray(var_fulldomain_1D_l)

    var_2D = convert_1D_to_2D(var_fulldomain_1D,weight_area) 
    var_2D_l = []
    for t in np.arange(0,shape(var_2D)[0],1):
        var_2D_frame = np.squeeze(var_2D[t,:,:])
        var_2D_norm = np.divide(var_2D_frame,weight_area)
        var_2D_l.append(var_2D_norm)
    var_2D = np.asarray(var_2D_l)

    return var_2D, var_fulldomain_1D

def convert_global2D_to_domain1D(var_yearseries_2D,mask_region):
    nvar = shape(var_yearseries_2D); ntime = nvar[0]; len_lat = nvar[1]; len_lon = nvar[2]
    # apply spatial masking
    var_yearseries_2D_l = []
    for t in np.arange(0,ntime,1):
        var_timeframe_2D = []
        var_timeframe_2D = np.squeeze(var_yearseries_2D[t,:,:])
        var_timeframe_2D = np.multiply(var_timeframe_2D,mask_region)
        var_yearseries_2D_l.append(var_timeframe_2D)
    var_yearseries_2D = np.asarray(var_yearseries_2D_l)

    var_yearseries_fulldomain_1D_l = []
    for t in np.arange(0,ntime,1):
        var_2D_into_1D = np.array(np.squeeze(var_yearseries_2D[t,:,:])).flatten()
        var_yearseries_fulldomain_1D_l.append(var_2D_into_1D)

    var_yearseries_fulldomain_1D = np.asarray(var_yearseries_fulldomain_1D_l)
    var_yearseries_fulldomain_1D[np.where(np.isnan(var_yearseries_fulldomain_1D))]=0
    var_total_mask_l = np.ones(shape(var_yearseries_fulldomain_1D)[1])
    for t in np.arange(0,ntime,1):
        var_mask_l = np.zeros(shape(var_yearseries_fulldomain_1D)[1]) 
        var_yearseries_fulldomain_1D_timeframe = []
        var_yearseries_fulldomain_1D_timeframe = np.squeeze(var_yearseries_fulldomain_1D[t,:]) 
        var_mask_l[np.where(var_yearseries_fulldomain_1D_timeframe!=0)] = 1.0  
        var_total_mask_l = np.multiply(var_mask_l,var_total_mask_l)
    var_total_mask = np.asarray(var_total_mask_l)
 
    var_yearseries_domain_1D_l = []     
    for t in np.arange(0,ntime,1):
        var_yearseries_fulldomain_1D_timeframe = []
        var_yearseries_fulldomain_1D_timeframe_tr = []
        var_yearseries_fulldomain_1D_timeframe = np.squeeze(var_yearseries_fulldomain_1D[t,:])
        var_yearseries_fulldomain_1D_timeframe_tr = var_yearseries_fulldomain_1D_timeframe[np.where(var_total_mask!=0)]
        var_yearseries_domain_1D_l.append(var_yearseries_fulldomain_1D_timeframe_tr)        
    var_yearseries_domain_1D = np.asarray(var_yearseries_domain_1D_l)

    return var_yearseries_domain_1D, var_yearseries_fulldomain_1D, var_total_mask

def convert_1D_to_2D(var_1D,weight_2D):
    # Convert global 1D to global 2D
    len_lat=shape(weight_2D)[0]; len_lon=shape(weight_2D)[1]; nVar1D=shape(var_1D); nComp=nVar1D[0]
    var_2D_list=[]
    for t in np.arange(0,nComp,1):
        var_2D_timeframe=[] 
        for j in np.arange(0,len_lat,1):
            i_s=j*len_lon
            i_e=(j+1)*len_lon
            row=np.squeeze(var_1D[t,i_s:i_e])
            var_2D_timeframe.append(row)
        var_2D_list.append(var_2D_timeframe) 
    var_2D = np.asarray(var_2D_list)     

    return var_2D

def remove_timemean_yearseries_2D(var1_yearseries_2D):
    nvar = shape(var1_yearseries_2D); ntime = nvar[0]
    var1_timemean_2D = np.mean(var1_yearseries_2D, axis=0)
    var1_timemean_yearseries_2D_l = []
    for t in np.arange(0,ntime,1):
        var1_timemean_yearseries_2D_l.append(var1_timemean_2D)
    var1_timemean_yearseries_2D = np.asarray(var1_timemean_yearseries_2D_l)
    var1_yearseries_anom_2D = np.subtract(var1_yearseries_2D,var1_timemean_yearseries_2D)

    return var1_yearseries_anom_2D 
