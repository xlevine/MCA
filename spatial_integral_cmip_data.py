import sys, os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap 
from pylab import *
from os.path import expanduser
from load_cmip_data import load_input_var, define_regional_mask, compute_cell_area, var_labels
from toolbox_cmip_data import convert_varglobal2D_to_vardomain1D, convert_vardomain1D_to_varglobal2D
from svd_cmip_data import mca_1D, return_output
home = expanduser("~")

# METADATA: Contains name convention and data path of CMIP6. THIS MUST BE CHANGED FOR EACH USER. 
metadata_dict = {'expt_name':'historical-CMIP6','expt':'a1st','version':'v20200120','expt_name_path':'ssp245','model_type':'EC-Earth3','expt_folder':"/esarchive/exp/ecearth/",'file_path':"/cmorfiles/ScenarioMIP/EC-Earth-Consortium/",'sim_config':'r7i1p1f1','year_ini':2015,'miss_val':1.e20}
# INPUT: User defines the input of the MCA: variable name, season and region for each variable of the covariance pair
input_dict={'varname1':'siconc','season1':'winter','region1':'TotalArc','year_start':2015,'year_end':2030}
###################################################### I: Master Script ######################################################
def main():
    # P1: Initializing data
    [dataset1,surface_area1,mask_region1,lon1,lat1] = define_integral_inputs()
    # P2: Performing Spatial Integral
    [data_avg1, area_sum1] = compute_spatial_integral(dataset1,surface_area1,mask_region1,lon1,lat1)
    # P3: Plotting
    plot_integral(data_avg1,input_dict)

###################################################### II: Main Dependence  ######################################################
## P1: Initializing data 
def define_integral_inputs():
    year_ini = metadata_dict['year_ini']
    varname1 = input_dict['varname1']; season1 = input_dict['season1']; region1 = input_dict['region1']; year_start = input_dict['year_start']; year_end = input_dict['year_end']

    member_start = year_start - year_ini; member_end = year_end - year_ini
    ### Define 2D surface area mask matrix 
    [var1,lon1,lat1] = load_input_var(varname1,season1,member_start,member_end) 
    area1 = compute_cell_area(varname1)
    region1 = define_regional_mask(varname1,region1)

    return var1, area1, region1, lon1, lat1

## P2: Performing Spatial Integral 
def compute_spatial_integral(dataset1,surface_area1,mask_region1):

    # load 2-D yearly timeseries (x,y,t)
    ntime = shape(dataset1)[0]; nlon1 = shape(dataset1)[1]; nlat1 = shape(dataset1)[2] 
    var1_yearseries_2D = np.nan * ones((ntime,nlon1,nlat1)); var1_yearseries_2D = dataset1

    mask_area1 = np.multiply(surface_area1,mask_region1)
    data_mask_area1 = np.multiply(dataset1,mask_area1)
    data_mask_area_sum1 = np.squeeze(np.nansum(np.squeeze(np.nansum(data_mask_area1,axis=2)),axis=1))
    area_sum1 = np.squeeze(np.nansum(np.squeeze(np.nansum(mask_area1,axis=1)),axis=0))    

    data_avg1 = data_mask_area_sum1 / area_sum1

    return data_avg1, area_sum1

## P3: Plotting
def plot_integral(var1_timeseries,input_dict):

    varname1 = input_dict['varname1']; season1 = input_dict['season1']; region1 = input_dict['region1']
    year_start = input_dict['year_start']; year_end = input_dict['year_end']

    var1_dic = var_labels(varname1)

    ### PLOT MODE PATTERNS AND TIME-SERIES
    var1_timeseries = np.squeeze(var1_timeseries)

    # Plot time series
    title_name = 'Time-Series for ' + var1_dic['long_name'] + ' in ' + season1 + ' over ' + region1
    figure_name = 'Time_series_' + varname1 + '_in_' + season1 + '_over_' + region1 + '_Y' + str(year_start) + 'Y' + str(year_end)
    plot_series(var1_timeseries,year_start,year_end,title_name,figure_name)

############################################ III: Toolbox ##########################################################
def plot_series(var_1d_yearseries,year_start,year_end,title,savefile_name):
    year_duration = shape(var_1d_yearseries)[0]; year_label = []; year_step = 20
    for t in np.arange(year_start, year_end+year_step, year_step):
        year_label.append(str(t))           
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure()                                                          
    plt.plot(np.arange(1, year_duration+1, 1), var_1d_yearseries, 'r')
    plt.plot(np.arange(1, year_duration+1, 1), var_1d_yearseries, 'ro')
    plt.plot(np.arange(1, year_duration+1, 1), np.zeros(shape(var_1d_yearseries)), 'k--')
    plt.xticks(np.arange(1, year_duration+year_step, year_step), year_label)
    plt.xlabel('Year')
    plt.title(title, fontsize=8)
    plt.show()
    fig.savefig(savefile_name+'.png') #,optimize=True,quality=85)
    plt.close()
############## EOF ##############
