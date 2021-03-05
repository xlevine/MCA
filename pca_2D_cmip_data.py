import sys, os
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from mpl_toolkits.basemap import Basemap 
from matplotlib.offsetbox import AnchoredText
from pylab import *
from os.path import expanduser
from load_cmip_data import load_input_var_2D, define_regional_mask, var_labels, compute_cell_area
from toolbox_cmip_data import convert_varglobal2D_to_vardomain1D, convert_vardomain1D_to_varglobal2D
from svd_cmip_data import pca_1D, return_output
from pca_cmip_data import compute_pca
home = expanduser("~")

# Principal Component analysis (PCA) is powerful tool to assess the dominant modes of climatic variability; PCA determines the spatial pattern of a climatological variable (e.g. precipitation, sea surface temperature) that can explain the largest fraction of its spatio-temporal variance (e.g. interannual covariance), globally or over a specific region. This is done by performing a singular value decomposition (SVD) on the covariance matrix of the variable. A PCA returns a left singular vectors, which correspond to the spatial pattern of the leading modes of variability. Associated with each Leading Mode is a timeseries (the eigenvalues). 

# Inputs of PCA: X_1[p,y,t], lat1[p,y], lev1[p,y], areas1[p,y], region1[p,y]
# Outputs of PCA: E1_N[p,y]; P_1[t] (for each leading mode)

###########################################################################
# METADATA: Contains name convention and data path of CMIP6. THIS MUST BE CHANGED FOR EACH USER. 
metadata_dict = {'expt_name':'historical-CMIP6','expt':'a1st','version':'v20200120','expt_name_path':'ssp245','model_type':'EC-Earth3','expt_folder':"/esarchive/exp/ecearth/",'file_path':"/cmorfiles/ScenarioMIP/EC-Earth-Consortium/",'sim_config':'r7i1p1f1','year_ini':2015,'miss_val':1.e20}
# INPUT: User defines the input of the MCA: variable name, season and region for each variable of the covariance pair
input_dict={'varname1':'ua','season1':'winter','region1':'Northern_Hemisphere','year_start':2015,'year_end':2020}

###################################################### I: Master Script ######################################################
def main():
    # P1: Initializing data
    [dataset1,surface_area1,mask_region1,lev1,lat1] = define_pca_2D_inputs()
    # P2: Performing PCA
    [varnorm,var1_modes_patterns,var1_modes_timeseries] = compute_pca(dataset1,surface_area1,mask_region1,lev1,lat1)
    # P3: Plotting
    var1_clim = np.nanmean(dataset1,axis=0)
    plot_pca_2D(varnorm,var1_modes_patterns,var1_modes_timeseries,lev1,lat1,var1_clim,input_dict)

###################################################### II: Main Dependence  ######################################################
## P1: Initializing data 
def define_pca_2D_inputs():
    year_ini = metadata_dict['year_ini']
    varname1 = input_dict['varname1']; season1 = input_dict['season1']; region1 = input_dict['region1']; year_start = input_dict['year_start']; year_end = input_dict['year_end']

    member_start = year_start - year_ini; member_end = year_end - year_ini
    ### Define 2D surface area mask matrix (NA ocean)
    [var1_2D,lev1,lat1] = load_input_var_2D(varname1,season1,region1,member_start,member_end) 
    area1 = compute_cell_area(varname1)
    region1 = define_regional_mask(varname1,region1)
    [lat1_2D, lev1_2D] = convert_lev2D(lat1,lev1)
    [area1_2D, region1_2D] = compute_2D_var(area1, region1, lev1)
    print(np.shape(var1_2D), np.shape(area1_2D), np.shape(region1_2D), np.shape(lat1_2D), np.shape(lev1_2D))

    return var1_2D, area1_2D, region1_2D, lev1_2D, lat1_2D

## P3: Plotting
def plot_pca_2D(varnorm,var1_modes_patterns,var1_modes_timeseries,lev1_2d,lat1_2d,var1_clim,input_dict):

    lev1_2d = np.multiply(lev1_2d,1.00000*np.ones(shape(lev1_2d))); lat1_2d = np.multiply(lat1_2d,1.00000*np.ones(shape(lat1_2d)))
    varname1 = input_dict['varname1']; season1 = input_dict['season1']; region1 = input_dict['region1']
    year_start = input_dict['year_start']; year_end = input_dict['year_end']

    var1_dic = var_labels(varname1)

    ### PLOT MODE PATTERNS AND TIME-SERIES
    N_trunc=2; frac_cticks = 0.6
    for N in np.arange(0,N_trunc,1):
        print('Leading mode ' +  str(N+1) + ' explains ' + str(int(varnorm[N,N]*100))  + '%' + ' of variance')
        var1_mode_patterns = np.squeeze(var1_modes_patterns[N,:,:])  
        var1_mode_timeseries = np.squeeze(var1_modes_timeseries[N,:])
        min_var1 = -np.nanmax(var1_modes_patterns)
        cticks_1 = np.arange(frac_cticks*min_var1,-frac_cticks*min_var1*(.99+2./6),-frac_cticks*2.*min_var1/6)

        # Plot left pattern
        title_name = 'Mode ' + str(int(N+1)) + ' Pattern for ' + var1_dic['long_name']  + ' in ' + season1 + ' (' + str(int(varnorm[N,N]*100))  + '%' + ' VAR) ' + '[' + var1_dic['units'] + ']' 
        figure_name = 'PCA' + str(int(N+1)) + '_pattern_' + varname1 + '_in_' + season1 + '_over_' + region1 + '_Y' + str(year_start) + 'Y' + str(year_end)
        plot_2D_contour(lev1_2d,lat1_2d,var1_mode_patterns,cticks_1,var1_clim,title_name,figure_name)

        # Plot left time series
        title_name = 'Mode ' + str(int(N+1)) + ' Time-Series for ' + var1_dic['long_name'] + ' in ' + season1 + ' (' + str(int(varnorm[N,N]*100))  + '%' + ' VAR)'
        figure_name = 'PCA' + str(int(N+1)) + '_series_' + varname1 + '_in_' + season1 + '_over_' + region1 + '_Y' + str(year_start) + 'Y' + str(year_end)
        plot_series(var1_mode_timeseries,year_start,year_end,title_name,figure_name)

############################################ III: Toolbox ##########################################################
### P1: Processing zonal-mean data 
def compute_2D_var(area1, region1, lev1):
    nlev = len(lev1); nlat = np.shape(area1)[0]
    area1_zmean = np.nanmean(area1,axis=-1) 
    region1_zmean = np.nanmean(region1,axis=-1) # region1 = 0 / 1
    region1_2D = []; area1_2D = []
    for k in np.arange(0,nlev,1):
        area1_2D.append(area1_zmean)
        region1_2D.append(region1_zmean)
    area1_2D = np.asarray(area1_2D)    
    region1_2D = np.asarray(region1_2D)

    dlev1 = delta_plev(lev1)
    dlev1_2D = np.matlib.repmat(dlev1,nlat,1).T    
    area1_2D = area1_2D * dlev1_2D

    return area1_2D, region1_2D

def convert_lev2D(lat,lev):
    nlat = np.shape(lat)[0]; nlev = len(lev)
    lat_2D = []; lev_2D = []
    for k in np.arange(0,nlev,1):
        lat_k = lat[:,0]
        lev_k = []
        for j in np.arange(0,nlat,1):
            lev_k.append(lev[k])
        lat_2D.append(lat_k)
        lev_2D.append(lev_k)
    lat_2D = np.asarray(lat_2D)
    lev_2D = np.asarray(lev_2D)
    
    return lat_2D, lev_2D

def delta_plev(plev):
    # define difference
    plev_f = np.ones(len(plev)+1)
    plev_f[1:-1] = np.add(plev[0:-1],plev[1:]) / 2.0
    plev_f[0] = 102450.0 
    plev_f[-1] = 0 
    dplev = diff(plev_f)

    return dplev

### P3: Plotting
def plot_2D_contour(lev_2d,lat_2d,var_2d,color_ticks,var_clim,title,savefigure_name):

    lev_2d = lev_2d / 100
    plt.rcParams.update({'font.size': 17})
    fwidth=8; fheight=8; margin = 0.6
    left_margin  =  1.0 / fwidth; 
    right_margin =  0.3 / fwidth;
    bottom_margin =  margin / fheight; 
    top_margin = margin / fheight
    l = left_margin    # horiz. position of bottom-left corner
    r = bottom_margin  # vert. position of bottom-left corner
    w = 1 - (left_margin + right_margin) # width of axes
    h = 1 - (bottom_margin + top_margin) # height of axes
    fig = plt.figure(figsize=(fwidth, fheight))
    ax = fig.add_axes([l, r, w, h])

    ######## anomalies
    color_map = 'RdBu_r'
    cs = plt.pcolormesh(lat_2d, lev_2d, var_2d, cmap=color_map, vmin=min(color_ticks), vmax=max(color_ticks))
    cbar = plt.colorbar(cs, ticks=color_ticks,pad=0.01)
    plt.clim(min(color_ticks),max(color_ticks))

    ######## climatology
    max_val = np.nanmax([np.nanmax(var_clim),-np.nanmin(var_clim)])
    nval =  2.*max_val/24
    levels_p = np.arange(nval, max_val, nval)
    levels_n = -levels_p[::-1]
    plt.contour(lat_2d, lev_2d, var_clim, levels=levels_n, linestyles='dashed', linewidths=1.0, colors='k') #, alpha=0)
    plt.contour(lat_2d, lev_2d, var_clim, levels=levels_p, linestyles='solid', linewidths=2.0, colors='k') # , alpha=0)

    color_ticks_label = []
    nticks = len(color_ticks)
    for n in np.arange(0,nticks,1):
        if (n % 2) == 0: 
            color_ticks_label.append(str(round(color_ticks[n],2)))
        else:
            color_ticks_label.append('') 
    print(color_ticks_label)    
    cbar.ax.set_yticklabels(color_ticks_label)
    plt.xlim((-30, 90)) 
    lat_label = ['-30$^{\circ}$', '', '0$^{\circ}$', '', '30$^{\circ}$', '', '60$^{\circ}$', '', '90$^{\circ}$']

    xlabel('Latitude')
    ylabel('Pressure Level [hPa]')
    plt.xticks(np.arange(-30, 105, 15), lat_label)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=8)
    save_file = savefigure_name + '.png'
    fig.savefig(save_file) #,op                                                                                 
    plt.close()

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
