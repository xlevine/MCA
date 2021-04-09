import sys, os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap 
from pylab import *
from os.path import expanduser
from load_cmip_data import load_input_var, define_regional_mask, var_labels, compute_cell_area
from toolbox_cmip_data import convert_varglobal2D_to_vardomain1D, convert_vardomain1D_to_varglobal2D
from svd_cmip_data import pca_1D, return_output
home = expanduser("~")

# Principal Component analysis (PCA) is powerful tool to assess the dominant modes of climatic variability; PCA determines the spatial pattern of a climatological variable (e.g. precipitation, sea surface temperature) that can explain the largest fraction of its spatio-temporal variance (e.g. interannual covariance), globally or over a specific region. This is done by performing a singular value decomposition (SVD) on the covariance matrix of the variable. A PCA returns a left singular vectors, which correspond to the spatial pattern of the leading modes of variability. Associated with each Leading Mode is a timeseries (the eigenvalues). 

# Inputs of PCA: X_1[x,y,t], lat1[x,y], lon1[x,y], areas1[x,y], region1[x,y]
# Outputs of PCA: E1_N[x,y]; P_1[t] (for each leading mode)

###########################################################################
# METADATA: Contains name convention and data path of CMIP6. THIS MUST BE CHANGED FOR EACH USER. 
metadata_dict = {'expt_name':'historical-CMIP6','expt':'a1st','version':'v20200120','expt_name_path':'ssp245','model_type':'EC-Earth3','expt_folder':"/esarchive/exp/ecearth/",'file_path':"/cmorfiles/ScenarioMIP/EC-Earth-Consortium/",'sim_config':'r7i1p1f1','year_ini':2015,'miss_val':1.e20}
# INPUT: User defines the input of the MCA: variable name, season and region for each variable of the covariance pair
input_dict={'varname1':'siconc','season1':'winter','region1':'TotalArc','year_start':2015,'year_end':2030}

###################################################### I: Master Script ######################################################
def main():
    # P1: Initializing data
    [dataset1,surface_area1,mask_region1,lon1,lat1] = define_pca_inputs()
    # P2: Performing PCA
    [varnorm,var1_modes_patterns,var1_modes_timeseries] = compute_pca(dataset1,surface_area1,mask_region1,lon1,lat1)
    # P3: Plotting
    plot_pca(varnorm,var1_modes_patterns,var1_modes_timeseries,lon1,lat1,input_dict)

###################################################### II: Main Dependence  ######################################################
## P1: Initializing data 
def define_pca_inputs():
    year_ini = metadata_dict['year_ini']
    varname1 = input_dict['varname1']; season1 = input_dict['season1']; region1 = input_dict['region1']; year_start = input_dict['year_start']; year_end = input_dict['year_end']

    member_start = year_start - year_ini; member_end = year_end - year_ini
    ### Define 2D surface area mask matrix 
    [var1,lon1,lat1] = load_input_var(varname1,season1,member_start,member_end) 
    area1 = compute_cell_area(varname1)
    region1 = define_regional_mask(varname1,region1)

    return var1, area1, region1, lon1, lat1

## P2: Performing MCA
def compute_pca(dataset1,area1,region1,lon1,lat1):
    # load 2-D yearly timeseries (x,y,t)
    ntime = shape(dataset1)[0]; nlon1 = shape(dataset1)[1]; nlat1 = shape(dataset1)[2] 
    var1_yearseries_2D = np.nan * ones((ntime,nlon1,nlat1)); var1_yearseries_2D = dataset1

    #### Convert from 2D to 1D and weight input by surface area mask matrix
    [var1_yearseries_domain_1D,var1_yearseries_fulldomain_1D,var1_total_mask] = convert_varglobal2D_to_vardomain1D(var1_yearseries_2D,area1,region1,lon1,lat1)

    [u, v, varnorm] = pca_1D(var1_yearseries_domain_1D)
    [u_2D, u_fulldomain_1D] = convert_vardomain1D_to_varglobal2D(u,var1_total_mask,area1)

    [var1_modes_timeseries,var1_modes_patterns] = return_output(u_2D,u_fulldomain_1D,u,var1_yearseries_fulldomain_1D)

    return varnorm,var1_modes_patterns,var1_modes_timeseries

## P3: Plotting
def plot_pca(varnorm,var1_modes_patterns,var1_modes_timeseries,lon1_2d,lat1_2d,input_dict):

    lon1_2d = np.multiply(lon1_2d,1.00000*np.ones(shape(lon1_2d))); lat1_2d = np.multiply(lat1_2d,1.00000*np.ones(shape(lat1_2d)))
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
        plot_contour(lon1_2d,lat1_2d,var1_mode_patterns,cticks_1,'notfilled',title_name,figure_name)

        # Plot left time series
        title_name = 'Mode ' + str(int(N+1)) + ' Time-Series for ' + var1_dic['long_name'] + ' in ' + season1 + ' (' + str(int(varnorm[N,N]*100))  + '%' + ' VAR)'
        figure_name = 'PCA' + str(int(N+1)) + '_series_' + varname1 + '_in_' + season1 + '_over_' + region1 + '_Y' + str(year_start) + 'Y' + str(year_end)
        plot_series(var1_mode_timeseries,year_start,year_end,title_name,figure_name)

############################################ III: Toolbox ##########################################################
### P3: Plotting
def plot_contour(lon_2d,lat_2d,var_2d,color_ticks,fill_continents,title,savefigure_name):
    lon_2d_off = np.subtract(lon_2d,360.0*np.ones(shape(lon_2d)))
    lon_2d_ext = np.concatenate((lon_2d_off,lon_2d),1)
    lat_2d_ext = np.concatenate((lat_2d,lat_2d),1)
    var_2d_ext = np.concatenate((var_2d,var_2d),1)
    var_dim = shape(var_2d_ext)
    len_lon = var_dim[1]
    len_lat = var_dim[0]

    color_map = 'RdBu_r'
    fig = plt.figure()
#    map = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='l') 
    map = Basemap(projection='npaeqd', boundinglat=50, lon_0=10, resolution='l')
    if fill_continents=='filled':
        map.fillcontinents(color='tan')

    map.drawcoastlines(linewidth=0.25)
    map.drawparallels(np.arange(-80.,81.,20.))
    map.drawmeridians(np.arange(-180.,181.,20.)) 
    xi, yi = map(lon_2d_ext,lat_2d_ext)

    cs = map.pcolormesh(xi,yi,var_2d_ext, cmap=color_map, vmin=min(color_ticks), vmax=max(color_ticks))
    cbar = plt.colorbar(cs, ticks=color_ticks) 
    plt.clim(min(color_ticks),max(color_ticks))        
    plt.title(title, fontsize=8)
    save_file_name = savefigure_name 
    save_file = save_file_name + '.png'
    fig.savefig(save_file)
    plt.close()

def plot_series(var_1d_yearseries,year_start,year_end,title,savefile_name):
    year_duration = shape(var_1d_yearseries)[0]; year_label = []; year_step = 20
    for t in np.arange(year_start, year_end+year_step, year_step):
        year_label.append(str(t))           
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
