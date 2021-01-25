import sys, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np
import time
from mpl_toolkits.basemap import Basemap,maskoceans
from netCDF4 import Dataset, num2date, date2num
from pylab import *
from sklearn import decomposition
from sklearn.decomposition import TruncatedSVD
from os.path import expanduser
home = expanduser("~")

# Maximum covariance analysis (MCA) is powerful tool to study remote connection between climatic variables [see Bretherton et al., 1992; Wallace et al., 1992, for a description of MCA as an exploratory method]; MCA determines the spatial pattern of 2 climatological variables (e.g. precipitation and sea surface temperature) that can explain the largest fraction of their covariance (e.g. interannual covariance). In particular, it can be used to find the dominant mode of interaction between those 2 variables globally or over a specific region. This is done by performing a singular value decomposition (SVD) on the covariance matrix of the 2 variables. A MCA returns left (right) singular vectors, which correspond to the spatial pattern of the leading modes of intermodel variability in the first (second) variable that is most correlated with the interannual variability of the second (first) variable (the eigenvectors). Associated with each Leading Mode is a timeseries (the eigenvalues). 

# Inputs of MCA: X_1[x,y,t], X_2[x,y,t], lat1[x,y], lat2[x,y], lon1[x,y], lon2[x,y], areas1[x,y], areas2[x,y], region1[x,y], region2[x,y]
# Outputs of MCA: E1_N[x,y]; E2_N[x,y]; P_1[t]; P_2[t] (for each leading modes)

###########################################################################
# LISTS: Defines name of variables (following CMIP6 convention) and seasons 
month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
season_dict = {'winter':['dec','jan','feb'],'summer':['jun','jul','aug'],'spring':['mar','apr','may'],'fall':['sep','oct','nov']}
varlist_Omon = ['msftyz', 'mlotst', 'evs','friver','hfds','mlotst','sos','vmo','ficeberg2d','rsntds','nshfls','downempfluxoce','downnoswfluxoce','downswfluxoce', 'thetao', 'so', 'tos']
varlist_SImon = ['siage','sithick','sialb','siconc','sidmassevapsubl','sisnthick','sitemptop','sivol','sistrxdtop','downempfluxice','downnoswfluxice','downswfluxice','freshwater','siheco','sinudg','sstnudg','totdownfluxice','vfxice','vfxsnw']

# METADATA: Contains name convention and data path of CMIP6. THIS MUST BE CHANGED FOR EACH USER. 
metadata_dict = {'expt_name':'historical-CMIP6','expt':'a1st','version':'v20200120','expt_name_path':'ssp245','model_type':'EC-Earth3','expt_folder':"/esarchive/exp/ecearth/",'file_path':"/cmorfiles/ScenarioMIP/EC-Earth-Consortium/",'sim_config':'r7i1p1f1','year_ini':2015}
mask_dict = {'mask_path_ocean':"/esarchive/autosubmit/con_files/",'mask_name_ocean':'mask.regions.Ec3.2_O1L75.nc','mask_path_atmos':"/esarchive/exp/ecearth/constant/",'mask_name_atmos':'mask.regions.ifs_512x256.nc'}
area_dict={'area_root_path':"/esarchive/scratch/xlevine/weight_area/",'file_gridcell_ocean':'areacello_LR.nc','var_gridcell_ocean':'areacello','file_gridcell_atmos':'weight_area_atmos_LR.nc','var_gridcell_atmos':'cell_area'}

# INPUT: User defines the input of the MCA: variable name, season and region for each variable of the covariance pair
input_dict={'varname1':'tas','season1':'winter','region1':'TotalArc','varname2':'psl','season2':'winter','region2':'BarKara','year_start':2015,'year_end':2055}

###################################################### I: Master Script ######################################################
def main():
    # P1: Initializing data 
    [dataset1,dataset2,surface_area1,surface_area2,mask_region1,mask_region2,lon1,lon2,lat1,lat2] = define_mca_inputs()
    # P2: Performing MCA
    [varnorm,var1_modes_patterns,var1_modes_timeseries,var2_modes_patterns,var2_modes_timeseries] = compute_mca(dataset1,dataset2,surface_area1,surface_area2,mask_region1,mask_region2,lon1,lon2,lat1,lat2)
    # P3: Plotting
    plot_mca(varnorm,var1_modes_patterns,var1_modes_timeseries,var2_modes_patterns,var2_modes_timeseries,lon1,lat1,lon2,lat2)

###################################################### II: Main Dependence  ######################################################
## P1: Initializing data 
def define_mca_inputs():
    varname1 = input_dict['varname1']; varname2 = input_dict['varname2']; season1 = input_dict['season1']; season2 = input_dict['season2']; region1 = input_dict['region1']; region2 = input_dict['region2']; year_start = input_dict['year_start']; year_end = input_dict['year_end']

    [var1,var2,lon1,lon2,lat1,lat2,area1,area2] = load_input_vars(varname1,varname2,season1,season2,year_start,year_end) 

    ### Define 2D surface area mask matrix 
    region1 = define_regional_mask(varname1,region1); region2 = define_regional_mask(varname2,region2)

    return var1, var2, area1, area2, region1, region2, lon1, lon2, lat1, lat2

## P2: Performing MCA
def compute_mca(dataset1,dataset2,area1,area2,region1,region2,lon1,lon2,lat1,lat2):
    # load 2-D yearly timeseries (x,y,t)
    ntime = max(shape(dataset1)[0],shape(dataset2)[0])
    nlon1 = shape(dataset1)[1]; nlat1 = shape(dataset1)[2] 
    nlon2 = shape(dataset2)[1]; nlat2 = shape(dataset2)[2] 

    var1_yearseries_2D = np.nan * ones((ntime,nlon1,nlat1)); var1_yearseries_2D = dataset1
    var2_yearseries_2D = np.nan * ones((ntime,nlon2,nlat2)); var2_yearseries_2D = dataset2    

    #### Convert from 2D to 1D and weight input by surface area mask matrix
    [var1_yearseries_domain_1D,var1_yearseries_fulldomain_1D,var1_total_mask] = convert_varglobal2D_to_vardomain1D(var1_yearseries_2D,area1,region1,lon1,lat1)
    [var2_yearseries_domain_1D,var2_yearseries_fulldomain_1D,var2_total_mask] = convert_varglobal2D_to_vardomain1D(var2_yearseries_2D,area2,region2,lon2,lat2)

    [u, v, var1_total_mask, var2_total_mask, varnorm] = mca_1D(var1_yearseries_domain_1D,var2_yearseries_domain_1D,var1_total_mask,var2_total_mask)

    [u_2D,var1_modes_timeseries,var1_modes_patterns] = return_output(u,var1_yearseries_fulldomain_1D,var1_total_mask,area1,lat1)
    [v_2D,var2_modes_timeseries,var2_modes_patterns] = return_output(v,var2_yearseries_fulldomain_1D,var2_total_mask,area2,lat2)

    return varnorm,var1_modes_patterns,var1_modes_timeseries,var2_modes_patterns,var2_modes_timeseries
## P3: Plotting
def plot_mca(varnorm,var1_modes_patterns,var1_modes_timeseries,var2_modes_patterns,var2_modes_timeseries,lon1_2d,lat1_2d,lon2_2d,lat2_2d):

    lon1_2d = np.multiply(lon1_2d,1.00000*np.ones(shape(lon1_2d))); lat1_2d = np.multiply(lat1_2d,1.00000*np.ones(shape(lat1_2d)))
    lon2_2d = np.multiply(lon2_2d,1.00000*np.ones(shape(lon2_2d))); lat2_2d = np.multiply(lat2_2d,1.00000*np.ones(shape(lat2_2d)))
    varname1 = input_dict['varname1']; season1 = input_dict['season1']; region1 = input_dict['region1']
    varname2 = input_dict['varname2']; season2 = input_dict['season2']; region2 = input_dict['region2']
    year_start = input_dict['year_start']; year_end = input_dict['year_end']

    ### PLOT MODE PATTERNS AND TIME-SERIES
    N_trunc=1; frac_cticks = 0.6
    for N in np.arange(0,N_trunc,1):
        print('Leading mode ' +  str(N+1) + ' explains ' + str(int(varnorm[N,N]*100))  + '%' + ' of squared covariance')
        var1_mode_patterns = np.squeeze(var1_modes_patterns[N,:,:])  
        var1_mode_timeseries = np.squeeze(var1_modes_timeseries[N,:])
        var2_mode_patterns = np.squeeze(var2_modes_patterns[N,:,:])
        var2_mode_timeseries = np.squeeze(var2_modes_timeseries[N,:])
        min_var1 = -np.nanmax(var1_modes_patterns)
        cticks_1 = np.arange(frac_cticks*min_var1,-frac_cticks*min_var1*(.99+2./6),-frac_cticks*2.*min_var1/6)
        min_var2 = -np.nanmax(var2_modes_patterns)
        cticks_2 = np.arange(frac_cticks*min_var2,-frac_cticks*min_var2*(.99+2./6),-frac_cticks*2.*min_var2/6)

        # Plot left pattern
        title_name = 'Lead Mode ' + str(int(N+1)) + ' Pattern for ' + varname1 + ' in ' + season1 + ' (' + str(int(varnorm[N,N]*100))  + '%' + ' Squared COV)'
        figure_name = 'MCA' + str(int(N+1)) + '_pattern_' + varname1 + '_in_' + season1 + '_over_' + region1 + '_with_' + varname2 + '_in_' + season2 + '_over_' + region2 + '_on_' + varname1 + '_Y' + str(year_start) + 'Y' + str(year_end)
        contour_function(lon1_2d,lat1_2d,var1_mode_patterns,cticks_1,'notfilled',title_name,figure_name)

        # Plot right pattern
        title_name = 'Lead Mode ' + str(int(N+1)) + ' Pattern for ' + varname2 + ' in ' + season2 + ' (' + str(int(varnorm[N,N]*100))  + '%' + ' Squared COV)'
        figure_name = 'MCA' + str(int(N+1)) + '_pattern_' + varname1 + '_in_' + season1 + '_over_' + region1 + '_with_' + varname2 + '_in_' + season2 + '_over_' + region2 + '_on_' + varname2 + '_Y' + str(year_start) + 'Y' + str(year_end)
        contour_function(lon2_2d,lat2_2d,var2_mode_patterns,cticks_2,'notfilled',title_name,figure_name)

        # Plot left time series
        title_name = 'Lead Mode ' + str(int(N+1)) + ' Time-Series for ' + varname1 + ' in ' + season2 + ' (' + str(int(varnorm[N,N]*100))  + '%' + ' Squared COV)'
        figure_name = 'MCA' + str(int(N+1)) + '_series_' + varname1 + '_in_' + season1 + '_over_' + region1 + '_with_' + varname2 + '_in_' + season2 + '_over_' + region2 + '_on_' + varname1 + '_Y' + str(year_start) + 'Y' + str(year_end)
        plot_interannualseries(var1_mode_timeseries,year_start,year_end,title_name,figure_name, '[]') #,'[W m-2]')

        # Plot right time series
        title_name = 'Lead Mode ' + str(int(N+1)) + ' Time-Series for ' + varname2 + ' in ' + season2 + ' (' + str(int(varnorm[N,N]*100))  + '%' + ' Squared COV)'
        figure_name = 'MCA' + str(int(N+1)) + '_series_' + varname1 + '_in_' + season1 + '_over_' + region1 + '_with_' + varname2 + '_in_' + season2 + '_over_' + region2 + '_on_' + varname2 + '_Y' + str(year_start) + 'Y' + str(year_end)
        plot_interannualseries(var2_mode_timeseries,year_start,year_end,title_name,figure_name, '[]') #,'[W m-2]')

############################################ III: Toolbox ##########################################################
### P1: Initializing data
def define_regional_mask(varname,region):
    if ((varname in varlist_Omon) or (varname in varlist_SImon)):
        path = mask_dict['mask_path_ocean']
        region_mask_name = mask_dict['mask_name_ocean']
    else:
        path = mask_dict['mask_path_atmos']
        region_mask_name = mask_dict['mask_name_atmos']

    mask_path = path + region_mask_name
    var_file = Dataset(mask_path, 'r')
    mask_region_3D = var_file.variables[region]
    mask_region_3D = np.asarray(mask_region_3D)

    if ((varname in varlist_Omon) or (varname in varlist_SImon)):
        mask_region = np.squeeze(mask_region_3D[0,0,:,:])
    else:
        mask_region = mask_region_3D  
    mask_region = np.asarray(mask_region)

    return mask_region

def load_input_vars(varname1, varname2, season1, season2, year_start, year_end):
    nyears = year_end - year_start + 1; nmemb = nmemb=np.arange(1,nyears+1,1) 
    var1 = []; var2 = []
    for memb in nmemb:
        print(memb)
        [var1_memb, lon1, lat1, area1] = load_var(memb,varname1,season1)
        [var2_memb, lon2, lat2, area2] = load_var(memb,varname2,season2)
        var1.append(var1_memb)
        var2.append(var2_memb)
    var1 = np.asarray(var1); var2 = np.asarray(var2)
        
    return var1, var2, lon1, lon2, lat1, lat2, area1, area2

def load_var(num_member, varname, month_name):    
    if month_name in month_list:
        [var, lon_2d, lat_2d, area] = read_var_month(num_member, varname, month_name)
    else:
        var_season = []
        for month in season_dict[month_name]: 
            [var_month, lon_2d, lat_2d, area] = read_var_month(num_member, varname, month)
            var_season.append(var_month)
        var_season = np.asarray(var_season)
        var = np.nanmean(var_season,axis=0)
        
    return var, lon_2d, lat_2d, area

def read_var_month(num_member, varname, month_name):
    month_index = get_timestamp(month_name)
    [var_year, lon_2d, lat_2d] = read_var(num_member, varname)
    var = np.squeeze(var_year[int(month_index),:,:])

    if ((varname in varlist_Omon) or (varname in varlist_SImon)):
        area = compute_cell_area('ocean')
    else:
        area = compute_cell_area('atmos')

    return var, lon_2d, lat_2d, area

    var_file.close()

def compute_cell_area(domain):
    root_path=area_dict['area_root_path']
    if domain=='ocean':
        file_gridcell=root_path + area_dict['file_gridcell_ocean']
        gridcell_file = Dataset(file_gridcell, 'r')
        weight_area = gridcell_file.variables[area_dict['var_gridcell_ocean']]     
    else:
        file_gridcell=root_path + area_dict['file_gridcell_atmos']
        gridcell_file = Dataset(file_gridcell, 'r')
        weight_area = gridcell_file.variables[area_dict['var_gridcell_atmos']]     
    
    weight_area= np.asarray(weight_area)
    weight_area[np.where(weight_area>=1.e20)]=np.nan
    weight_area = np.divide(weight_area,np.nansum(weight_area)*np.ones(shape(weight_area)))

    return weight_area

def read_var(num_member, varname):
    sim_array = get_filename(num_member, varname)
    var = []
    for k in np.arange(0,len(sim_array),1):
        var_file = Dataset(sim_array[k], 'r')
        var_k = var_file.variables[varname]
        var.extend(var_k)
    var=np.asarray(np.squeeze(var))

    if ((varname in varlist_Omon) or (varname in varlist_SImon)):
        lon_2d = var_file.variables['longitude']; lat_2d = var_file.variables['latitude']
        lon_2d = np.asarray(lon_2d); lat_2d = np.asarray(lat_2d) 
        lon_2d_pos = np.add(lon_2d,180.0*np.ones(shape(lon_2d)))
        lon_2d[np.where(lon_2d<0)] = lon_2d_pos[np.where(lon_2d<0)]
    else:
        lon = var_file.variables['lon']; lat = var_file.variables['lat']
        lon_2d = np.matlib.repmat(lon,len(lat),1); lat_2d_T = np.matlib.repmat(lat,len(lon),1)
        lat_2d = lat_2d_T.T
        lon_2d = np.asarray(lon_2d); lat_2d = np.asarray(lat_2d) 

    var[np.where(var>=1.0e20)] = np.nan
    return var, lon_2d, lat_2d

def get_filename(num_member, varname):
    if varname in varlist_Omon:
        domain='Omon'; sim_type='gn'
    elif varname in varlist_SImon:
        domain='SImon'; sim_type='gn'
    else:
        domain='Amon'; sim_type='gr'

    expt=metadata_dict['expt']; version=metadata_dict['version']; expt_name_path=metadata_dict['expt_name_path']; model_type=metadata_dict['model_type']; year_ini=metadata_dict['year_ini']; expt_folder=metadata_dict['expt_folder']; file_path=metadata_dict['file_path']; sim_config=metadata_dict['sim_config']

    slash="/"; udscr='_'; dash='-'; end_file_nc='.nc'
    year_membr = year_ini + num_member - 1; date_year = [str(year_membr) + '01' + dash + str(year_membr) + '12']
    varfile_path = file_path + model_type + slash
    varname_expt_root = varname + udscr + domain + udscr + model_type + udscr + expt_name_path + udscr + sim_config + udscr + sim_type
    root_path = expt_folder + expt + varfile_path + expt_name_path + slash + sim_config + slash + domain + slash + varname + slash + sim_type + slash + version

    sim_array = []
    for k in np.arange(0,len(date_year),1):
        varname_expt_file = varname_expt_root + udscr + date_year[k] + end_file_nc        
        sim = root_path + slash + varname_expt_file
        sim_array.append(sim)        
    sim_array = np.asarray(sim_array)

    return sim_array

def get_timestamp(month_name):
    month_l = [i for i, s in enumerate(month_list) if month_name in s]
    month = month_l[0] + 1
    month_index = int(month) - 1

    return month_index

### P2: Performing MCA
def mca_1D(var1_yearseries_domain_1D,var2_yearseries_domain_1D,var1_total_mask,var2_total_mask):

    # Compute SVD of covariance matrix
    var1_yearseries_domain_1D_T = np.transpose(var1_yearseries_domain_1D)  
    var1_var2_yearseries_covar_1D = np.dot(var1_yearseries_domain_1D_T,var2_yearseries_domain_1D)
    var1_var2_yearseries_covar_1D = np.divide(var1_var2_yearseries_covar_1D,shape(var2_yearseries_domain_1D)[0]-1)

    # 50 components truncation
    u, Sigma, v = compute_eof_modes(var1_var2_yearseries_covar_1D,50)
    # compute squared covariance fraction
    s = np.diag(Sigma)
    varnorm = np.power(s,2)/np.sum(np.power(s,2))

    return u, v, var1_total_mask, var2_total_mask, varnorm 

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

def return_output(u,var_yearseries_fulldomain_1D,var_total_mask,weight_area,lat_2d):
    # Convert local 1D to global 1D

    u_2D, u_fulldomain_1D = convert_vardomain1D_to_varglobal2D(u,var_total_mask,weight_area,lat_2d)

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
   
    return u_2D,var_modes_timeseries,var_modes_patterns

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

def convert_vardomain1D_to_varglobal2D(var_domain_1D,var_total_mask,weight_area,lat_2D):
    var_fulldomain_1D_l = []
    for t in np.arange(0,shape(var_domain_1D)[1],1):
        var_fulldomain_1D_frame = np.zeros(shape(lat_2D)[0]*shape(lat_2D)[1])
        var_fulldomain_1D_frame[np.where(var_total_mask!=0)]=np.squeeze(var_domain_1D[:,t])
        var_fulldomain_1D_l.append(var_fulldomain_1D_frame)
    var_fulldomain_1D = np.asarray(var_fulldomain_1D_l)

    var_2D = convert_1D_to_2D(var_fulldomain_1D,lat_2D) 

    var_2D_l = []
    for t in np.arange(0,shape(var_2D)[0],1):
        var_2D_frame = np.squeeze(var_2D[t,:,:])
        var_2D_norm = np.divide(var_2D_frame,weight_area)
        var_2D_l.append(var_2D_norm)
    var_2D = np.asarray(var_2D_l)

    return var_2D,var_fulldomain_1D

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

def convert_1D_to_2D(var_1D,lat_2d):
    # Convert global 1D to global 2D
    len_lat=shape(lat_2d)[0]; len_lon=shape(lat_2d)[1]; nVar1D = shape(var_1D); nComp = nVar1D[0]
    var_2D_list = []
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

### P3: Plotting
def contour_function(lon_2d,lat_2d,var_2d,color_ticks,fill_continents,title,savefigure_name):
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
    plt.title(title)
    save_file_name = savefigure_name 
    save_file = save_file_name + '.png'
    fig.savefig(save_file)
    plt.close()

def plot_interannualseries(var_1d_yearseries,year_start,year_end,title,savefile_name,unit):
    year_duration = shape(var_1d_yearseries)[0]; year_label = []; year_step = 20
    for t in np.arange(year_start, year_end+year_step, year_step):
        year_label.append(str(t))           
    fig = plt.figure()                                                          
    plt.plot(np.arange(1, year_duration+1, 1), var_1d_yearseries, 'r')
    plt.plot(np.arange(1, year_duration+1, 1), var_1d_yearseries, 'ro')
    plt.plot(np.arange(1, year_duration+1, 1), np.zeros(shape(var_1d_yearseries)), 'k--')
    plt.xticks(np.arange(1, year_duration+year_step, year_step), year_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.show()
    fig.savefig(savefile_name+'.png') #,optimize=True,quality=85)
    plt.close()
############## EOF ##############
