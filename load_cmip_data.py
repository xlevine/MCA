import sys, os
import numpy as np
from netCDF4 import Dataset
from pylab import *
from os.path import expanduser
home = expanduser("~")

###########################################################################
# LISTS: Defines name of variables (following CMIP6 convention) and seasons 
month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
season_dict = {'winter':['dec','jan','feb'],'summer':['jun','jul','aug'],'spring':['mar','apr','may'],'fall':['sep','oct','nov']}

varlist_Amon = ['clt','evspsbl','hurs','pr','rlds','rsds','rsut','ta','tasmin','ttrea','utendepfd','va','vurea','wurea','epfy','hfls','hursmin','prc','rlus','rsdt','sfcWind','tas','tos','ua','utrea','vas','vvrea','wvrea','epfz','hfss','huss','psl','rlut','rsus','siconca','tasmax','ts','uas','uurea','vtrea','wap','zg', 'Qflux']
varlist_Omon = ['msftyz', 'mlotst', 'evs','friver','hfds','mlotst','sos','vmo','ficeberg2d','rsntds','nshfls','downempfluxoce','downnoswfluxoce','downswfluxoce', 'thetao', 'so', 'tos', 'uo', 'vo', 'wfonocorr']
varlist_SImon = ['siage','sithick','sialb','siconc','sidmassevapsubl','sisnthick','sitemptop','sivol','sistrxdtop','downempfluxice','downnoswfluxice','downswfluxice','freshwater','siheco','sinudg','sstnudg','totdownfluxice','vfxice','vfxsnw','seaice_net_enthflux','seaice_total_enthflux','heatnudg']

# METADATA: Contains name convention and data path of CMIP6. THIS MUST BE CHANGED FOR EACH USER. 
metadata_dict = {'expt_name':'historical-CMIP6','expt':'a1st','version':'v20200120','expt_name_path':'ssp245','model_type':'EC-Earth3','expt_folder':"/esarchive/exp/ecearth/",'file_path':"/cmorfiles/ScenarioMIP/EC-Earth-Consortium/",'sim_config':'r7i1p1f1','year_ini':2015,'miss_val':1.e20}
mask_dict = {'mask_path_ocean':"/esarchive/autosubmit/con_files/",'mask_name_ocean':'mask.regions.Ec3.2_O1L75.nc','mask_path_atmos':"/esarchive/exp/ecearth/constant/",'mask_name_atmos':'mask.regions.ifs_512x256.nc'}
area_dict={'area_root_path':"",'file_gridcell_ocean':'areacello_fx_EC-Earth3_gn.nc','var_gridcell_ocean':'areacello','file_gridcell_atmos':'areacella_fx_EC-Earth3_gr.nc','var_gridcell_atmos':'areacella'}
custom_box_dic = {'global':(-90,90,0,360), 'north_hem':(0,90,0,360), 'north_hem_polar':(55,90,0,360), 'north_hem_midpolar':(35,90,0,360), 'north_hem_mid':(35,55,0,360), 'north_hem_subpolar':(55,70,0,360), 'north_hem_tropics': (-5,35,0,360), 'north_hem_midfront':(65,90,0,360)}
# 45,65
def define_regional_mask(varname,region):
    grid_type = define_grid(varname)
    if region in custom_box_dic:
        mask_region = load_custombox(region,grid_type)
    else:
        mask_region = load_oceanarea(region,grid_type)
        
    return mask_region

def load_custombox(region, grid_type):
    [lat_2d, lon_2d] = get_axis_from_grid(grid_type) 
    (lat_south, lat_north, lon_west, lon_east) = custom_box_dic[region]
    lon_2dT = transform_lon(lon_2d,grid_type)
    mask_region = np.ones(np.shape(lat_2d))
    mask_region[np.where(lat_2d>=lat_north)]=0
    mask_region[np.where(lat_2d<=lat_south)]=0
    mask_region[np.where(lon_2dT>=lon_east)]=0
    mask_region[np.where(lon_2dT<=lon_west)]=0    

    return mask_region

def transform_lon(lon_2d,grid_type):
    if grid_type=='areacello':
        lon_pos = np.add(lon_2d,180.0*np.ones(shape(lon_2d)))
        lon_2d[np.where(lon_2d<0)] = lon_pos[np.where(lon_2d<0)]
        
    return lon_2d

def get_axis_from_grid(grid_type):
    if grid_type=='areacello':
        [_, lon_2d, lat_2d, _] = load_var(0,'tos','jan')
    elif grid_type=='areacella':
        [_, lon_2d, lat_2d, _] = load_var(0,'tas','jan')

    return lat_2d, lon_2d

def load_oceanarea(region,grid_type):
    if 'areacello' in grid_type:
        path = mask_dict['mask_path_ocean']
        region_mask_name = mask_dict['mask_name_ocean']
    else:
        path = mask_dict['mask_path_atmos']
        region_mask_name = mask_dict['mask_name_atmos']

    mask_path = path + region_mask_name
    var_file = Dataset(mask_path, 'r')
    mask_region_3D = var_file.variables[region]
    mask_region_3D = np.asarray(mask_region_3D)

    if 'areacello' in grid_type:
        mask_region = np.squeeze(mask_region_3D[0,0,:,:])
    else:
        mask_region = mask_region_3D  
    mask_region = np.asarray(mask_region)

    return mask_region

def load_input_var(varname1, season1, member_start, member_end):
    nmemb=np.arange(member_start,member_end+1,1)
    var1 = []
    for memb in nmemb:
        print(memb)
        [var1_memb, lon1, lat1, plev1] = load_var(memb,varname1,season1)
        var1.append(var1_memb)
    var1 = np.asarray(var1)
        
    return var1, lon1, lat1

def load_input_var_2D(varname1, season1, region1, member_start, member_end):
    nmemb=np.arange(member_start,member_end+1,1)
    region1 = define_regional_mask(varname1,region1)
    region1[np.where(region1==0)]=np.nan
    var1_2D = []
    for memb in nmemb:
        print(memb)
        [var1_memb, lon1, lat1, lev1] = load_var(memb,varname1,season1)        
        var1_memb = var1_memb * region1
        var1_memb_2D = compute_zmean(var1_memb)                        
        var1_2D.append(var1_memb_2D)
    var1_2D = np.asarray(var1_2D)

    return var1_2D, lev1, lat1

def compute_zmean(var1):
    var1_zmean = np.nanmean(var1,axis=-1)
    
    return var1_zmean

def load_var(num_member, varname, month_name):    
    if month_name in month_list:
        [var, lon_2d, lat_2d, plev] = read_var_month(num_member, varname, month_name)
    else:
        var_season = []
        for month in season_dict[month_name]: 
            [var_month, lon_2d, lat_2d, plev] = read_var_month(num_member, varname, month)
            var_season.append(var_month)
        var_season = np.asarray(var_season)
        var = np.nanmean(var_season,axis=0)
        
    return var, lon_2d, lat_2d, plev

def read_var_month(num_member, varname, month_name):
    month_index = get_timestamp(month_name)
    [var_year, lon_2d, lat_2d, plev] = read_var(num_member, varname)

    if len(np.shape(var_year))==4:
        var = np.squeeze(var_year[int(month_index),:,:,:])
    else:
        var = np.squeeze(var_year[int(month_index),:,:])

    return var, lon_2d, lat_2d, plev

def compute_cell_area(varname):
    root_path=area_dict['area_root_path']
    grid_type = define_grid(varname)
    if 'areacello' in grid_type:
        file_gridcell=root_path + area_dict['file_gridcell_ocean']
        gridcell_file = Dataset(file_gridcell, 'r')
        weight_area = gridcell_file.variables[area_dict['var_gridcell_ocean']]     
    else:
        file_gridcell=root_path + area_dict['file_gridcell_atmos']
        gridcell_file = Dataset(file_gridcell, 'r')
        weight_area = gridcell_file.variables[area_dict['var_gridcell_atmos']]     
    weight_area= np.asarray(weight_area)
    weight_area[np.where(weight_area>=metadata_dict['miss_val'])]=np.nan
    weight_area = np.divide(weight_area,np.nansum(weight_area)*np.ones(shape(weight_area)))

    return weight_area

def read_var(num_member, varname):
    sim_array = get_filename(num_member, varname)
    var = []; lev = []
    for k in np.arange(0,len(sim_array),1):
        var_file = Dataset(sim_array[k], 'r')
        var_k = var_file.variables[varname]
        var.extend(var_k)
    var=np.asarray(np.squeeze(var))
    try:
        grid_type = var_file.variables[varname].cell_measures
    except:
        grid_type = 'areacello' 

    if 'areacello' in grid_type:
        lon_2d = var_file.variables['longitude']; lat_2d = var_file.variables['latitude']
        lon_2d = np.asarray(lon_2d); lat_2d = np.asarray(lat_2d) 
        lon_2d_pos = np.add(lon_2d,180.0*np.ones(shape(lon_2d)))
        lon_2d[np.where(lon_2d<0)] = lon_2d_pos[np.where(lon_2d<0)]
        if len(np.shape(var))==4:
            lev = var_file.variables['lev'][:]
    else:
        lon = var_file.variables['lon']; lat = var_file.variables['lat']
        lon_2d = np.matlib.repmat(lon,len(lat),1); lat_2d_T = np.matlib.repmat(lat,len(lon),1)
        lat_2d = lat_2d_T.T
        lon_2d = np.asarray(lon_2d); lat_2d = np.asarray(lat_2d) 
        if len(np.shape(var))==4:
            lev = var_file.variables['plev'][:]

    var_file.close()
    var[np.where(var>=metadata_dict['miss_val'])] = np.nan
    return var, lon_2d, lat_2d, lev

def var_labels(varname):
    try:
        sim_array = get_filename(0, varname)
        var_file = Dataset(sim_array[0], 'r')
        units = var_file.variables[varname].units
        long_name = var_file.variables[varname].long_name
    except:
        units = ''
        long_name = ''

    var_dic={'units':units,'long_name':long_name}

    return var_dic

def define_grid(varname):
    if (varname in varlist_Omon) or (varname in varlist_SImon):
        grid_type = 'areacello'
    else:
        grid_type = 'areacella'

    return grid_type

def get_filename(num_member, varname):
    if varname in varlist_Omon:
        domain='Omon'; sim_type='gn'
    elif varname in varlist_SImon:
        domain='SImon'; sim_type='gn'
    elif varname in varlist_Amon:
        domain='Amon'; sim_type='gr'
    expt=metadata_dict['expt']; version=metadata_dict['version']; expt_name_path=metadata_dict['expt_name_path']; model_type=metadata_dict['model_type']; year_ini=metadata_dict['year_ini']; expt_folder=metadata_dict['expt_folder']; file_path=metadata_dict['file_path']; sim_config=metadata_dict['sim_config']

    slash="/"; udscr='_'; dash='-'; end_file_nc='.nc'
    year_membr = year_ini + num_member; date_year = [str(year_membr) + '01' + dash + str(year_membr) + '12']
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


#    sim_array = get_filename(0, varname)
#    var_file = Dataset(sim_array[0], 'r')
#    try:
#        grid_type = var_file.variables[varname].cell_measures
#    except:
#        grid_type = 'areacello'
