#!/usr/bin/env python3
import sys
from sys import argv
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ncplotlib as ncplt

# Open dataset
nc = sys.argv[1]
DS = xr.open_dataset(nc)

# Select specific diagnostics or take all
specific_diagnostics = 1
# Coupled?
casim = 1
socrates = 0

if specific_diagnostics == 1:
    if casim == 1:
        names_1D = ['VWP_mean', 'LWP_mean', 'RWP_mean', 'reske_mean', 'subke_mean', 'w_max', 'senhf_mean', 'lathf_mean', 'surface_precip_mean']
        names_2D = ['theta_mean', 'vapour_mmr_mean', 'liquid_mmr_mean', 'rain_mmr_mean', 'rh_mean', 'subgrid_turbulent_transport', 'total_cloud_fraction', 'liquid_cloud_fraction', 'resolved_turbulent_transport', 'tke_tendency','u_wind_mean', 'v_wind_mean', 'w_wind_mean', 'uu_mean', 'vv_mean', 'ww_mean']
        if socrates == 1:
            names_3D = ['surface_precip', 'vwp', 'lwp', 'rwp', 'reske', 'subke', 'cltop', 'clbas', 'toa_up_longwave', 'toa_down_shortwave', 'toa_up_shortwave']  
        elif socrates == 0:
            names_3D = ['surface_precip', 'vwp', 'lwp', 'rwp', 'reske', 'subke', 'cltop', 'clbas'] 
    elif casim == 0:
        names_1D = ['VWP_mean', 'LWP_mean', 'reske_mean', 'subke_mean', 'w_max', 'senhf_mean', 'lathf_mean']
        names_2D = ['theta_mean', 'vapour_mmr_mean', 'liquid_mmr_mean', 'rh_mean', 'subgrid_turbulent_transport', 'total_cloud_fraction', 'liquid_cloud_fraction', 'resolved_turbulent_transport', 'tke_tendency','u_wind_mean', 'v_wind_mean', 'w_wind_mean', 'uu_mean', 'vv_mean', 'ww_mean']
        if socrates == 1:
            names_3D = ['vwp', 'lwp', 'reske', 'subke', 'cltop', 'clbas', 'toa_up_longwave', 'toa_down_shortwave', 'toa_up_shortwave'] 
        elif socrates == 0:
            names_3D = ['vwp', 'lwp', 'reske', 'subke', 'cltop', 'clbas']
elif specific_diagnostics == 0:
    names_1D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 1]
    names_2D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 2]
    names_3D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 3]

names_4D = [key for (key, val) in DS.data_vars.items() if len(val.dims) == 4]

#names_1D = ['LWP_mean']
#names_2D = ['liquid_mmr_mean']
#names_3D = ['lwp', 'cltop', 'clbas']
#names_4D = ['q_cloud_liquid_number']

# Create object list
scalars = [DS[name] for name in names_1D]
profiles = [DS[name] for name in names_2D]
scenes = [DS[name] for name in names_3D]
slices = [DS[name] for name in names_4D]

# Plot using ncplotlib
savepath = 'testplots/oat2/oat2_'

for var in scalars:
    fig, axes = plt.subplots()
    fig = ncplt.scalar(fig, axes, var)
    fig.savefig(savepath + var.name + '.png')
    plt.close()

for var in profiles:
    fig, axes = plt.subplots()
    fig = ncplt.profile(fig, axes, var)
    fig.savefig(savepath + var.name + '.png')
    plt.close()

##for var in scenes:
##    #plt.subplots()
##    ncplt.scene(var, 10, savepath)
##    print(var.name + ' saved')
##
##print('*** scenes processed ***')
   
[ncplt.scene(var, savepath) for var in scenes]
[ncplt.vslice(var, 60, 'x', savepath) for var in slices]
                                   
