import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

nc = 'mbl_sc_dycoms_dg_7260.0.nc'
DS = xr.open_dataset(nc)

def hslice_plots(var, height):
    time = var.dims[0]
    if var.dims[3] == 'z':
        var.sel(z=height, method='nearest').plot(x='x', y='y', col=time)
    else:
        var.sel(zn=height, method='nearest').plot(x='x', y='y', col=time)
#    plt.title(var.name, fontsize=12)
    plt.show()

    
height = 300
[hslice_plots(val, height) for (key, val) in DS.data_vars.items() if len(val.dims) == 4] 
