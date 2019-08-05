import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

nc = 'mbl_sc_dycoms_dg_7260.0.nc'
DS = xr.open_dataset(nc)

def vslice_plots(var, depth, horizontal_dim):
    time = var.dims[0]
    z = var.dims[3]
    if horizontal_dim == 'x':
        var[:,depth,:,:].plot(x='y', y=z, col=time)
    elif horizontal_dim == 'y':
        var[:,:,depth,:].plot(x='x', y=z, col=time)
#    plt.title(var.name, fontsize=12)
    plt.show()

horizontal_dim = 'y'
depth = 30

[vslice_plots(val, depth, horizontal_dim) for (key, val) in DS.data_vars.items() if len(val.dims) == 4]
