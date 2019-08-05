import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

nc = 'mbl_sc_dycoms_dg_7260.0.nc'
DS = xr.open_dataset(nc)

def scene_plots(var):
    var.plot(x='x', y='y', col=var.dims[0])
#    plt.title(var.name, fontsize=12)
    plt.show()

[scene_plots(val) for (key, val) in DS.data_vars.items() if len(val.dims) == 3] 
