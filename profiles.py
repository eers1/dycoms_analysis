import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

nc = 'mbl_sc_dycoms_dg_7260.0.nc'
DS = xr.open_dataset(nc)

def profile_plots(var):
    var_T = var.transpose()
    var_T.plot()
    plt.title(var.name, fontsize=12)
    plt.show()

[profile_plots(val) for (key, val) in DS.data_vars.items() if len(val.dims) == 2] 
