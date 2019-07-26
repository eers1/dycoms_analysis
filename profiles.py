import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

nc = '../../MONC/dycoms_simulation/mbl_sc_dycoms_dg_7260.0.nc'
DS = xr.open_dataset(nc)

def profile_plots(DS, obj, name):
    var_T = obj.transpose()
    var_T.plot()
    plt.title(name, fontsize=12)
    plt.show()

[profile_plots(DS, obj, name) for (name, obj) in DS.data_vars.items() if len(obj.dims) == 2] 
