import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

nc = '../../MONC/dycoms_simulation/mbl_sc_dycoms_dg_7260.0.nc'
DS = xr.open_dataset(nc)

def scalar_plot(var, name):
    var.plot()
    plt.title(name, fontsize=12)
    plt.show()

[scalar_plot(val, key) for (key, val) in DS.data_vars.items() if len(val.dims) == 1] 
