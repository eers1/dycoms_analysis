import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

nc = 'mbl_sc_dycoms_dg_7260.0.nc'
DS = xr.open_dataset(nc)

def scalar_plot(var):
    var.plot()
    plt.title(var.name, fontsize=12)
    plt.show()

[scalar_plot(val) for (key, val) in DS.data_vars.items() if len(val.dims) == 1] 
