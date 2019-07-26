import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

nc = '../MONC/mbl_sc_casim_dg_7260.0.nc'
DS = xr.open_dataset(nc)

def scalar_plot(DS, var):
    DS[var].plot()
    plt.title(var, fontsize=12)
    plt.show()

[scalar_plot(DS, name) for (name, obj) in DS.data_vars.items() if len(obj.dims) == 1] 
