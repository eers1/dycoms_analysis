from netCDF4 import Dataset
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt


class Diagnostic_scalar:
    def __init__(self, name, data, units):
        self.name = name
        self.data = data
        self.units = units

    def plot_scalar(self, timeseries):
        plt.figure()
        ax = plt.subplot(111)
        
        ax.plot(timeseries, self.data, label=self.name, linewidth=2)
        ax.set_ylabel(self.name + " " + self.units, fontsize=15)
        ax.set_xlabel("Time (s)", fontsize=15)

        plt.title(self.name, fontsize=15)
        return plt.show()



###### Main #######
file = "/nfs/see-fs-01_users/eers/MONC/mbl_sc_casim_dg_7260.0.nc"
dataset = Dataset(file, "r")
scalar_time = dataset.variables["time_series_75_60.0"][:]

# plot scalar diagnostics
lwp = Diagnostic_scalar("Liquid Water Path", dataset.variables["LWP_mean"][:]*1000, "g m$^{-2}$")
lwp.plot_scalar(scalar_time)

vwp = Diagnostic_scalar("Vapour Water Path", dataset.variables["VWP_mean"][:]*1000, "g m$^{-2}$")
vwp.plot_scalar(scalar_time)


