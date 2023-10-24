#!/usr/bin/env python3.9

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import decimal
from collections import Counter
from scipy.stats import norm

#rc('text', usetex=True)
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 15})


class Emulator:
    def __init__(self, data_path, predicted_path, **kwargs):
        self.data = np.loadtxt(data_path, delimiter=",")
        self.lwp_ppe = self.data[:20, 2]
        self.lwp_ppe_o = [self.data[0,2],self.data[1,2],self.data[3,2],self.data[4,2],self.data[5,2],self.data[6,2],self.data[7,2],self.data[9,2],self.data[11,2],self.data[12,2], self.data[15,2]]
        self.lwp_ppe_ens = [self.data[2,2],self.data[8,2],self.data[10,2],self.data[13,2],self.data[14,2],self.data[16,2],self.data[17,2],self.data[18,2],self.data[19,2]]
        self.lwp_val = self.data[24:32, 2]
        self.lwp_extra = self.data[32:38, 2]
        self.lwp_oat = self.data[20:24, 2]
        self.lwp_base = self.data[38, 2]
        self.ppe_theta_arr = self.data[:20, 0]
        self.ppe_theta = self.ppe_theta_arr.tolist()
        self.ppe_theta_o = [self.data[0,0],self.data[1,0],self.data[3,0],self.data[4,0],self.data[5,0],self.data[6,0],self.data[7,0],self.data[9,0],self.data[11,0],self.data[12,0], self.data[15,0]]
        ens = np.asarray([2,8,10,13,14,16,17,18,19])
        self.ppe_theta_ens = self.ppe_theta_arr[ens]
        self.ppe_qt_arr = self.data[:20, 1]
        self.ppe_qt = self.ppe_qt_arr.tolist()
        self.ppe_qt_o = [self.data[0,1],self.data[1,1],self.data[3,1],self.data[4,1],self.data[5,1],self.data[6,1],self.data[7,1],self.data[9,1],self.data[11,1],self.data[12,1], self.data[15,1]]
        self.ppe_qt_ens = [self.data[2,1],self.data[8,1],self.data[10,1],self.data[13,1],self.data[14,1],self.data[16,1],self.data[17,1],self.data[18,1],self.data[19,1]]
        self.val_theta = self.data[24:32, 0].tolist()
        self.val_qt = self.data[24:32, 1].tolist()
        self.extra_theta = self.data[32:38, 0].tolist()
        self.extra_qt = self.data[32:38, 1].tolist()
        self.oat_qt = [-7.5, -7.5, 0, -9]
        self.oat_theta = [2, 20, 8.5, 8.5]
        self.base_qt = -7.5
        self.base_theta = 8.5
        self.predict_data = np.loadtxt(predicted_path,delimiter=",",skiprows=1)
        self.predictions = self.predict_data[:,1]
        self.lower95 = self.predict_data[:,3]
        self.upper95 = self.predict_data[:,4]
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def plot_scatter_result(self, title, output_label, cmap, figname, withval,
                            valcolour, withoat, withbase, withextra,numbers,withcbar,fig=None,ax=None,vmin=None,vmax=None):
        if ax is None:
            fig, ax = plt.subplots()
        if vmax is None:
            vmin = self.predictions.min()
            vmax = self.predictions.max()
        norm = colors.TwoSlopeNorm(vcenter=(vmax-(vmax-vmin)/2),
                                       vmin=vmin,
                                       vmax=vmax)
        cm = plt.cm.get_cmap(cmap)
        if withval == True:
            if type(valcolour) == str:
                ax.scatter(self.val_theta, self.val_qt, c=valcolour, marker='s',norm=norm,s=self.dsize,edgecolors=valcolour)
            else:
                ax.scatter(self.val_theta,
                            self.val_qt,
                            c=self.lwp_val,
                            marker='o',
                            norm=norm,
                            s=self.dsize,
                            cmap=cm,edgecolors="black",linewidths=self.edgesize)
        if withoat == True:
            ax.scatter(self.oat_theta,
                        self.oat_qt,
                        c=self.lwp_oat,
                        norm=norm,
                        s=self.dsize,
                        cmap=cm,edgecolors='black',linewidths=self.edgesize)
        if withbase == True:
            ax.scatter([self.base_theta], [self.base_qt],
                        c=[self.lwp_base],
                        marker='v',
                        norm=norm,
                        s=self.dsize,
                        cmap=cm,edgecolors='black',linewidths=self.edgesize)
        if withextra == True:
            ax.scatter([self.extra_theta], [self.extra_qt],
                        c=[self.lwp_extra],
                        norm=norm,
                        s=self.dsize,
                        cmap=cm,
                        edgecolors='red',
                        linestyle='--',linewidths=self.edgesize)
        ax.scatter(self.ppe_theta,
                    self.ppe_qt,
                    c=self.lwp_ppe,
                    norm=norm,
                    s=self.dsize,
                    cmap=cm, edgecolors='black',linewidths=self.edgesize)
        if numbers == True:
            for i,(x,y) in enumerate(zip(self.ppe_theta,self.ppe_qt)):
                ax.text(x+0.1,y+0.05,('D'+str(i+1)), color='black', fontsize=15)
            for i,(x,y) in enumerate(zip(self.val_theta,self.val_qt)):
                ax.text(x+0.1,y+0.05,('V'+str(i+1)), color='black', fontsize=15)
            for i,(x,y) in enumerate(zip(self.extra_theta,self.extra_qt)):
                ax.text(x+0.1,y+0.05,('E'+str(i+1)), color='black', fontsize=15)
            for i,(x,y) in enumerate(zip(self.oat_theta,self.oat_qt)):
                ax.text(x+0.1,y+0.05,('O'+str(i+1)), color='black', fontsize=15)
            ax.text(self.base_theta,self.base_qt,('B'), color='black', fontsize=12)
        if withcbar==True:
            cbar = plt.colorbar()
            cbar.set_label(output_label,fontsize=self.font)
        ax.set_xlabel(self.xlabel,fontsize=self.font)
        #ax.set_ylabel(self.ylabel,fontsize=self.font)
        ax.set_ylim([-9, 0])
        ax.set_xlim([2, 20])
        ax.set_title(title,fontsize=self.font)
        if self.savebool == True:
            fig.savefig(self.path + figname)
        return fig,ax

    def plot_response_surface(self, title, output_label, figname, clfractend, fig=None,ax=None):
        if ax is None:
            fig = plt.figure(figsize=(8,6.5))
            ax = fig.add_subplot(111, projection='3d')
        #Z = np.reshape(np.asarray(self.predictions),(50,50))
        surf = ax.scatter(self.samples[:,0],self.samples[:,1],self.predictions)
        #surf = ax.plot_surface(self.theta_list,self.qt_list,Z)
        ax.set_xlabel(self.xlabel,fontsize=self.font)
        ax.set_ylabel(self.ylabel,fontsize=self.font)
        ax.set_zlabel(output_label,fontsize=self.font)
        ax.set_xlim(2, 20)
        ax.set_ylim(-9, 0)
        ax.set_title(title,fontsize=self.font)
        if clfractend==True:
            ax.view_init(azim=320)
        else:
            ax.view_init(azim=230)
        if self.savebool == True:
            fig.savefig(self.path + figname)
        #return fig

    def plot_2DCmap(self, title, output_label, cmap,levels, diverging, climited,norm, figname,fig=None,ax=None,cbar=None,transect=False, legloc=None, extend=None, ensemble_select=None):
        axis_labels = False
        if ax is None:
            fig, ax = plt.subplots()
            axis_labels = True
        cm = plt.cm.get_cmap(cmap)
        X, Y = np.meshgrid(self.theta_list, self.qt_list)
        Z = np.reshape(self.predictions, (self.newpoints, self.newpoints))
        self.contour = ax.contourf(X,Y,Z,levels, cmap=cm, extend=extend)#, norm=norm)
        self.contour = ax.contourf(X,Y,Z,levels, cmap=cm, extend=extend)#, norm=norm)
        self.valmark=ax.scatter(self.val_theta,
                                self.val_qt,
                                c=self.lwp_val,
                                norm=norm,
                                s=self.dsize,
                                cmap=cm,
                                marker='s',
                                edgecolors='black',
                                #edgecolors="#C2476E",
                                linewidths=self.edgesize)
        self.ppemark=ax.scatter(self.ppe_theta_o,
                                self.ppe_qt_o,
                                c=self.lwp_ppe_o,
                                s=self.dsize,
                                cmap=cm,
                                norm=norm,
                                edgecolors='black',
                                marker='o',
                                #edgecolors="#2D93AD",
                                linewidths=self.edgesize,
                                facecolors=None)
        self.ensemblemark=ax.scatter(self.ppe_theta_ens,
                                     self.ppe_qt_ens,
                                     c=self.lwp_ppe_ens,
                                     s=self.dsize,
                                     cmap=cm,
                                     norm=norm,
                                     edgecolors='black',
                                     marker='o',
                                     #edgecolors="#2D93AD",
                                     linewidths=self.edgesize,
                                     facecolors=None)
        if self.extras==True:
            self.extramark=ax.scatter(self.extra_theta,
                   self.extra_qt,
                   c=self.lwp_extra,
                   s=self.dsize,
                   cmap=cm,
                   norm=norm,
                   edgecolors='black',
                   marker='^',
                   #edgecolors="#2D93AD",
                   #linestyle=(0,(1,1)),
                   linewidths=self.edgesize)
        if self.pltvar == True:
            ax.scatter([
                3.06, 2.56, 3.06, 3.06, 3.56, 14.85, 14.35, 14.85, 14.85,
                15.35, 2.52, 2.02, 2.52, 2.52, 3.02
            ], [
                -0.02, -0.02, 0, -0.22, -0.02, -7.36, -7.36, -7.16, -7.56,
                -7.36, -7.08, -7.08, -6.88, -7.28, -7.08
            ],
                       c=ensembles,
                       s=0.5*self.dsize,
                       cmap=cm,
                       edgecolors="grey")
        if cbar == True:
            cbar = plt.colorbar(self.contour)
            cbar.set_label(output_label)

        #ax.scatter([10, 7.5, 12, 15.3,3], [-1, -4, -6.5, -8.5,-1.5], c ="b")
        ax.set_ylabel(self.ylabel,fontsize=self.font)
        ax.set_xlabel(self.xlabel,fontsize=self.font)
        ax.set_ylim([-9, 0])
        ax.set_xlim([2, 20])
        ax.set_title(title,fontsize=self.font)
               #labelsize=self.font)
               #size=self.font)
        if axis_labels == False:
            ax.set_xlabel('.', color=(0,0,0,0))
            ax.set_ylabel('.', color=(0,0,0,0))
        if transect!=True:
            self.dycomsmark=ax.scatter(8.5, -7.5,
                                       c=self.lwp_base,
                                       cmap=cm,norm=norm,
                                       marker="v",
                                       edgecolors='white',
                                       #edgecolors="#348357",
                                       s=self.dsize,
                                       linewidths=self.edgesize)
            #self.add_Dycoms(ax)
            self.kparamline=self.add_Kparam(ax,"#CE8964")
        else:
            #arr = np.asarray([2,8,10,13,14,16,17,18,19])
            #if any(ensemble_select)==-1:
            #    ensemble_plot=False
            #else: # ensemble_select is not None:
            #    ensemble_plot=True
            theta_ens_plot = self.ppe_theta_arr[ensemble_select]
            qt_ens_plot = self.ppe_qt_arr[ensemble_select]
            lwp_ens_plot = self.lwp_ppe[ensemble_select]
            #else:
            #    ensemble_plot=True
            #    theta_ens_plot = self.ppe_theta_arr[arr]
            #    qt_ens_plot = self.ppe_qt_arr[arr]
            #    lwp_ens_plot = self.lwp_ppe[arr]

            if ensemble_select is not None:
                ax.scatter(theta_ens_plot,
                        qt_ens_plot,
                        c=lwp_ens_plot,
                        s=self.dsize,
                        cmap=cm,
                        norm=norm,
                        marker='o',
                        edgecolors="#2D93AD",
                        linewidths=2*self.edgesize,
                        facecolors=None)
        if self.savebool == True:
            fig.savefig(self.path + figname)
        if legloc is not None:
            ax.legend((self.ppemark,self.valmark,self.extramark,self.dycomsmark,self.kparamline),('Training data','Validation data','Extra simulations','DYCOMS-II RF01',r'\kappa parameter'),fontsize=10,loc=legloc)
        return self.contour

    def plot_transect(self, ax, output_label, line_label, line_colour, ensembles=None, values=None, vmin=None, vmax=None, fig=None, through_points=None, design_predictions = None, line_legend=None, ylim=None):
        '''
        specific_points should be added as [[x0, ... ,xN],[y0, ... ,yN]]
        '''
        Z = np.reshape(self.predictions, (self.newpoints, self.newpoints))
        u95 = np.reshape(self.upper95, (self.newpoints, self.newpoints))
        l95 = np.reshape(self.lower95, (self.newpoints, self.newpoints))
        transect_vals = []
        upper_tvals = []
        lower_tvals = []
        if through_points is None:
            for n in range(50):
                transect_vals.append(Z[n,n])
                upper_tvals.append(u95[n,n])
                lower_tvals.append(l95[n,n])
        else:
            f = interpolate.interp1d(through_points[0],through_points[1],fill_value='extrapolate')
            
            xnew = np.linspace(2,20,50)
            ynew = f(xnew)
            np.savetxt('./transect_points.csv',np.stack((xnew,ynew),axis=-1),delimiter=',')
            theta_list50 = []
            for n,m in zip(xnew,ynew):
                idn = (np.abs(np.array(self.theta_list)-n)).argmin()
                idm = (np.abs(np.array(self.qt_list)-m)).argmin()
                transect_vals.append(Z[idn,idm])
                upper_tvals.append(u95[idn,idm])
                lower_tvals.append(l95[idn,idm])
                theta_list50.append(self.theta_list[idn])
        if design_predictions is not None:
            des_pred_mean = design_predictions[[25,13,2,1,6],1]
            des_pred_l95 = design_predictions[[25,13,2,1,6],3]
            des_pred_u95 = design_predictions[[25,13,2,1,6],4]

            x_inds = []
            x_inds = [6,19,27,36,42]
            for i in range(5):
                transect_vals = np.insert(transect_vals,x_inds[i],des_pred_mean[i])
                upper_tvals = np.insert(upper_tvals,x_inds[i],des_pred_u95[i])
                lower_tvals = np.insert(lower_tvals,x_inds[i],des_pred_l95[i])
                theta_list50 = np.insert(theta_list50,x_inds[i],through_points[0][i])
                #print(transect_vals[x_inds[i]-1],des_pred_mean[i],transect_vals[x_inds[i]], transect_vals[x_inds[i]+1])
                #print(des_pred_u95[i], upper_tvals[x_inds[i]-1], upper_tvals[x_inds[i]+1])
                #print(des_pred_l95[i], lower_tvals[x_inds[i]-1], lower_tvals[x_inds[i]+1])
                #print(theta_list50[x_inds[i]-1], through_points[0][i], theta_list50[x_inds[i]], theta_list50[x_inds[i]+1])

        transect_line, = ax.plot(theta_list50, transect_vals, label=line_label + " mean", color=line_colour)
        upper_line, = ax.plot(theta_list50, upper_tvals, label=line_label+" upper 95%", color=line_colour, linestyle='dashed')
        lower_line, = ax.plot(theta_list50, lower_tvals, label=line_label+" lower 95%", color=line_colour, linestyle='dashed')
        # if ensembles is not None:
        #     print("ensembles is not none")
        #     thick_stdper = np.std(ensembles[:5])/np.mean(ensembles[:5])
        #     kparam_stdper = np.std(ensembles[5:10])/np.mean(ensembles[5:10])
        #     thin_stdper = np.std(ensembles[10:14])/np.mean(ensembles[10:])
        #     d7 = values[0]
        #     d7_stdper = d7*thick_stdper
        #     d3 = values[1]
        #     d3_stdper = d3*kparam_stdper
        #     e6 = values[2]
        #     e6_stdper = e6*thin_stdper
        #     ax.plot([15.62,15.62], [d7-np.abs(2*d7_stdper), d7+np.abs(2*d7_stdper)],color='darkblue')
        #     ax.plot([10.86,10.86], [d3-np.abs(2*d3_stdper), d3+np.abs(2*d3_stdper)],color='darkblue')
        #     ax.plot([4,4], [e6-np.abs(2*e6_stdper), e6+np.abs(2*e6_stdper)],color='darkblue')
        #ax.set_title(line_label,fontsize=self.font)
        #ax.set_ylabel(output_label)
        if line_legend is True:
            ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)
        elif vmin is not None:
            ax.set_ylim([vmin, vmax])

            
        ax.set_xlabel('.', color=(0,0,0,0))
        ax.set_ylabel('.', color=(0,0,0,0))

        return transect_line, upper_line, lower_line

    def add_Dycoms(self, ax):
        dycomsmark=ax.scatter(8.5, -7.5, marker="v", color="#348357", s=180)
        # ax.text(8.5,
        #         -8.5, r"DYCOMS-II"
        #         "\n"
        #         r"RF01",
        #         fontsize=self.markerfonts,
        #         ha="center")
        return dycomsmark

    def add_Kparam(self, ax, kparam_col):
        kparamline=ax.plot(self.kx, self.ky, color="white", linestyle='--')#color=kparam_col)
        # ax.text(2.25,
        #         -3.1,
        #         r"$\kappa$ for DYCOMS-II",
        #         ha="left",
        #         fontsize=self.markerfonts,
        #         rotation=-46.5)
        return kparamline

    def plot_residuals(self, ax, hist, bin_quantity, line_label=None, higher_bound=None, lower_bound=None, ensemble_comp=None):
        '''
        Plot frequencies of residuals (in bins)
        '''
        pred = np.reshape(self.predictions, (self.newpoints,self.newpoints))
        r_qt = np.linspace(-9,0,self.newpoints)
        r_th = np.linspace(2,20,self.newpoints)

        residuals = []
        for i in range(20):
            th = self.ppe_theta[i]
            qt = self.ppe_qt[i]
            actual = self.lwp_ppe[i]
            em_val = find_nearest(pred, r_qt, r_th, qt, th)
            r = (actual - em_val)/(em_val)
            residuals.append(r)

        for i in range(6):
            th = self.extra_theta[i]
            qt = self.extra_qt[i]
            actual = self.lwp_extra[i]
            em_val = find_nearest(pred, r_qt, r_th, qt, th)
            r = (actual - em_val)/(em_val)
            residuals.append(r)

        #print(residuals)
            
        bins = create_bins(bin_quantity, residuals, higher_bound, lower_bound)
        starts = [start for start, end in bins]

        binned_weights = []
        for res in residuals:
            ind = find_bin(res, bins)
            binned_weights.append(ind)
        frequencies = Counter(binned_weights)
        heights = []
        [heights.append(frequencies[i]) for i in range(0,len(bins))]
        residuals.sort()
        ax.plot(residuals, norm.pdf(residuals))
        #if ensemble_comp is not None:
        #    ax.plot(ensemble_comp, norm.pdf(ensemble_comp))
        #if hist==True:
        #ax.hist(residuals, bins=bin_quantity) #bins=starts+[bins[-1][1]]) #, density=True)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 0.45)
        if line_label is not None:
            ax.set_title(line_label,fontsize=self.font)

def find_nearest(array, range_qt, range_th, qt, th):
    '''
    Finds closest index to specific qt and theta
    values in emulator grid of predictions
    '''
    ind1 = (np.abs(range_qt - qt)).argmin()
    ind2 = (np.abs(range_th - th)).argmin()
    return array[ind1, ind2]

def create_bins(quantity, residuals=None, higher=None, lower=None):
    if higher==None:
        rmin = min(residuals)
        rmax = max(residuals)
        extreme = max(abs(rmin), abs(rmax))
        rounded_extreme = decimal.Decimal(extreme).quantize(decimal.Decimal('0.00'), rounding=decimal.ROUND_CEILING)
        higher = rounded_extreme
        lower = -rounded_extreme
    bin_width = (higher - lower)/quantity
    bins = []
    low = lower
    for b in range(quantity):
        bins.append((low, low + bin_width))
        low = low + bin_width
    return bins 

def find_bin(value, bins):
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
            return i
    return -1

def create_samples(qt_min, qt_max, theta_min, theta_max, new_points):
    samples = np.empty([new_points**2, 2])
    qt_grid = np.linspace(qt_min, qt_max, new_points)
    theta_grid = np.linspace(theta_min, theta_max, new_points)
    i = 0
    qt_list = []
    theta_list = []
    for qt in qt_grid:
        qtval = qt
        qt_list.append(qt)
        for theta in theta_grid:
            thetaval = theta
            samples[i, 0] = thetaval
            samples[i, 1] = qtval
            i += 1
    for theta in theta_grid:
        theta_list.append(theta)
    return samples, theta_list, qt_list

def create_samples_gen(param_dict, new_points):
   # samples = np.empty([new_points**len(param_dict.items()), len(param_dict.items())])

    X = [np.linspace(val[0], val[1], new_points) for key,val in param_dict.items()]
    g1,g2,g3,g4,g5,g6 = np.meshgrid(X[0], X[1], X[2], X[3], X[4], X[5])
    return g1, g2,g3,g4,g5,g6
