# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use("qt5agg")
from matplotlib import pyplot as plt
from solmat import Solmat
from fch_solve import cal_mass,cal_energy


def load_csv(str):
    return np.loadtxt(fname,delimiter=",")

if __name__ == '__main__':
    T_max = 10.
    N = int(T_max*1000)
    alpha = 1.9
    fname = "./csv/CH_u_{alpha}_T{T_max}_h0.01_dt0.001.csv".format(alpha=alpha,T_max=T_max)
    #fname = "./csv/CH_u_2.0_T10.0_h0.02_dt0.0001.csv"
    U = Solmat(alpha,100,N,0.,1.,T_max,1.)
    U.mat = load_csv(fname)
    print(len(U.mat[:,0]))
    num_frames = U.time_step
    U.plot_Animation(2,num_frames,save=False)
    # calcurate discret Mass
    """
    U.set_Mass(cal_mass(U,"neumann"))
    U.plot_Mass()
    U.set_Mass(cal_mass(U,"periodic"))
    U.plot_Mass()
    """
    p = -1.0
    q = -0.0005
    r = 1.0
    U.set_Energy(cal_energy(U,p,q,r))
    for k in [0,50,100,200,300,400,500,1000,5000,9999]:
        plt.plot(U.mat[:,k],label="{k} time step".format(k=k))
    plt.legend()
    plt.show()
    """

    plt.plot(np.r_[U.E_vec[:200],U.E_vec[200::500]],marker="s",mec="None",ms="3")
    #U.plot3D(save=False,diff_way="CH",cmap=None)

    for i in [0,100,500,1000,5000,9000,-1]:
        plt.plot(U.mat[:,i],label=str(i))
        plt.legend()
    plt.show()
    for k in [0,3,4,7,8,11,12]:
        plt.plot(U.mat[:,k],label=str(k))
        plt.legend()
    plt.show()
    """
