# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use("qt5agg")
from matplotlib import pyplot as plt
from solmat import Solmat
from fch_solve import cal_mass


def load_csv(str):
    return np.loadtxt(fname,delimiter=",")

if __name__ == '__main__':
    fname = "./csv/CH_u_2.0_T3.0_h0.01_dt0.001.csv"
    #fname = "./csv/CH_u_2.0_T10.0_h0.02_dt0.0001.csv"
    U = Solmat(2.0,100,3000,0.,1.,3.0,1.)
    U.mat = load_csv(fname)
    print(len(U.mat[:,0]))
    num_frames = U.time_step
    #U.plot_Animation(20,num_frames,save=False)
    # calcurate discret Mass
    U.set_Mass(cal_mass(U,"neumann"))
    U.plot_Mass()
    U.set_Mass(cal_mass(U,"periodic"))
    U.plot_Mass()

    """

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
