# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("qt5agg")
from matplotlib import pyplot as plt

from solmat import Solmat
import fractional_diff as f_diff
import copy

import numpy as np
from scipy import linalg
from scipy import optimize


def sech(x):
    return 1. / np.cosh(x)


def ini_func(x):
    #return np.exp(1.j*(x**2))*sech(x)
    #return 4*sech(2.*(x-10.))*np.exp(1.j*x)+2.*sech(x-20.)*np.exp(1.j*x/2.)
    #return 0.1*np.sin(2.*np.pi*x) + 0.01*np.cos(4.*np.pi*x) \
    #        + 0.06*np.sin(4.*np.pi*x) + 0.02*np.cos(10.*np.pi*x)
    return np.sin(np.pi*x)


def first_step(solmat):
    I = np.identity(len(solmat.mat[:, 0]))
    iFDF = f_diff.FT_diff_mat(solmat.mat, solmat.alpha, solmat.period)
    f_step = np.dot(I - solmat.dt * 1.j * iFDF + solmat.dt *
                    1.j * abs(solmat.mat[:, 0])**2 * I, solmat.mat[:, 0])
    return f_step


def CH_conserved_scheam(solmat):
    for n in np.arange(solmat.time_step - 1):
    #for n in np.arange(10):
        D_2 = f_diff.diff_2_p(solmat.mat[:,0])
        x = optimize.root(DFD_eq,[0]*solmat.sp_size,
            args=(
                solmat.mat[:,n],
                solmat.h,
                solmat.dt,
                D_2,
                solmat.alpha,
                solmat.period,
                True
            ),
            tol=1e-12,
            method="hybr")
        solmat.mat[:,n+1] = x.x
        if n % 10 == 0:
            print("alpha=", solmat.alpha, n, "loop\n")

def cb_func(x,f):
    print(f)

def DFD_eq(x,u,dx,dt,D_2,alpha,period,frac=True):
    p = -1.0
    q = -0.0005
    r = 1.0
    u_half = (x+u)/2.
    if frac :
        f = (x - u) + f_diff.FFT_diff(u_half,alpha,period).real*dt
    else:
        f = (x - u)*(dx**2) - np.dot(f_diff.diff_2_n(u),DV)*dt
    return f

def fnls_Wang_scheam(solmat):
    I = np.identity(solmat.sp_size-2)
    C = f_diff.grunwald_diff_mat(solmat.mat[1:-1], solmat.alpha)*solmat.h**(-solmat.alpha)
    for n in np.arange(solmat.time_step - 2):
        dtmat = 1.j * solmat.dt * \
            ((C) - (solmat.beta * (np.abs(solmat.mat[1:-1, n+1]))**2 * I))
        A = I + dtmat
        b = np.dot(I - dtmat, solmat.mat[1:-1, n])
        x = linalg.solve(A, b)
        solmat.mat[1:-1, n + 2] = x[:]
        if n % 10 == 0:
            print("alpha=", solmat.alpha, n, "loop\n")


def cal_mass(solmat,bc):
    if bc=="neumann" :
        Q_vec = (solmat.mat.sum(axis=0) - (solmat.mat[0,:]+solmat.mat[-1,:])/2.)*(solmat.h)
    elif bc=="periodic" :
        Q_vec = solmat.mat.sum(axis=0)*(solmat.h)

    #Q = np.linalg.norm(solmat.mat, ord=2, axis=0)**2
    #Q_vec = (Q[:-1] + Q[1:]) / 2.
    return Q_vec


def cal_energy(solmat,p,q,r):
    G = (1./2.)*(solmat.mat**2)
    G_sum = (G.sum(axis=0)) * solmat.h
    return G_sum



def cal_energy_wang(solmat):
    C = f_diff.grunwald_diff_mat(
        solmat.mat, solmat.alpha / 2.)*solmat.h**-solmat.alpha
    dU = np.dot(C, solmat.mat)

    E1 = ((np.linalg.norm(dU[:, 1:], axis=0))**2 +
          (np.linalg.norm(dU[:, :-1], axis=0))**2) / 2.
    E2 = -solmat.beta / 2. * \
        np.linalg.norm(solmat.mat[:, :-1] * solmat.mat[:, 1:], axis=0)**2
    E_vec = (E1 + E2) * solmat.h
    return E_vec



if __name__ == '__main__':
    import sys

    param = sys.argv
    # global var
    g_vals = {
        "alpha": 1.5,
        "beta": 1.,
        "L": 0.,  # 左区間
        "R": 1.,  # 右区間
        "M": 100,  # 空間分割数
        "Tmax":1.,  # 秒数[s]
        "N": int(1.*1000)  # ステップ数
    }
    """
    dx=0.02
    dt=0.0001
    """
    g_vals.update({"h": (g_vals["R"] - g_vals["L"]) / g_vals["M"],  # space-step
                   "dt": g_vals["Tmax"] / g_vals["N"],  # time-step
                   # 空間離散点
                   "xs": np.linspace(g_vals["L"], g_vals["R"], g_vals["M"] + 1, endpoint=False),
                   # 時間離散点
                   "ts": np.linspace(0., g_vals["Tmax"], g_vals["N"], endpoint=True),
                   "period": g_vals["R"] - g_vals["L"]
                   })

    print(""" 
    Tmax={Tmax}
    dx={h}
    dt={dt}
    """.format(**g_vals))

    # create Solution matrix
    U = Solmat(g_vals["alpha"], g_vals["M"], g_vals["N"],
               g_vals["L"], g_vals["R"], g_vals["Tmax"], g_vals["beta"])

    # initialize solmat
    U.initialize(ini_func)

    # create fractional order diff matrix
    #iFDF = f_diff.FT_diff_mat(U.mat, U.alpha, g_vals["period"])
    #C = f_diff.grunwald_diff_mat(U.mat, U.alpha)*U.h **-U.alpha
    V = copy.deepcopy(U)

    
    # fnls linear conservative scheam with fourier transform
    CH_conserved_scheam(U)
    #fnls_Wang_scheam(V)


    # file_operating
    import os

    if os.path.exists("csv") and os.path.isdir("csv"):
        pass
    else:
        os.mkdir("csv")
    if os.path.exists("fig") and os.path.isdir("fig"):
        pass
    else:
        os.mkdir("fig")
    os.chdir("csv")
    np.savetxt("CH_u_{alpha}_T{Tmax}_h{h}_dt{dt}.csv".format(**g_vals), U.mat.view(float), delimiter=",")
    os.chdir("..")




    # plot solmat as 3D figure
    ###U.plot3D(save=True,diff_way="CH")
    #V.plot3D(save=False,diff_way="G")

    # calcurate discret Mass
    U.set_Mass(cal_mass(U,bc="periodic"))
    #V.set_Mass(cal_mass(V))
    U.plot_Mass()

    # calcurate discret Energy
    ###U.set_Energy(cal_energy(U))
    ###U.plot_Energy()
    #V.set_Energy(cal_energy_wang(V))
    """
    """
    #U.ene_comp_plot(V)
    """
    print U.mat[:10,0]
    print U.mat[:10,1]
    """
    # plot Animation |U|
    ###U.plot_Animation(10)
    """
    for k in [0,1,2,5,9]:
        plt.plot(U.xs,U.mat[:,k],label=str(k))
        plt.legend()
    plt.show()

    for k in [0,100,500,1000,5000,9000,-1]:
        plt.plot(U.xs,U.mat[:,k],label=str(k))
        plt.legend()
    plt.show()
    # calcurate discret Mass
    U.set_Mass(cal_mass(U))
    U.plot_Mass()
    """
