# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("qt5agg")
from matplotlib import pyplot as plt


import fractional_diff

import numpy as np
import copy

from scipy import linalg
from solmat import Solmat


def sech(x):
    return 1. / np.cosh(x)


def ini_func(x):
    #return np.exp(1.j*(x**2))*sech(x)
    #return np.exp(1.j*x*2.)*sech(x)
    #return 4*sech(2.*(x-10.))*np.exp(1.j*x)+2.*sech(x-20.)*np.exp(1.j*x/2.)
    return sech(x+10.)*np.exp(2*1.j*(x+10.)) + sech(x-10.)*np.exp(-2*1.j*(x-10.))


def first_step(solmat):
    I = np.identity(len(solmat.mat[:, 0]))
    iFDF = fractional_diff.FT_diff_mat(solmat.mat, solmat.alpha, solmat.period)
    f_step = np.dot(I - solmat.dt * 1.j * iFDF + solmat.dt *
                    1.j * abs(solmat.mat[:, 0])**2 * I, solmat.mat[:, 0])
    return f_step


def fnls_FT_conserved_scheam(solmat):
    I = np.identity(solmat.sp_size)
    iFDF = fractional_diff.FT_diff_mat(solmat.mat, solmat.alpha, solmat.period)
    for n in np.arange(solmat.time_step - 2):
        dtmat = 1.j * solmat.dt * \
            ((iFDF) - (solmat.beta * (np.abs(solmat.mat[:, n + 1]))**2 * I))
        A = I + dtmat
        b = np.dot(I - dtmat, solmat.mat[:, n])
        x = linalg.solve(A, b)
        solmat.mat[:, n + 2] = x[:]
        if n % 100 == 0:
            print("eigvals of A =",max(np.abs(np.linalg.eigvals(dtmat))))
            print "alpha=", solmat.alpha, n, "loop\n"


def fnls_Wang_scheam(solmat):
    I = np.identity(solmat.sp_size-2)
    C = fractional_diff.grunwald_diff_mat(solmat.mat[1:-1], solmat.alpha)*solmat.h**(-solmat.alpha)
    for n in np.arange(solmat.time_step - 2):
        dtmat = 1.j * solmat.dt * \
            ((C) - (solmat.beta * (np.abs(solmat.mat[1:-1, n+1]))**2 * I))
        A = I + dtmat
        b = np.dot(I - dtmat, solmat.mat[1:-1, n])
        x = linalg.solve(A, b)
        solmat.mat[1:-1, n + 2] = x[:]
        if n % 100 == 0:
            print "alpha=", solmat.alpha, n, "loop\n"


def cal_mass(solmat):
    Q = np.linalg.norm(solmat.mat, ord=2, axis=0)**2
    Q_vec = (Q[:-1] + Q[1:]) / 2.
    return Q_vec


def cal_energy(solmat):
    iFDF = fractional_diff.FT_diff_mat(
        solmat.mat, solmat.alpha / 2., solmat.period)
    dU = np.dot(iFDF, solmat.mat)

    E1 = ((np.linalg.norm(dU[:, 1:], axis=0))**2 +
          (np.linalg.norm(dU[:, :-1], axis=0))**2) / 2.
    E2 = -solmat.beta / 2. * \
        np.linalg.norm(solmat.mat[:, :-1] * solmat.mat[:, 1:], axis=0)**2
    E_vec = (E1 + E2) * solmat.h
    return E_vec


def cal_energy_wang(solmat):
    C = fractional_diff.grunwald_diff_mat(
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
        "alpha": 2.0,
        "beta": 2.,
        "L": -20.,  # 左区間
        "R": 20,  # 右区間
        "Tmax": 40.,  # 秒数[s]
        "M": 800,  # 空間分割数
        "N": 2000  # ステップ数
    }
    g_vals.update({"h": (g_vals["R"] - g_vals["L"]) / g_vals["M"],  # space-step
                   "dt": g_vals["Tmax"] / g_vals["N"],  # time-step
                   # 空間離散点
                   "xs": np.linspace(g_vals["L"], g_vals["R"], g_vals["M"] + 1, endpoint=False),
                   # 時間離散点
                   "ts": np.linspace(0., g_vals["Tmax"], g_vals["N"], endpoint=True),
                   "period": g_vals["R"] - g_vals["L"]
                   })

    print """ 
    Tmax={Tmax}
    dx={h}
    dt={dt}
    """.format(**g_vals)

    # create Solution matrix
    U = Solmat(g_vals["alpha"], g_vals["M"] + 1, g_vals["N"],
               g_vals["L"], g_vals["R"], g_vals["Tmax"], g_vals["beta"])

    # initialize solmat
    U.initialize(ini_func)

    # gain first step
    U.first_step(first_step)

    # create fractional order diff matrix
    #iFDF = fractional_diff.FT_diff_mat(U.mat, U.alpha, g_vals["period"])
    #C = fractional_diff.grunwald_diff_mat(U.mat, U.alpha)*U.h **-U.alpha
    V = copy.deepcopy(U)

    
    # fnls linear conservative scheam with fourier transform
    fnls_FT_conserved_scheam(U)
    #fnls_Wang_scheam(V)

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
    np.savetxt("u_{alpha}_T{Tmax}_h{h}_dt{dt}.csv".format(**g_vals), U.mat.view(float), delimiter=",")
    os.chdir("..")

    # plot solmat as 3D figure
    #U.plot3D(save=False,diff_way="F")
    #V.plot3D(save=False,diff_way="G")

    # plot Animation |U|
    U.plot_Animation(3)

    # calcurate discret Energy
    U.set_Mass(cal_mass(U))
    #V.set_Mass(cal_mass(V))
    U.plot_Mass()

    # calcurate discret Energy
    U.set_Energy(cal_energy(U))
    U.plot_Energy()
    #V.set_Energy(cal_energy_wang(V))
    """
    """
    #U.ene_comp_plot(V)
    """
    print U.mat[:10,0]
    print U.mat[:10,1]
    """
