# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma

def FT_mat(vec):
    # spectl limits
    min_sp = -(len(vec)-1)/2
    max_sp = (len(vec)-1)/2
    spectls = np.arange(min_sp,max_sp+1)*(2.*np.pi/len(vec))
    # FT matrix
    F = np.array([np.exp(-1.j*k*np.arange(len(vec))) for k in spectls])
    return F



def FT_diff_mat(vec,alpha,period):
    """
    fractional order(alpha) different with Fourier transform
    """
    # spectl limits
    min_sp = -(len(vec)-1)/2
    max_sp = (len(vec)-1)/2
    sp_num = np.roll(np.arange(min_sp,max_sp+1),max_sp+1)
    splist = np.arange(len(vec))*2.*np.pi/len(vec)
    spectls = np.roll(np.arange(min_sp,max_sp+1)*(2.*np.pi/len(vec)),max_sp+1)

    # FT matrix
    F = np.array([np.exp(-1.j*k*np.arange(len(vec))) for k in splist])
    # iverse FT matrix
    #iF = np.linalg.inv(F)#/len(vec)
    iF = np.array([np.exp(1.j*k*np.arange(len(vec))) for k in splist])/float(len(vec))
    # matrix D^alpha
    D_alpha = abs(spectls*len(vec)/period)**alpha * np.identity(len(vec))
    """
    dlist_p = np.array([2*np.pi*1.j/period * k for k in range((len(vec)+1)/2)])
    dlist_n = np.array([-2*np.pi*1.j/period * (k+1) for k in range((len(vec)-1)/2)])
    dlist = np.r_[abs(dlist_p)**alpha, abs(dlist_n[::-1])**alpha]
    D_alpha = np.diag(dlist)
    """

    DF = np.dot(D_alpha,F)
    iFDF = np.dot(iF,DF)
    return iFDF


def cof_seq(alpha, sp_size):
    alpha = float(alpha)
    c = np.array([0. for i in xrange(sp_size/2+1)])
    c[0] = (gamma(alpha+1))/(gamma(1.+alpha/2.)*gamma(1+alpha/2.))
    for k in range(sp_size/2):
        c[k+1] = ((2*k-alpha)/(2*k+alpha+2))*c[k]
    c_conec = np.r_[c,c[-1:0:-1]]
    return c_conec


def grunwald_diff_mat(vec,alpha):
    sp_size = len(vec)
    C = np.zeros((sp_size, sp_size), dtype=np.float)
    for k in range(len(C)):
        C[k,:] = np.roll(cof_seq(alpha, sp_size-1), k)
    return C

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("qt5agg")
    from matplotlib import pyplot as plt

    alpha = 1.4
    xs = np.linspace(-np.pi,np.pi,400+1,endpoint=False)
    h = abs(xs[0]-xs[1])
    vec = 1./np.cosh(xs)*2
    F = FT_mat(vec)
    Fu = np.dot(F,vec)
    iFDF = FT_diff_mat(vec,alpha,2.*np.pi)
    df = np.dot(iFDF,vec)#*(len(vec))
    C = grunwald_diff_mat(vec,alpha) * h**-alpha
    cf = np.dot(C,vec)
    plt.plot(xs,abs(Fu))
    plt.show()
    plt.plot(xs,np.angle(Fu))
    plt.show()


