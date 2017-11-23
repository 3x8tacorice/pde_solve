# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import gamma
from scipy.fftpack import fft
from scipy.fftpack import ifft

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
    min_sp = -int((len(vec)-1)/2)
    max_sp = int((len(vec)-1)/2)
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

def diff_2_n(vec):
    D_2 = np.diag((len(vec)-1)*[1.],k=-1) \
            +np.diag(len(vec)*[-2.])\
            +np.diag((len(vec)-1)*[1.],k=1)
    D_2[0,1] = 2.
    D_2[-1,-2] = 2.
    return D_2


def diff_2_p(vec):
    D_2 = np.diag((len(vec)-1)*[1.],k=-1) \
            +np.diag(len(vec)*[-2.])\
            +np.diag((len(vec)-1)*[1.],k=1)
    D_2[0,-1] = 1.
    D_2[-1,0] = 1.
    return D_2


def FFT(vec):
    return fft(vec)


def iFFT(vec):
    return ifft(vec)

def FFT_diff(vec,alpha,period):
    # return difference of alpha order

    Fv = FFT(vec)

    id_sp = 2*np.pi*1.j/period
    right_sp = np.array([id_sp*k for k in np.arange(1,(int(len(vec))/2))])
    left_sp = -right_sp[::-1]
    diag = np.r_[[0],right_sp,[0],left_sp]
    D = np.diag(abs(diag)**alpha)
    


    """

    # spectl limits
    min_sp = -int((len(vec)-1)/2)
    max_sp = int((len(vec)-1)/2)
    sp_num = np.roll(np.arange(min_sp,max_sp+1),max_sp+1)
    splist = np.arange(len(vec))*2.*np.pi/len(vec)
    spectls = np.roll(np.arange(min_sp,max_sp+1)*(2.*np.pi/len(vec)),max_sp+1)
    D = abs(spectls*len(vec)/period)**alpha * np.identity(len(vec))
    """
    DFv = np.dot(D,Fv)
    iFDFv = iFFT(DFv)
    return iFDFv


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("qt5agg")
    from matplotlib import pyplot as plt

    xs = np.linspace(0.,1.,100,endpoint=False)
    f = np.sin(xs*2*np.pi)
    ft = -FFT_diff(f,2.0,1.)
    plt.plot(xs,ft.real,label="FFT")
    plt.legend()
    plt.show()
    """
    d2 = diff_2_p(f)
    F=FT_diff_mat(f,2.0,1.)
    print(ft)
    #plt.plot(xs,np.dot(F,f),label="FT")
    #plt.plot(xs,np.dot(d2,f)/((xs[1]-xs[0])**2),label="diff")
    #plt.plot(xs,((np.dot(d2,f)/((xs[1]-xs[0])**2))-ft)/abs(ft))
    #plt.show()
    alpha = 2
    period = 1.
    vec = f
    id_sp = 2*np.pi*1.j/period
    right_sp = np.array([id_sp*k for k in np.arange(1,(int(len(vec))/2))])
    left_sp = -right_sp[::-1]
    diag = np.r_[[0],right_sp,[0],left_sp]
    D = np.diag(abs(diag)**alpha)
    print(D)
    """
