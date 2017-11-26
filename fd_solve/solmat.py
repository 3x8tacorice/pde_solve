# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("qt5agg")
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from matplotlib import animation
import numpy as np


fp = FontProperties(fname=r'C:\Users\nakagawa\Documents\font\SpicaNeueFontFamily\SpicaNeueP-Regular.ttf', size=14)


class Solmat:

    """
    left =-2.  #左区間
    right = 2. #右区間
    beta = 2.
    Tmax = 50.0  #秒数[s]
    sp_size = 800  #空間分割数
    time_step = 10000 #ステップ数
    """
    def __init__(self,alpha,sp_size,time_step,left,right,Tmax,beta):
        self.sp_size = sp_size
        self.time_step = time_step
        self.left = left
        self.right = right
        self.Tmax = Tmax
        self.alpha = alpha
        self.beta = beta
        self.mat = np.zeros((self.sp_size,self.time_step),dtype=float)
        self.xs = np.linspace(self.left,self.right,self.sp_size,endpoint=False)  #空間離散点
        self.ts = np.linspace(0.,self.Tmax,self.time_step,endpoint=True) #時間離散点
        self.h = (self.right-self.left)/self.sp_size #space-step
        self.dt = self.Tmax/self.time_step #time-step
        self.period = self.right - self.left



    def initialize(self,func):
        self.mat[:,0] = func(self.xs)
    def first_step(self,func=(lambda x:x)):
        self.mat[:,1] = func(self)
    def set_period(self,period):
        self.period = period

    def set_Energy(self,E_vec):
        self.E_vec = E_vec
    
    def set_Mass(self,Q_vec):
        self.Q_vec = Q_vec

    def plot_Energy(self):
        err_E = self.E_vec[:-1]-self.E_vec[1:]
        plt.plot(self.ts[1:-1],err_E/abs(self.E_vec[1:]))
        plt.show()
    
    def plot_Mass(self):
        err_Q = self.Q_vec[:-1]-self.Q_vec[1:]
        #plt.plot(self.ts[:-1],err_Q/abs(self.Q_vec[1:]))
        plt.plot(self.ts[:-1],err_Q)
        plt.show()

    def plot3D(self,diff_way="F",save=False, cmap="summer"):
        X, T = np.meshgrid(self.xs,self.ts)
        rstride=100
        cstride=10
        fig = plt.figure(figsize=(10,8))
        ax = Axes3D(fig)
        #plt.title("$|U| \ \\alpha={0}$".format(self.alpha),fontsize="xx-large")
        
        if cmap == "jet"    : ax.plot_surface(X,T,abs(self.mat.T),rstride=2,cstride=2,cmap=cm.jet,linewidth=0,alpha=1.0,antialiased=False)
        if cmap == "summer" : ax.plot_surface(X,T,abs(self.mat.T),rstride=10,cstride=10,cmap=cm.summer,alpha=0.4)
        if cmap == None : ax.plot_surface(X,T,self.mat.T,rstride=1000,cstride=1,cmap=cm.binary,alpha=0.7)
        #ax.plot_surface(X,T,abs(U.T),color="c",rstride = rstride,cstride = cstride)

        ax.set_xlabel('$x$',fontsize="xx-large")
        ax.set_xlim(self.left, self.right)
        ax.set_ylabel('$t$',fontsize="xx-large")
        #ax.set_ylim(, 40)
        #plt.savefig("fig/{2}_abs_u_a{0}_{1}s_frame.png".format(self.alpha,self.Tmax,diff_way))
        if save:
            plt.savefig("fig/"+diff_way+"u_a{0}_T{1}_dt{2}.png".format(self.alpha,self.Tmax,self.dt),transparent=True)
            plt.close()
        else:
            plt.show()


    def ene_comp_plot(self,solmat):
        err_E = self.E_vec[:-1]-self.E_vec[1:]
        cerr_E = solmat.E_vec[:-1]-solmat.E_vec[1:]
        fig = plt.figure(figsize=(10,8))
        plt.plot(self.ts[1:-1],cerr_E/abs(solmat.E_vec[1:]),".:",color="#19647e",ms=8,lw=3.0,label=u"既存解法")
        plt.plot(self.ts[1:-1],err_E/abs(self.E_vec[1:]),".",color="red",ms=9,lw=3.0,label=u"提案法")
        #plt.legend(loc="lower right",prop=fp,fontsize="xx-large")
        plt.savefig("fig/energy_comp_a{0}_T{1}_dt{2}.png".format(self.alpha,self.Tmax,self.dt),transparent=True)
        plt.show()

    def plot_Animation(self,rate,num_frames,save):
        # Set up formatting for the movie files
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        fig = plt.figure(figsize=(10,8))
        rate = rate
        def update(i):
            if i !=0:
                plt.cla()
            plt.plot(self.xs, self.mat[:,i],color="blue")
            plt.ylim(-1.5,1.5)
            plt.title("CH eq"+str(i))

        ani = animation.FuncAnimation(fig, update,interval=rate, frames = num_frames, repeat_delay=3000)
        #if save:
        #    ani.save("test.mp4", writer=writer)
        plt.show()

def FT_diff_mat(Solmat,alpha):
    """
    fractional order(alpha) different with Fourier transform
    """

    # FT matrix
    F = np.array([])





if __name__ == '__main__':
    solmat = Solmat(1.4,400,1000,-20,20,100)
    print(solmat.xs)
    print(solmat.Tmax)
    print(solmat.sp_size)
    print(solmat.time_step)
    solmat.initialize(lambda x:np.exp(1.j*x*2.*np.pi/(solmat.right-solmat.left)))
    solmat.plot3D()