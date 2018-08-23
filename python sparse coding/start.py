# where the code start
from parm import Para as P
import scipy.io as sio
import numpy as np
from tenor2block import * 
from initbase import *
def denoise(X):
    Xc = t2b(X,P)
    print(Xc.shape)
    size_Xc = Xc.shape  #(25,33614,5)
    Xhat = np.fft.fft(Xc,axis=-1)  #fft along 3rd
    D0 = init3D(P)
    B = np.zeros((P.r,size_Xc[1],size_Xc[2]))
    for i in range(P.maxiter):
        B = tsta()
        D0 = tendl()
        b = tsta()
        lu = tensor_prod()




    



    





if __name__ =='__main__':
    X = sio.loadmat('caseandresult/origin.mat')
    X = X['noisy_msi']
    denoise(X)
