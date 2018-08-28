# where the code start
from parm import Para as P
import scipy.io as sio
import numpy as np
from tenor2block import * 
from initbase import *
from tsa import *
def denoise(X):
    size_X = X.shape
    Xc = t2b(X,P)
    print(Xc.shape)
    size_Xc = Xc.shape  #(25,33614,5)
    Xhat = np.fft.fft(Xc,axis=-1)  #fft along 3rd
    D0 = init3D(P)
    # defalt = np.zeros((P.r,size_Xc[1],size_Xc[2]))
    for i in range(P.maxiter):
        if i == 0:
            B = tsta(Xc,P,D0)
        else :
            B = tsta(Xc,P,D0,B0)
        D0 = tendl(Xhat,B,P)
        B0 = tsta(Xc,P,D0)
        lu = tensor_prod(D0,'a',B0,'a')
        emsi = b2t(lu,P,size_X)
        ps = psnr(emsi*255,OX*255)
        print('iter={},current PSNR = {}'.format(i,ps)) 





if __name__ =='__main__':
    OX = sio.loadmat('caseandresult/Omsi.mat')
    OX = OX['Omsi']
    size_OX = OX.shape
    X = OX+0.2*np.random.rand(size_OX[0],size_OX[1],size_OX[2])
    denoise(X)
