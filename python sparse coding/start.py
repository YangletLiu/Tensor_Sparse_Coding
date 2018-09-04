# where the code start
from parm import Para as P
import scipy.io as sio
import numpy as np
from tenor2block import * 
from initbase import *
from tsa import *
from tprod import *
from tdl import *
from psnr3d import *
import matplotlib.pyplot as plt
import os

def denoise(OX):
    plt.figure(figsize = (10,10))    
    plt.subplot(3,4,1),plt.title('origin')
    plt.imshow(OX[:,:,1],cmap = 'gray'),plt.axis('off')

    size_OX = OX.shape
    X = OX+0.2*np.random.rand(size_OX[0],size_OX[1],size_OX[2])
     
    plt.subplot(3,4,2),plt.title('dirty')
    plt.imshow(X[:,:,1],cmap = 'gray'),plt.axis('off')

    size_X = X.shape
    Xc = t2b(X,P)
    size_Xc = Xc.shape  #(25,33614,5)
    Xhat = np.fft.fft(Xc,axis=-1)  #fft along 3rd
    D0 = init3D(P)
    # defalt = np.zeros((P.r,size_Xc[1],size_Xc[2]))
    if not os.path.exists('./result'):
        os.mkdir('./result')
    for i in range(P.maxiter):
        if i == 0:
            B = tsta(Xc,P,D0)
        else :
            B = tsta(Xc,P,D0,B0)
        D0 = tendl(Xhat,B,P)
        B0 = tsta(Xc,P,D0)
        lu = tensor_prod(D0,'a',B0,'a')
        emsi = b2t(lu,P,size_X)

        plt.subplot(3,4,i+3),plt.title(str(i))
        plt.imshow(emsi[:,:,1],cmap = 'gray'),plt.axis('off')

        ps = psnr(OX*255,emsi*255)
        print('iter={},current PSNR = {}'.format(i,ps)) 
    
    plt.savefig('./result/sparsecoding.png')
    plt.show()


if __name__ =='__main__':
    OX = sio.loadmat('caseandresult/Omsi.mat')
    OX = OX['Omsi']
    denoise(OX)
