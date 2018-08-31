# learn the coefficient
import numpy as np
from parm import Para as P
from tprod import *
import math


def t2m(A):
    sz_A = A.shape
    dim = np.zeros((2))
    dim[0] = sz_A[0]*sz_A[2]
    dim[1] = sz_A[1]*sz_A[2]
    Ac = np.zeros((int(dim[0]),int(dim[1])))
    Amat = np.reshape(np.transpose(A,[0,2,1]),[int(dim[0]),sz_A[1]],order = 'F')
    for k in range(1,sz_A[2]):
        Ac[:,sz_A[1]*k:(k+1)*sz_A[1]] = np.roll(Amat,sz_A[0]*k,axis = 0)

    return Ac



def tsta(*args):
    if len(args) == 3:
        Xc = args[0]
        P = args[1]
        D0 = args[2]
        size_Xc = Xc.shape
        B0 = np.zeros((P.r,size_Xc[1],size_Xc[2]))
    else :
        Xc = args[0]
        P = args[1]
        D0 = args[2]
        B0 = args[3]
    r = P.r
    maxiter = P.maxiter
    beta = P.beta
    eta = P.eta
    D0tD0 = tensor_prod(D0,'t',D0,'a')
    D0c  = t2m(D0tD0)
    L0 = np.linalg.norm(D0c, ord=2)
    D0tX = tensor_prod(D0,'t',Xc,'a')
    C1 = B0
    t1=1;
    for iter in range(1,maxiter+1):
        L1 = eta**iter*L0
        gradC1 = tensor_prod(D0tD0,'t',C1,'a') - D0tX
        Temp = C1 - gradC1/L1
        B1 = np.sign(Temp)*np.maximum(abs(Temp)-beta/L1,0)
        t2 = (1+math.sqrt(1+4*t1**2))/2; 
        C1 = B1 + ((t1 - 1)/t2)*(B1 - B0)
        B0  = B1
        t1 = t2

    B = B1

    return B
