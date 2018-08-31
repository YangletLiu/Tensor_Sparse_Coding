# compute the psnr 

import numpy as np
import math

def psnr(image1,image2):
    [m,n,k] = image1.shape
    [mm,nn,kk] = image2.shape
    m = min(m,mm)
    n = min(n,nn)
    k = min(k,kk)
    image1 = image1[0:m,0:n,0:k]
    image2 = image2[0:m,0:n,0:k]
    psn = 0
    for i in range(k):
        mse =np.square(image1[:,:,i]-image2[:,:,i]).sum()/m*n
        psn = psn+10*math.log10(255**2/mse)

    psn = psn/k

    return psn
