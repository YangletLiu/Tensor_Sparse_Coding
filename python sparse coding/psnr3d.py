# compute the psnr 

import numpy as np
import math

def psnr(image1,image2):

    [m,n,k] = image1.shape
    gg =0
    for kk in range(k):
        mse =np.square(image1[:,:,kk]-image2[:,:,kk]).sum()/(m*n)
        gg = gg + 10*math.log10(255**2/mse)
    
    gg = gg/k
    return gg

if __name__ == '__main__':
   a = np.random.randint(0,5,size = [2,3,2])
   b = np.random.randint(0,5,size = [2,3,2])
   print(a[:,:,0])
   print(a[:,:,1])
   print(b[:,:,0])
   print(b[:,:,1])
   ab= psnr(a,b)
  # a = np.array([[1,2,3],[1,2,3]])
  # print(a)
  # b = np.array([[2,2,2],[3,3,3]])
  # print(b)
  # ab = psnr(a,b)
   print(ab)


