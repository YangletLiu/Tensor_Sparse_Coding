# change the size 
import numpy as np

def t2b(X,P):
    patsize = P.patsize
    step = P.step
    sz = np.shape(X)
    print(sz)
    TotalpatNum =int((np.floor((sz[0]-patsize)/step)+1)*(np.floor((sz[1]-patsize)/step)+1)*(np.floor((sz[2]-patsize)/step)+1))
    Z = np.zeros([patsize,patsize,patsize,TotalpatNum])
    for i in range(patsize):
        for j in range(patsize):
            for k in range(patsize):
                tempPatch = X[i:sz[0]-patsize+i+1,j:sz[1]-patsize+j+1,k:sz[2]-patsize+k+1][::step,::step,::step]
                Z[i,j,k,:] =np.reshape(tempPatch,[1,TotalpatNum],order = 'F')

    Y = np.transpose(np.reshape(Z,[patsize*patsize,patsize,TotalpatNum],order = 'F'),[0,2,1])
    return Y

    


