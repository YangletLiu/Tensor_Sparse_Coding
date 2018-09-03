# change the size
# recovery the size 
import numpy as np
from parm import Para as P
def t2b(X,P):
    patsize = P.patsize
    step = P.step
    sz = np.shape(X)
    TotalpatNum =int((np.floor((sz[0]-patsize)/step)+1)*(np.floor((sz[1]-patsize)/step)+1)*(np.floor((sz[2]-patsize)/step)+1))
    Z = np.zeros([patsize,patsize,patsize,TotalpatNum])
    for i in range(patsize):
        for j in range(patsize):
            for k in range(patsize):
                tempPatch = X[i:sz[0]-patsize+i+1,j:sz[1]-patsize+j+1,k:sz[2]-patsize+k+1][::step,::step,::step]
                Z[i,j,k,:] =np.reshape(tempPatch,[1,TotalpatNum],order = 'F')

    Y = np.transpose(np.reshape(Z,[patsize*patsize,patsize,TotalpatNum],order = 'F'),[0,2,1])
    return Y

   
def b2t(lu,P,size_X):
    patsize = P.patsize
    step = P.step
    TempR = int(np.floor((size_X[0]-patsize)/step))+1
    TempC = int(np.floor((size_X[1]-patsize)/step))+1
    TempS = int(np.floor((size_X[2]-patsize)/step))+1
    TempOffsetR = np.arange(0,(TempR-1)*step+1,step)
    TempOffsetC = np.arange(0,(TempC-1)*step+1,step)
    TempOffsetS = np.arange(0,(TempS-1)*step+1,step)
    xx = np.size(TempOffsetR)
    yy = np.size(TempOffsetC)
    zz = np.size(TempOffsetS)
    
    E_V = np.zeros(size_X)
    Weight = np.zeros(size_X)
    N = lu.shape[1]
    ZPat = np.reshape(np.transpose(lu,[0,2,1]),[patsize,patsize,patsize,N],order = 'F')
    for i in range(patsize):
        for j in range(patsize):
            for k in range(patsize):
                E_V[i:(TempR-1)*step+1+i,j:(TempC-1)*step+1+j,k:(TempS-1)*step+1+k][::step,::step,::step] += np.reshape(ZPat[i,j,k,:],[xx,yy,zz],order = 'F') 
                Weight[i:(TempR-1)*step+1+i,j:(TempC-1)*step+1+j,k:(TempS-1)*step+1+k][::step,::step,::step] += np.ones([xx,yy,zz])

    E_V = E_V/(Weight+np.spacing(1))
    return E_V

if __name__ == '__main__':

    han = np.random.rand(101,101,31)
    print(t2b(han,P).shape)
    print('nihao')
    ll = np.random.rand(25,33614,5)
    ss= np.array([101,101,31])
    print(b2t(ll,P,ss).shape)

