# tensor prod

import numpy as np

# if ch1 == t ch2 != t A'*B
# if ch1 != t ch2 == t A*B'
# if ch1 == t ch2 == t A'*B'
# if ch1 != t ch2 != t A*B
def tensor_prod(A,ch1,B,ch2):
    sz_A = np.shape(A)
    sz_B = np.shape(B)
    sz = np.zeros(3)  #result
    sz[2] = sz_B[2]
    if ch1 == 't':
        sz[0] = sz_A[1]
    else :
        sz[0] = sz_A[0]

    if ch2 == 't':
        sz[1] = sz_B[0]
    else :
        sz[1] = sz_B[1]

    print(sz.shape)
    sz= sz.astype(np.int32)
    print(sz.dtype)
    chat = np.zeros(sz,dtype = complex)
    ahat = np.fft.fft(A,axis = -1)
    bhat = np.fft.fft(B,axis = -1)
    if ch1 == 't' and ch2 == 't':
        for k in range(sz[2]):
            chat[:,:,k] =np.dot(np.conj(ahat[:,:,k]),T,np.conj(bhat[:,:,k]).T)
    elif ch1 == 't':
        for k in range(sz[2]):
            chat[:,:,k] =np.dot(np.conj(ahat[:,:,k]).T,bhat[:,:,k])
    elif ch2 =='t':
        for k in range(sz[2]):
            chat[:,:,k] =np.dot(ahat[:,:,k],np.conj(bhat[:,:,k]).T)
    else : 
        for k in range(sz[2]):
            chat[:,:,k] =np.dot(ahat[:,:,k],bhat[:,:,k])
    
    C = np.real(np.fft.ifft(chat,axis=-1))
    return C


if __name__ == '__main__':
    A = np.random.randint(0,5,size=[2,3,2])
    B = np.random.randint(0,5,size=[3,2,2])
    print(A[:,:,0])
    print(A[:,:,1])
    print(B[:,:,0])
    print(B[:,:,1])
    C = tensor_prod(A,'a',B,'b')
    print(C[:,:,0])
    print(C[:,:,0])
