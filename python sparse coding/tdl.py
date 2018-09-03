# tensor base learning
import scipy.optimize as sco
import numpy as np
from parm import Para as P

def tendl(Xhat,S,P):
    r = P.r
    Shat = np.fft.fft(S,axis = -1)
    dual_lambda = 10*abs(np.random.randn(r))
    m,_,k = Xhat.shape
    SSt = np.zeros((r,r,k))
    SSt = np.fft.fft(SSt,axis = -1)
    XSt = np.zeros((m,r,k))
    XSt = np.fft.fft(XSt,axis = -1)
    for kk in range(k):
        xhatk = Xhat[:,:,kk]
        shatk = Shat[:,:,kk]

        SSt[:,:,kk] = np.dot(shatk,np.conj(shatk).T)
        XSt[:,:,kk] = np.dot(xhatk,np.conj(shatk).T)

    # optimise return x
    bnds = tuple((0,np.infty) for i in range(len(dual_lambda))) 
    fun = lambda x :fobj(x,XSt,SSt,k)
    res = sco.minimize(fun,dual_lambda,method = 'L-BFGS-B',bounds = bnds)

    Lambda = np.diag(res.x)
    Bhat = np.zeros((m,r,k))
    Bhat = np.fft.fft(Bhat,axis = -1)
    for kk in range(k):
        SStk = SSt[:,:,kk]
        XStk = XSt[:,:,kk]
        Bhatkt =np.dot(np.linalg.pinv(SStk+Lambda),np.conj(XStk).T)
        Bhat[:,:,kk] = np.conj(Bhatkt).T

    B = np.fft.ifft(Bhat,axis = -1)
    B[np.where(np.isnan(B) == True)] = 0
    B = np.real(B)

    return B

def fobj(lam,XSt,SSt,k):
    m = XSt.shape[0]
    r = np.size(lam)
    Lam = np.diag(lam)
    f = 0
    for kk in range(k):
        XStk = XSt[:,:,kk]
        SStk = SSt[:,:,kk]
        SSt_inv = np.linalg.pinv(SStk+Lam)
        if m>r:
            f = f+np.trace(np.dot(SSt_inv,np.dot(np.conj(XStk).T,XStk)))
        else :
            f = f+np.trace(np.dot(np.dot(XStk,SSt_inv),np.conj(XStk.T)))
    f = np.real(f+k*sum(lam))

    return f

if __name__ == '__main__':
    ss = np.random.rand(30,33614,25)
    xx = np.random.rand(25,33614,25)
    xx_h = np.fft.fft(xx,axis=-1)
    print(tendl(xx_h,ss,P).shape)

