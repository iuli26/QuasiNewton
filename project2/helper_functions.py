import numpy as  np

def gradient_approx(f, x, h=1e-6):
    n = len(x)

    grad = np.zeros(n)
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = h
        grad[j] = (f(x+dx) - f(x-dx))/(2*h)
    return grad

def Hessian(f, x, h=1e-6):

    n = len(x)
    hess = np.zeros((n, n))
    for j in range (n):
        dxj = np.zeros(n)
        dxj[j] = h
        for k in range(n):
            dxk = np.zeros(n)
            dxk[k] = h
            hess[j][k] = 1/(4*h**2) * (f(x+dxj+dxk) - f(x - dxk + dxj) - f(x+dxk-dxj) + f(x-dxk-dxj))

    hess = 0.5 * (hess + hess.T)
    return hess


