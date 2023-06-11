import numpy as np
from types import SimpleNamespace

def Initialize_NMPC():
    # Snake Model
    Nl = 5
    l = 0.18 / 2
    m = 1.56
    J = 1/3 * m * l**2
    ct = 4.45
    cn = 17.3
    mun = 2.57
    lambda1 = 5.26e-8
    lambda2 = 0.0120
    lambda3 = 8.1160e-04
    Vxc = -0.00
    Vyc = -0.00

    # NMPC Parameters
    N = 20
    M = 4
    T = 5
    Nstates = 2 * Nl + 4
    Ncontrols = Nl - 1
    x0 = np.zeros(2 * Nl + 4)

    # Store parameters for the use of constraint and objective function
    auxdata = SimpleNamespace(N=N, M=M, T=T, Nstates=Nstates, Ncontrols=Ncontrols, x0=x0)

    SnakeModel = SimpleNamespace(Nl=Nl, l=l, m=m, J=J, ct=ct, cn=cn, mun=mun, lambda1=lambda1,
                                 lambda2=lambda2, lambda3=lambda3, Vxc=Vxc, Vyc=Vyc)

    A = np.diag(np.ones(Nl), 0) + np.diag(np.ones(Nl - 1), 1)
    A = A[:Nl - 1, :]
    SnakeModel.A = A

    D = np.diag(np.ones(Nl), 0) - np.diag(np.ones(Nl - 1), 1)
    D = D[:Nl - 1, :]
    SnakeModel.D = D

    H = np.zeros((2 * Nl, 2 * Nl))
    for i in range(Nl):
        H[i, i:Nl] = 1
        H[Nl + i, Nl + i:2 * Nl] = 1
    SnakeModel.H = H
    SnakeModel.H_inv = np.linalg.inv(H)

    H_inv2 = np.zeros((Ncontrols, Nstates))
    H_inv2[:Nl - 1, :Nl] = SnakeModel.H_inv[:Nl - 1, :Nl]
    SnakeModel.H_inv2 = H_inv2

    SnakeModel.e = np.ones(Nl)
    SnakeModel.K = A.T @ (np.linalg.inv(D @ D.T) @ D)
    SnakeModel.V = A.T @ (np.linalg.inv(D @ D.T) @ A)

    return auxdata, SnakeModel