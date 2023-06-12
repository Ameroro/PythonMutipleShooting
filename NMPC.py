import casadi as ca
import numpy as np
import Init_NMPC3
import matplotlib.pyplot as plt


def dynamics(x, u, SnakeModel):
    Nl = SnakeModel.Nl
    l = SnakeModel.l
    m = SnakeModel.m
    J = SnakeModel.J
    ct = SnakeModel.ct
    cn = SnakeModel.cn
    e = SnakeModel.e
    K = SnakeModel.K
    V = SnakeModel.V
    D = SnakeModel.D
    lambda2 = SnakeModel.lambda2

    St = ca.diag(ca.MX.sin(x[:Nl]))
    Ct = ca.diag(ca.MX.cos(x[:Nl]))
    Mt = J * ca.MX.eye(Nl) + m * (l**2) * St @ V @ St + m * (l**2) * Ct @ V @ Ct
    Wt = (m * (l**2) * St @ V @ Ct - m * (l**2) * Ct @ V @ St) @ (x[Nl:2*Nl]**2)
    Xd = l * K.T @ (St @ x[Nl:2*Nl]) + e @ x[2*Nl+2]
    Yd = -l * K.T @ (Ct @ x[Nl:2*Nl]) + e @ x[2*Nl+3]
    Fx = (ct * (Ct**2) + cn * (St**2)) @ Xd + (ct - cn) * (St @ Ct) @ Yd
    Fy = (ct - cn) * St @ Ct @ Xd + (ct * (St**2) + cn * (Ct**2)) @ Yd
    Tr = -lambda2 * x[Nl:2*Nl]

    dx = ca.MX.zeros(x.shape)

    print("dx[Nl:2*Nl] shape:", dx[Nl:2*Nl].shape)
    print("RHS shape:", (-Wt + Tr + l * St @ K @ Fx - l * Ct @ K @ Fy + D.T @ u).shape)
    print("Wt: ", Wt.shape)
    print("Mt: ", Mt.shape)
    print("Xd: ", Xd.shape)
    print("Yd: ", Yd.shape)
    print("Fx: ", Fx.shape)
    print("Fy: ", Fy.shape)
    print("Tr: ", Tr.shape)
    print(dx)

    
  
    dx[:Nl] = x[Nl:2*Nl]
    dx[Nl:2*Nl] = ca.mldivide(Mt, -Wt + Tr + l * St @ K @ Fx - l * Ct @ K @ Fy + D.T @ u)
    dx[2*Nl] = x[2*Nl+2]
    dx[2*Nl+1] = x[2*Nl+3]
    dx[2*Nl+2] = e.T @ Fx
    dx[2*Nl+3] = e.T @ Fy

    return dx


def gait_controller(x_init, u_init, phi_ref, auxdata, SnakeModel):

    opti = ca.Opti()
    opti.solver('ipopt')

    # Initialize the parameters
    auxdata, SnakeModel = Init_NMPC3.Initialize_NMPC()

    # System dimensions
    n = auxdata.Nstates # State dimension
    m = auxdata.Ncontrols  # Input dimension

    # Parameters
    delta = 0.1  # Sampling time
    T = 5  # Continuous-time prediction horizon
    N = int(T / delta)  # Discrete-time prediction horizon
    tmeasure = 0.0  # Initial time
    x0 = ca.DM(auxdata.x0)  # Initial state
    SimTime = 5  # Simulation time
    mpciterations = int(SimTime / delta)  # Discrete simulation time

    # Define optimization variables
    x = opti.variable(n, N+1)  # Predicted state sequence
    u = opti.variable(m, N)  # Predicted input sequence

    # Cost function weights
    Q = 5.0 * ca.MX.eye(SnakeModel.Nl-1)
    R = 0.1 * ca.MX.eye(SnakeModel.Nl-1)

    # Define parameters
    xt = opti.parameter(n, 1)
    u_m = opti.parameter(m, 1)

    
    # Dynamic constraint
    for k in range(N):
        k1 = dynamics(x[:, k], u[:, k], SnakeModel)
        k2 = dynamics(x[:, k] + delta / 2 * k1, u[:, k], SnakeModel)
        k3 = dynamics(x[:, k] + delta / 2 * k2, u[:, k], SnakeModel)
        k4 = dynamics(x[:, k] + delta * k3, u[:, k], SnakeModel)
        x_next = x[:, k] + delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        opti.subject_to(x_next == x[:, k+1])

    opti.set_initial(x, x_init)
    opti.set_initial(u, u_init)

    #Costfunction
    C = 0
    for k in range(N+1):
        C += (SnakeModel.H_inv2@x[:, k] - phi_ref[:, k]).T @ Q @ (SnakeModel.H_inv2@x[:, k] - phi_ref[:,k])

    for k in range(N):
        C +=  (u[:,k].T @ R @ u[:,k])
    opti.minimize(C)


    # Solve the optimization problem
    sol = opti.solve()
    x_OL = sol.value(x)
    u_OL = sol.value(u)

    u_new = np.array(u_OL[:,0])

    return u_new

    
