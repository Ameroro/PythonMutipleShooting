import casadi as ca
import numpy as np
import Init_NMPC3
import matplotlib.pyplot as plt

# Create an optimization problem
opti = ca.Opti()
opti.solver('ipopt')

# Initialize the parameters
auxdata, SnakeModel = Init_NMPC3.Initialize_NMPC()

# System parameters
mu = 0.5

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

#Q = ca.diag([0.5, 0.5])  # State weight matrix
#R = 1.0  # Input weight

# Input constraint
u_max_norm = 1.0  # Maximum input magnitude

# Define dynamics function

def dynamics2(x, u, SnakeModel):
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

    St = np.diag(np.sin(x[:Nl]))
    Ct = np.diag(np.cos(x[:Nl]))
    Mt = J * np.eye(Nl) + m * (l**2) * St @ V @ St + m * (l**2) * Ct @ V @ Ct
    Wt = (m * (l**2) * St @ V @ Ct - m * (l**2) * Ct @ V @ St) * x[Nl+1:2*Nl+1]**2
    Xd = l * K.T @ (St @ x[Nl+1:2*Nl+1]) + e @ x[2*Nl+2]
    Yd = -l * K.T @ (Ct @ x[Nl+1:2*Nl+1]) + e @ x[2*Nl+3]
    Fx = (ct * (Ct**2) + cn * (St**2)) * Xd + (ct - cn) * (St @ Ct) * Yd
    Fy = (ct - cn) * St @ Ct * Xd + (ct * (St**2) + cn * (Ct**2)) * Yd
    Tr = -lambda2 * x[Nl+1:2*Nl+1]

    dx = np.zeros_like(x)
    dx[:Nl] = x[Nl+1:2*Nl+1]
    dx[Nl+1:2*Nl+1] = np.linalg.solve(Mt, -Wt + Tr + l * St @ K @ Fx - l * Ct @ K @ Fy + D.T @ u)
    dx[2*Nl+2] = Xd
    dx[2*Nl+3] = Yd
    dx[2*Nl+4] = e @ Fx
    dx[2*Nl+5] = e @ Fy

    return dx


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

# Define parameters
xt = opti.parameter(n, 1)
xf = opti.parameter(n, 1)
u_m = opti.parameter(m, 1)

# Dynamic constraint
for k in range(N):
    k1 = dynamics(x[:, k], u[:, k], SnakeModel)
    k2 = dynamics(x[:, k] + delta / 2 * k1, u[:, k], SnakeModel)
    k3 = dynamics(x[:, k] + delta / 2 * k2, u[:, k], SnakeModel)
    k4 = dynamics(x[:, k] + delta * k3, u[:, k], SnakeModel)
    x_next = x[:, k] + delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(x_next == x[:, k+1])

# Initial constraint
opti.set_value(xt, x0)
opti.subject_to(x[:, 0] == xt)

# Terminal constraint
#opti.set_value(xf, ca.DM.zeros(n, 1))
#pti.subject_to(x[:, N] == xf)

# Input constraint
opti.set_value(u_m, u_max_norm)
opti.subject_to(ca.vec(u) <= u_max_norm);
opti.subject_to(-ca.vec(u) <= u_max_norm);

#generate reference
phi_ref = ca.DM.zeros(SnakeModel.Nl-1, N+1)
alp = 20*np.pi/180
lam = 20*np.pi/180
delt = 60*np.pi/180
for i in range(N+1):
    for j in range(SnakeModel.Nl-1):
        phi_ref[j,i] = alp*ca.sin(lam*delta*i + delt*(j))

# Cost function
C = 0
for k in range(N+1):
    C += (SnakeModel.H_inv2@x[:, k] - phi_ref[:, k]).T @ Q @ (SnakeModel.H_inv2@x[:, k] - phi_ref[:,k])

for k in range(N):
    C +=  (u[:,k].T @ R @ u[:,k])
opti.minimize(C)

# Initial guess
opti.set_initial(x, ca.repmat(x0, 1, N+1))
opti.set_initial(u, ca.DM.zeros(m, N))

# Solve the optimization problem
sol = opti.solve()
x_OL = sol.value(x)
u_OL = sol.value(u)

# Plots
plt.figure(1)
plt.subplot(3, 1, 1)
plt.grid(True)
plt.plot(ca.vertcat(0, ca.linspace(delta, T, N)), x_OL[0, :]-x_OL[1, :])
plt.xlabel('t')
plt.ylabel('x_1(t)')
plt.xlim([0, T])
plt.subplot(3, 1, 2)
plt.plot(ca.vertcat(0, ca.linspace(delta, T, N)), x_OL[1, :] - x_OL[2,:])
plt.xlabel('t')
plt.ylabel('x_2(t)')
plt.xlim([0, T])
plt.subplot(3, 1, 3)
# Plotting the input sequence
edges = np.linspace(0, T-delta, len(u_OL[0,:])+1)
values = u_OL[0,:]
print(u_OL[0,:])
print(edges)
plt.stairs(values,edges)
plt.xlabel('t')
plt.ylabel('u(t)')
plt.xlim([0, T])
plt.show()

# Problem 2
x_MPC = ca.DM.zeros(n, mpciterations + 1)
x_MPC[:, 0] = x0
u_MPC = ca.DM.zeros(m, mpciterations)
t = 0.0

for ii in range(mpciterations):
    sol = opti.solve()
    x_OL = sol.value(x)
    u_OL = sol.value(u)

    u_MPC[:, ii] = u_OL[:, 0]
    x_MPC[:, ii+1] = x_OL[:, 1]
    t += delta

    # Update initial constraint
    opti.set_value(xt, x_MPC[:, ii+1])

    # Update initial guess
    opti.set_initial(x, ca.horzcat(x_OL[:, 1:], x_OL[:, -1]))
    #opti.set_initial(u, ca.vertcat(u_OL[1:], 0))
    opti.set_initial(u, ca.horzcat(u_OL[:, 1:], u_OL[:, -1]))

    # Plot state space
    plt.figure(2)
    plt.plot(x_MPC[0, 0:ii+2], x_MPC[1, 0:ii+2], 'b')
    plt.grid(True)
    plt.plot(x_OL[0, :], x_OL[1, :], 'g')
    plt.plot(x_MPC[0, 0:ii+2], x_MPC[1, 0:ii+2], 'ob')
    plt.xlabel('x(1)')
    plt.ylabel('x(2)')
    plt.title('State Space')
    plt.pause(0.01)
    if ii == mpciterations-1 :
        plt.show()
    
    # Plot states
    plt.figure(4)
    edges = np.linspace(0, T, len(x_OL[0,:]))
    values = x_OL[0,:] - x_OL[1,:]
    plt.plot(edges, values)
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.xlim([0, T])
    plt.pause(0.01)
    if ii == mpciterations-1 :
        plt.show()
    
    # Plot input sequences
    plt.figure(3)
    edges = np.linspace(0, T-delta, len(u_OL[0,:])+1)
    values = u_OL[0,:]
    plt.stairs(values,edges)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.xlim([0, T])
    plt.pause(0.01)
    if ii == mpciterations-1 :
        plt.show()
    


