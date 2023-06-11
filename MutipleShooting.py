import casadi as ca
import numpy as np

# Create an optimization problem
opti = ca.Opti()
opti.solver('ipopt')

# System parameters
mu = 0.5

# System dimensions
n = 2  # State dimension
m = 1  # Input dimension

# Parameters
delta = 0.1  # Sampling time
T = 5  # Continuous-time prediction horizon
N = int(T / delta)  # Discrete-time prediction horizon
tmeasure = 0.0  # Initial time
x0 = ca.DM([0.4, -0.5])  # Initial state
SimTime = 5  # Simulation time
mpciterations = int(SimTime / delta)  # Discrete simulation time

# Define optimization variables
x = opti.variable(n, N+1)  # Predicted state sequence
u = opti.variable(m, N)  # Predicted input sequence

# Cost function weights
Q = ca.diag([0.5, 0.5])  # State weight matrix
R = 1.0  # Input weight

# Input constraint
u_max_norm = 1.0  # Maximum input magnitude

# Define dynamics function
def dynamics(x, u):
    x_dot = ca.vertcat(x[1] + u * (mu + (1 - mu) * x[0]),
                       x[0] + u * (mu - 4 * (1 - mu) * x[1]))
    return x_dot

# Define parameters
xt = opti.parameter(n, 1)
xf = opti.parameter(n, 1)
u_m = opti.parameter(m, 1)

# Dynamic constraint
for k in range(N):
    k1 = dynamics(x[:, k], u[:, k])
    k2 = dynamics(x[:, k] + delta / 2 * k1, u[:, k])
    k3 = dynamics(x[:, k] + delta / 2 * k2, u[:, k])
    k4 = dynamics(x[:, k] + delta * k3, u[:, k])
    x_next = x[:, k] + delta / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(x_next == x[:, k+1])

# Initial constraint
opti.set_value(xt, x0)
opti.subject_to(x[:, 0] == xt)

# Terminal constraint
opti.set_value(xf, ca.DM.zeros(n, 1))
opti.subject_to(x[:, N] == xf)

# Input constraint
opti.set_value(u_m, u_max_norm)
opti.subject_to(u <= u_m)
opti.subject_to(-u <= u_m)

# Cost function
C = 0
for k in range(N+1):
    C += x[:, k].T @ Q @ x[:, k]
opti.minimize(C + R * (u @ u.T))

# Initial guess
opti.set_initial(x, ca.repmat(x0, 1, N+1))
opti.set_initial(u, ca.DM.zeros(m, N))

# Solve the optimization problem
sol = opti.solve()
x_OL = sol.value(x)
u_OL = sol.value(u)

# Plots
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(3, 1, 1)
plt.grid(True)
plt.plot(ca.vertcat(0, ca.linspace(delta, T, N)), x_OL[0, :])
plt.xlabel('t')
plt.ylabel('x_1(t)')
plt.xlim([0, T])
plt.subplot(3, 1, 2)
plt.plot(ca.vertcat(0, ca.linspace(delta, T, N)), x_OL[1, :])
plt.xlabel('t')
plt.ylabel('x_2(t)')
plt.xlim([0, T])
plt.subplot(3, 1, 3)
# Plotting the input sequence
edges = np.linspace(0, T-delta, len(u_OL)+1)
values = u_OL
print(u_OL)
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

    u_MPC[ii] = u_OL[0]
    x_MPC[:, ii+1] = x_OL[:, 2]
    t += delta

    # Update initial constraint
    opti.set_value(xt, x_MPC[:, ii+1])

    # Update initial guess
    opti.set_initial(x, ca.horzcat(x_OL[:, 1:], x_OL[:, -1]))
    opti.set_initial(u, ca.vertcat(u_OL[1:], 0))
    #opti.set_initial(u, ca.horzcat(u_OL[:, 1:], ca.DM.zeros(m, 1)))

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
    

    
    # Plot input sequences
    plt.figure(3)
    edges = np.linspace(0, T-delta, len(u_OL)+1)
    values = u_OL
    plt.stairs(values,edges)
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.xlim([0, T])
    plt.pause(0.01)
    if ii == mpciterations-1 :
        plt.show()
    


