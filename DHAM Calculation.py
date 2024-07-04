import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Define the instantaneous market impact function
def impact_function(v, delta):
    return np.sign(v) * np.abs(v) ** delta

# Define the decay kernel
def decay_kernel(tau, gamma):
    return tau ** (-gamma) if tau > 0 else 1.0 

# Compute the Toeplitz matrix Gij
def compute_G(N, T, gamma):
    dt = T / N
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i >= j:
                G[i, j] = decay_kernel((i - j) * dt, gamma) * dt
    return G

# Compute the A matrix based on G
def compute_A(G, N):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1):
            if i == j:
                A[i, j] = G[i, j] / 2
            else:
                A[i, j] = G[i, j]
    return A

# Calculate the expected cost
def expected_cost(v, A, delta, N):
    cost = 0
    for i in range(N):
        for j in range(N):
            cost += v[i] * impact_function(v[j], delta) * A[i, j]
    return cost

# Constraint function to ensure sum of trading rates equals total order size X
def constraint(v, X):
    return np.sum(v) - X

# Homotopy Analysis Method (HAM) with constraint enforcement
def ham(N, T, delta, gamma, X, initial_guess, order=10, tol=1e-6):
    G = compute_G(N, T, gamma)
    A = compute_A(G, N)
    v0 = initial_guess
    v = np.copy(v0)
    h = -1  # Convergence control parameter, can be optimized
    
    for m in range(1, order + 1):
        residual = np.zeros(N)
        for i in range(N):
            sum_term = 0
            for j in range(N):
                if i >= j:
                    sum_term += G[i, j] * impact_function(v[j], delta)
                else:
                    sum_term += G[i, j] * v[j] * delta * np.abs(v[i]) ** (delta - 1)
            residual[i] = sum_term - 1  # lambda is set to 1 for simplicity
        for i in range(N):
            v[i] += h * residual[i]
        
        # Enforce the constraint at each iteration
        v = v * (X / np.sum(v))
        
        if np.linalg.norm(residual) < tol:
            break
    
    return v, A

# Parameters
X = 0.1  # total order size as a percentage of total market volume
T = 1.0  # total time period
delta = 0.55  # impact function exponent (concave impact)
gamma = 0.45  # decay kernel exponent
num_steps = 100  # number of time steps for simulation

# Initial guess: VWAP strategy
initial_guess = np.full(num_steps, X / num_steps)

# Apply DHAM
optimal_trading_rate, A = ham(num_steps, T, delta, gamma, X, initial_guess)

# Calculate the expected cost
cost = expected_cost(optimal_trading_rate, A, delta, num_steps)
print("Expected Cost:", cost / X)

# Plot the optimal trading rate
time_points = np.linspace(0, T, num_steps)
plt.plot(time_points, optimal_trading_rate, label='Optimal Trading Rate')
plt.xlabel('Time')
plt.ylabel('Volumn in Percentage')
plt.title('Optimal Trading Strategy')

# Set the y-axis to scientific notation
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))
plt.gca().yaxis.set_major_formatter(formatter)

plt.legend()
plt.show()