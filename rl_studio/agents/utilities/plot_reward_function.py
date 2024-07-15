import math

import numpy as np
import matplotlib.pyplot as plt


def normalize_range(num, a, b):
    return (num - a) / (b - a)


def linear_function(cross_x, slope, x):
    return cross_x + (slope * x)


def sigmoid_function(start, end, x, slope=10):
    slope = slope / (end - start)
    sigmoid = 1 / (1 + np.exp(-slope * (x - ((start + end) / 2))))
    return sigmoid


def reward_proximity(state):
    if abs(state) > 0.7:
        return 0
    else:
        # return 1 - abs(state)
        # return linear_function(1, -1.4, abs(state))
        return pow(1 - abs(state), 4)
        # return 1 - sigmoid_function(0, 1, abs(state), 5)


def rewards_followline_velocity_center(v, pos, range_v):
    p_reward = reward_proximity(pos)
    v_norm = normalize_range(v, range_v[0], range_v[1])
    v_r = v_norm * pow(p_reward, 2)
    beta_pos = 0.7
    reward = (beta_pos * p_reward) + ((1 - beta_pos) * v_r)
    v_punish = reward * (1 - p_reward) * v_norm
    reward = reward - v_punish

    return reward


def rewards_easy(v, pos):
    if v < 1:
        return 0

    if abs(pos) > 0.3:
        return -1

    d_reward = math.pow(max(0.5 - abs(pos), 0)/0.5, 3)

    # reward Max = 1 here

    v_reward = v / 20
    v_eff_reward = v_reward * math.pow(d_reward, 5)

    beta = 0.7
    # TODO Ver que valores toma la velocity para compensarlo mejor
    function_reward = beta * d_reward + (1 - beta) * v_eff_reward

    return function_reward


range_v = [0, 20]

# Define the ranges for v, w, and pos
v_range = np.linspace(range_v[0], range_v[1], 50)  # Adjust the range as needed
pos_range = np.linspace(-1, 1, 50)  # Adjust the range as needed

# Create a grid of values for v, w, and pos
V, POS = np.meshgrid(v_range, pos_range)

# Calculate the rewards for each combination of v, w, and pos
rewards = np.empty_like(V)
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        # rewards[i, j] = rewards_followline_velocity_center(V[i, j], POS[i, j], range_v)
        rewards[i, j] = rewards_easy(V[i, j], POS[i, j])

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
surf = ax.plot_surface(POS, V, rewards, cmap='viridis')

# Set labels for the axes
ax.set_xlabel('Center Distance (pos)')
ax.set_ylabel('Linear Velocity (v)')
ax.set_zlabel('Reward')

# Show the color bar
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()