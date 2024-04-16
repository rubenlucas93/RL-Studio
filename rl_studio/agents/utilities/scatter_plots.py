import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generating sample data for velocity and state[0]
num_samples = 1000
velocity = np.random.normal(loc=50, scale=10, size=num_samples)  # Mean velocity of 50 and standard deviation of 10
state_0 = np.random.uniform(low=0, high=1, size=num_samples)    # Uniform distribution between 0 and 1

# Generating corresponding rewards and steer angles (just for completeness)
reward = np.random.normal(loc=0, scale=1, size=num_samples)      # Assuming rewards follow a normal distribution
steer_angle = np.random.uniform(low=-30, high=30, size=num_samples)  # Random steering angles within a range

# Plotting the 3D scatter plots
fig = plt.figure(figsize=(12, 10))

# Velocity - State[0] - Reward
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(velocity, state_0, reward)
ax1.set_xlabel('Velocity')
ax1.set_ylabel('State[0]')
ax1.set_zlabel('Reward')

# Steer Angle - State[0] - Reward
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(steer_angle, state_0, reward)
ax2.set_xlabel('Steer Angle')
ax2.set_ylabel('State[0]')
ax2.set_zlabel('Reward')

plt.tight_layout()
plt.show()
