import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf

def extract_tensorboard_data(log_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Retrieve metrics from TensorBoard logs as Tensors
    rewards = event_acc.Tensors('cum_rewards')  # Key name for cumulative rewards
    advanced_meters = event_acc.Tensors('advanced_meters')  # Key name for advanced meters
    avg_speed = event_acc.Tensors('avg_speed')  # Key name for average speed
    std_dev = event_acc.Tensors('std_dev')  # Key name for standard deviation
    # actions_v = event_acc.Tensors('actions_v')  # TODO (add this tough it is better on inference)
    # actions_w = event_acc.Tensors('actions_w')

    # Return the Tensors for processing
    return rewards, advanced_meters, avg_speed, std_dev

def tensor_to_value_list(tensor_events):
    """
    Extracts the step and value from each tensor event.
    TensorBoard tensors are stored in a tensor_proto format.
    This function converts them into a list of steps and values.
    """
    steps = [event.step for event in tensor_events]
    values = [tf.make_ndarray(event.tensor_proto) for event in tensor_events]  # Assuming single float tensor values
    return steps, values

def plot_metrics(rewards, advanced_meters, avg_speed, std_dev):
    plt.figure(figsize=(12, 8))

    # Convert tensor to value lists
    reward_steps, reward_values = tensor_to_value_list(rewards)
    advanced_meters_steps, advanced_meters_values = tensor_to_value_list(advanced_meters)
    avg_speed_steps, avg_speed_values = tensor_to_value_list(avg_speed)
    std_dev_steps, std_dev_values = tensor_to_value_list(std_dev)

    # Plot Reward evolution
    plt.subplot(2, 2, 1)
    plt.plot(reward_steps, reward_values)
    plt.title('Reward Evolution')
    plt.xlabel('Steps')
    plt.ylabel('Reward')

    # Plot Advanced Meters evolution
    plt.subplot(2, 2, 2)
    plt.plot(advanced_meters_steps, advanced_meters_values)
    plt.title('Advanced Meters Evolution')
    plt.xlabel('Steps')
    plt.ylabel('Meters')

    # Plot Average Speed evolution
    plt.subplot(2, 2, 3)
    plt.plot(avg_speed_steps, avg_speed_values)
    plt.title('Average Speed Evolution')
    plt.xlabel('Steps')
    plt.ylabel('Speed')

    # Plot Standard Deviation evolution
    plt.subplot(2, 2, 4)
    plt.plot(std_dev_steps, std_dev_values)
    plt.title('Standard Deviation Evolution')
    plt.xlabel('Steps')
    plt.ylabel('Std Dev')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    log_dir = '/home/ruben/Desktop/RL-Studio/rl_studio/logs/training/follow_lane_carla_ddpg_auto_carla_baselines/TensorBoard/DDPG_Actor_conv2d32x64_Critic_conv2d32x64-20240923-221222'
    rewards, advanced_meters, avg_speed, std_dev = extract_tensorboard_data(log_dir)

    plot_metrics(rewards, advanced_meters, avg_speed, std_dev)
