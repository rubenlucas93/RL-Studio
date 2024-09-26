import yaml
import pymongo
from pymongo import MongoClient
import datetime
import os
import plot_tensorboard_results
import extract_reward_function
import matplotlib.pyplot as plt
import base64
from io import BytesIO

yaml_file = '/home/ruben/Desktop/RL-Studio/rl_studio/config/config_training_followlane_bs_ddpg_f1_carla.yaml'  # Replace with your YAML file path

reward_filename = '/home/ruben/Desktop/RL-Studio/rl_studio/envs/carla/followlane/followlane_carla_sb.py'
reward_method = 'rewards_easy'

tensorboard_logs_dir = '/home/ruben/Desktop/RL-Studio/rl_studio/logs/retraining/follow_lane_carla_ddpg_auto_carla_baselines/TensorBoard/DDPG_Actor_conv2d32x64_Critic_conv2d32x64-20240924-202411'

def load_hyperparameters(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def plot_and_encode(x, y, title):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel('Values')

    # Save plot to a BytesIO object and convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64


def store_training_results(config, training_results):
    # Set up the MongoDB client (this assumes MongoDB is running locally)
    client = MongoClient('mongodb://localhost:27017/')
    db = client['training_db']  # Database name
    collection = db['training_results']  # Collection name

    del config["states"] # TODO removed for now since it is not mongo compatible
    # Prepare the document to store
    document = {
        'config': config,
        'results': training_results,
        'timestamp': datetime.datetime.utcnow()
    }

    # validate_document_keys(document)
    # Insert the document into MongoDB
    result = collection.insert_one(document)
    print(f"Inserted document with ID: {result.inserted_id}")

def validate_document_keys(doc):
    if isinstance(doc, dict):
        for key, value in doc.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key: {key}. All keys must be strings.")
            print("validating " + str(value))
            validate_document_keys(value)
    elif isinstance(doc, list):
        for item in doc:
            validate_document_keys(item)


def run_training_and_store_results(yaml_file):
    # Load the YAML configuration
    config = load_hyperparameters(yaml_file)

    reward_function = extract_reward_function.extract_reward_function(reward_filename, reward_method)

    rewards, advanced_meters, avg_speed, std_dev = plot_tensorboard_results.extract_tensorboard_data(tensorboard_logs_dir)
    reward_steps, reward_values = plot_tensorboard_results.tensor_to_value_list(rewards)
    advanced_meters_steps, advanced_meters_values = plot_tensorboard_results.tensor_to_value_list(advanced_meters)
    avg_speed_steps, avg_speed_values = plot_tensorboard_results.tensor_to_value_list(avg_speed)
    std_dev_steps, std_dev_values = plot_tensorboard_results.tensor_to_value_list(std_dev)

    # Plot figures and encode as base64 strings
    reward_plot = plot_and_encode(reward_steps, reward_values, 'Reward Function Plot')
    advanced_meters_plot = plot_and_encode(advanced_meters_steps, advanced_meters_values, 'Advanced Meters Plot')
    avg_speed_plot = plot_and_encode(avg_speed_steps, avg_speed_values, 'Average Speed Plot')
    std_dev_plot = plot_and_encode(std_dev_steps, std_dev_values, 'Standard Deviation Plot')

    # Simulate training results
    training_results = {
        'reward_function': reward_function,  # Storing reward function as a string
        'plots': {
            'reward_plot': reward_plot,
            'advanced_meters_plot': advanced_meters_plot,
            'avg_speed_plot': avg_speed_plot,
            'std_dev_plot': std_dev_plot
        }
    }

    # Store the results in MongoDB
    store_training_results(config, training_results)


if __name__ == '__main__':
    run_training_and_store_results(yaml_file)
