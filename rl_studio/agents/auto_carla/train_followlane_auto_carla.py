from datetime import datetime, timedelta
import glob
import time

import gymnasium as gym
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from rl_studio.agents.utilities.plot_npy_dataset import plot_rewards
from rl_studio.agents.utilities.push_git_repo import git_add_commit_push
from rl_studio.envs.carla.utils.navigation.basic_agent import BasicAgent
from rl_studio.envs.carla.utils.navigation.behavior_agent import BehaviorAgent
from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesDDPGCarla,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    print_messages,
    render_params,
    save_dataframe_episodes,
    LoggingHandler,
)
from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)

from rl_studio.algorithms.ppo_continuous import PPO

from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.carla_env import CarlaEnv

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

def combine_attributes(obj1, obj2, obj3):
    combined_dict = {}

    # Extract attributes from obj1
    obj1_dict = obj1.__dict__
    for key, value in obj1_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj2
    obj2_dict = obj2.__dict__
    for key, value in obj2_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj3
    obj3_dict = obj3.__dict__
    for key, value in obj3_dict.items():
        combined_dict[key] = value

    return combined_dict


# Function to update scatter plot with new data
def update_scatter_plot(ax, x, y, z, xlabel, ylabel, zlabel):
    ax.clear()
    ax.scatter(x, y, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.draw()
    plt.pause(0.001)

class TrainerFollowLaneAutoCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DDPG
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):

        self.actor_loss = 0
        self.critic_loss = 0
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.global_params = LoadGlobalParams(config)
        self.environment = LoadEnvVariablesDDPGCarla(config)
        self.loss = 0

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/auto-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        self.log = LoggingHandler(log_file)
        # self.log.logger.basicConfig(filename=log_file, level=logging.INFO)

        ## Load Carla server
        # CarlaEnv.__init__(self)

        self.environment.environment["debug_waypoints"] = True
        self.env = gym.make(self.env_params.env_name, **self.environment.environment)
        self.all_steps = 0
        self.current_max_reward = 0
        self.episodes_speed = []
        self.episodes_d_reward = []
        self.episodes_steer = []
        self.episodes_reward = []
        self.step_fps = []
        self.bad_perceptions = 0

        self.all_steps_reward = []
        self.all_steps_velocity = []
        self.all_steps_steer = []
        self.all_steps_state0 = []
        self.all_steps_state1 = []
        self.all_steps_state2 = []
        self.all_steps_state3 = []
        self.all_steps_state4 = []

        self.cpu_usages = 0
        self.gpu_usages = 0

        # Initialize the scatter plots
        fig = plt.figure(figsize=(12, 10))
        self.ax1 = fig.add_subplot(221, projection='3d')
        self.ax2 = fig.add_subplot(222, projection='3d')
        self.ax3 = fig.add_subplot(223, projection='3d')
        self.ax4 = fig.add_subplot(224, projection='3d')
        self.ax5 = fig.add_subplot(225, projection='3d')
        self.ax6 = fig.add_subplot(226, projection='3d')

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)


    def calculate_and_report_episode_stats(self, episode_time, step, cumulated_reward):
        avg_speed = np.mean(self.episodes_speed)
        max_speed = np.max(self.episodes_speed)
        cum_d_reward = np.sum(self.episodes_d_reward)
        max_reward = np.max(self.episodes_reward)
        steering_std_dev = np.std(self.episodes_steer)
        advanced_meters = avg_speed * episode_time
        bad_perceptions_perc = self.bad_perceptions / step
        self.tensorboard.update_stats(
            std_dev=self.exploration,
            steps_episode=step,
            cum_rewards=cumulated_reward,
            avg_speed=avg_speed,
            max_speed=max_speed,
            cum_d_reward=cum_d_reward,
            max_reward=max_reward,
            steering_std_dev=steering_std_dev,
            advanced_meters=advanced_meters,
            actor_loss=self.actor_loss,
            critic_loss=self.critic_loss,
            bad_perceptions_perc=bad_perceptions_perc,
            cpu=self.cpu_usages,
            gpu=self.gpu_usages
        )
        self.tensorboard.update_fps(self.step_fps)
        self.episodes_speed = []
        self.episodes_d_reward = []
        self.episodes_steer = []
        self.episodes_reward = []
        self.step_fps = []
        self.bad_perceptions = 0


    def set_stats(self, info, prev_state):
        self.episodes_speed.append(info["velocity"])
        self.episodes_steer.append(info["steering_angle"])
        self.episodes_d_reward.append(info["d_reward"])
        self.episodes_reward.append(info["reward"])

        self.all_steps_reward.append(info["reward"])
        self.all_steps_velocity.append(info["velocity"])
        self.all_steps_steer.append(info["steering_angle"])
        self.all_steps_state0.append(prev_state[0])
        self.all_steps_state1.append(prev_state[0])
        self.all_steps_state2.append(prev_state[0])
        self.all_steps_state3.append(prev_state[0])
        self.all_steps_state4.append(prev_state[4])

        pass
    def log_and_plot_rewards(self, episode, step, cumulated_reward):
        # Showing stats in screen for monitoring. Showing every 'save_every_step' value
        if not self.all_steps % self.env_params.save_every_step:
            file_name = save_dataframe_episodes(
                self.environment.environment,
                self.global_params.metrics_data_dir,
                self.global_params.aggr_ep_rewards,
            )
            plot_rewards(
                self.global_params.metrics_data_dir,
                file_name
            )
            git_add_commit_push("automatic_rewards_update")
            self.log.logger.debug(
                f"SHOWING BATCH OF STEPS\n"
                f"current_max_reward = {self.current_max_reward}\n"
                f"current epoch = {episode}\n"
                f"current step = {step}\n"
            )

    def one_step_iteration(self, episode, step, prev_state, cumulated_reward):
        self.all_steps += 1

        control = self.agent.run_step()
        action = [control.throttle, control.steer]

        state, reward, done, info = self.env.step(action)
        # reversed_v = -action[0]
        # reversed_w = -action[1]
        # state, reward, done, info = self.env.step([reversed_v, reversed_w])
        self.set_stats(info, prev_state)

        if self.global_params.show_monitoring:
            render_params(
                task=self.global_params.task,
                v=action[0],  # for continuous actions
                w=action[1],  # for continuous actions
                episode=episode,
                step=step,
                state=state,
                reward_in_step=reward,
                cumulated_reward_in_this_episode=cumulated_reward,
                _="--------------------------",
                # fps=fps,
                # best_episode_until_now=best_epoch,
                # with_highest_reward=int(current_max_reward),
            )
            # Update scatter plot
            update_scatter_plot(self.ax1, self.all_steps_velocity, self.all_steps_state0, self.all_steps_reward,
                                "Velocity", "State[0]", "Reward")
            update_scatter_plot(self.ax2, self.all_steps_velocity, self.all_steps_state1, self.all_steps_reward,
                                "Velocity", "State[9]", "Reward")
            update_scatter_plot(self.ax3, self.all_steps_velocity, self.all_steps_state2, self.all_steps_reward, "Steer",
                                "State[0]", "Reward")
            update_scatter_plot(self.ax4, self.all_steps_velocity, self.all_steps_state3, self.all_steps_reward, "Steer",
                                "State[9]", "Reward")
            update_scatter_plot(self.ax5, self.all_steps_velocity, self.all_steps_state4, self.all_steps_reward, "Steer",
                                "State[9]", "Reward")
            update_scatter_plot(self.ax6, self.all_steps_steer, self.all_steps_state0, self.all_steps_reward, "Steer",
                                "State[9]", "Reward")

        return state, cumulated_reward, done, 0

    def main(self):
        self.env.traffic_manager.set_synchronous_mode(True)
        hyperparams = combine_attributes(self.algoritmhs_params,
                                         self.environment,
                                         self.global_params)
        self.tensorboard.update_hyperparams(hyperparams)
        # best_epoch_training_time = 0
        # best_epoch = 1

        if self.global_params.mode == "retraining":
            checkpoint = self.environment.environment["retrain_ppo_tf_model_name"]
            trained_agent=f"{self.global_params.models_dir}/{checkpoint}"
            self.ppo_agent.load(trained_agent)

        self.log.logger.info(
            f"\nstates = {self.global_params.states}\n"
            f"states_set = {self.global_params.states_set}\n"
            f"states_len = {len(self.global_params.states_set)}\n"
            f"actions = {self.global_params.actions}\n"
            f"actions set = {self.global_params.actions_set}\n"
            f"actions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"logs_tensorboard_dir = {self.global_params.logs_tensorboard_dir}\n"
        )

        prev_state, _ = self.env.reset()
        self.agent = BasicAgent(self.env.car)
        # self.agent = BehaviorAgent(self.env.car, behavior="normal")

        ## -------------    START TRAINING --------------------
        for episode in tqdm(
                range(1, self.env_params.total_episodes + 1), ascii=True, unit="episodes"
        ):
            self.tensorboard.step = episode
            done = False
            cumulated_reward = 0
            step = 1
            start_time_epoch = datetime.now()

            if episode != 1:
                prev_state, _ = self.env.reset()

            while not done:
                state, cumulated_reward, done, loss = self.one_step_iteration(episode, step, prev_state, cumulated_reward)
                prev_state = state
                step += 1

                # self.log_and_plot_rewards(episode, step, cumulated_reward)
                self.env.display_manager.render()
            episode_time = step * self.environment.environment["fixed_delta_seconds"]
            self.calculate_and_report_episode_stats(episode_time, step, cumulated_reward)
            self.env.destroy_all_actors()
            self.env.display_manager.destroy()

        # self.env.close()
