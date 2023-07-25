from datetime import datetime, timedelta
import logging
import os
import pprint
import random
import time
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import gymnasium as gym

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesDQNGazebo,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    print_messages,
    print_dictionary,
    render_params,
    save_metadata,
    save_dataframe_episodes,
    save_batch,
    save_best_episode_dqn,
    LoggingHandler,
)
from rl_studio.algorithms.dqn_keras import (
    ModifiedTensorBoard,
    DQN,
)
from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO


class TrainerFollowLineDQNF1GazeboTF:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DQN
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesDQNGazebo(config)
        self.global_params = LoadGlobalParams(config)
        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)
        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"

    def main(self):
        # basename = f"{self.global_params.models_dir}/{self.global_params.states}/{self.global_params.actions}/{self.algoritmhs_params.model_name}_{time.strftime('%Y%m%d-%H%M%S')}metadata.md"
        # save_metadata(self.algoritmhs_params, self.environment, basename)
        log = LoggingHandler(self.log_file)

        ## Load Environment
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.algoritmhs_params.epsilon
        epsilon_discount = self.algoritmhs_params.epsilon_discount
        epsilon_min = self.algoritmhs_params.epsilon_min
        zero_epsilon_reward = 0

        ## Reset env
        state, state_size = env.reset()

        log.logger.info(
            f"\nstates = {self.global_params.states}\n"
            f"states_set = {self.global_params.states_set}\n"
            f"states_len = {len(self.global_params.states_set)}\n"
            f"actions = {self.global_params.actions}\n"
            f"actions set = {self.global_params.actions_set}\n"
            f"actions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"epsilon = {epsilon}\n"
            f"batch_size = {self.algoritmhs_params.batch_size}\n"
            f"logs_tensorboard_dir = {self.global_params.logs_tensorboard_dir}\n"
        )

        self.averaged_rewards = {}
        self.times_action = {}
        self.total_times_action = {}

        for state in range(-20, 21):
            self.averaged_rewards[state] = {}
            self.times_action[state] = {}

            for actions_list in self.global_params.actions_set.values():
                action_key = '-'.join(str(item) for item in actions_list)
                self.total_times_action[action_key] = 0
            for action in self.global_params.actions_set:
                self.averaged_rewards[state][action] = 0
                self.times_action[state][action] = 0
        ## --------------------- Deep Nets ------------------
        # Init Agent
        dqn_agent = DQN(
            self.environment.environment,
            self.algoritmhs_params,
            len(self.global_params.actions_set),
            state_size,
            self.global_params.models_dir,
            self.global_params,
        )
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        ## -------------    START TRAINING --------------------
        for episode in tqdm(
            range(1, self.env_params.total_episodes + 1),
            ascii=True,
            unit="episodes",
        ):
            tensorboard.step = episode
            done = False
            cumulated_reward = 0
            step = 1
            start_time_epoch = datetime.now()

            observation, _ = env.reset()

            while not done:
                if np.random.random() > epsilon:
                    action = np.argmax(dqn_agent.get_qs(observation))
                else:
                    # Get random action
                    action = np.random.randint(0, len(self.global_params.actions_set))

                new_observation, reward, done, info = env.step(action, step)
                fps = info["fps"]

                actions_list = self.global_params.actions_set.get(action)
                action_key = '-'.join(str(item) for item in actions_list)
                # self.times_action[observation[0]][action] += 1
                self.total_times_action[action_key] += 1
                # self.averaged_rewards[observation[0]][action] = \
                #     (self.averaged_rewards[observation[0]][action] * (self.times_action[observation[0]][action] - 1) + reward) / \
                #     self.times_action[observation[0]][action]

                # # Create flattened list of data values
                # flattened_data = [self.averaged_rewards[state][action] for state in self.averaged_rewards for action in self.averaged_rewards[state]]
                #
                # # Reshape flattened data to 3D tensor
                # data_tensor = tf.reshape(flattened_data, (len(self.averaged_rewards), len(self.averaged_rewards[0]), 1))
                #
                # tensorboard.update_histogram("state-actions-reward", data_tensor)

                # Every step we update replay memory and train main network
                # agent_dqn.update_replay_memory((state, action, reward, nextState, done))
                dqn_agent.update_replay_memory(
                    (observation, action, reward, new_observation, done)
                )

                # env._gazebo_pause()
                dqn_agent.train(done, step)
                # env._gazebo_unpause()

                cumulated_reward += reward
                observation = new_observation
                step += 1

                log.logger.debug(
                    f"\nobservation = {observation}\n"
                    # f"observation[0]= {observation[0]}\n"
                    f"observation type = {type(observation)}\n"
                    # f"observation[0] type = {type(observation[0])}\n"
                    f"new_observation = {new_observation}\n"
                    f"new_observation = {type(new_observation)}\n"
                    f"action = {action}\n"
                    f"actions type = {type(action)}\n"
                )

                # env._gazebo_pause()

                log.logger.debug(
                    f"\nepisode = {episode}\n"
                    f"step = {step}\n"
                    f"actions_len = {len(self.global_params.actions_set)}\n"
                    f"actions_range = {range(len(self.global_params.actions_set))}\n"
                    f"actions = {self.global_params.actions_set}\n"
                    f"epsilon = {epsilon}\n"
                    # f"v = {self.global_params.actions_set[action][0]}\n"
                    # f"w = {self.global_params.actions_set[action][1]}\n"
                    f"observation = {observation}\n"
                    f"reward_in_step = {reward}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"done = {done}\n"
                )
                # env._gazebo_unpause()

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step and self.global_params.show_monitoring:
                    render_params(
                        task=self.global_params.task,
                        # v=action[0][0], # for continuous actions
                        # w=action[0][1], # for continuous actions
                        episode=episode,
                        step=step,
                        state=observation,
                        action=action,
                        # v=self.global_params.actions_set[action][
                        #    0
                        # ],  # this case for discrete
                        # w=self.global_params.actions_set[action][
                        #    1
                        # ],  # this case for discrete
                        # self.env.image_center,
                        # self.actions_rewards,
                        reward_in_step=reward,
                        cumulated_reward_in_this_episode=cumulated_reward,
                        _="--------------------------",
                        # best_episode_until_now=best_epoch,
                        fps=fps,
                        # in_best_step=best_step,
                        with_highest_reward=int(current_max_reward),
                        # in_best_epoch_training_time=best_epoch_training_time,
                        epsilon=epsilon
                    )
                    log.logger.debug(
                        f"SHOWING BATCH OF STEPS\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"current epoch = {episode}\n"
                        f"current step = {step}\n"
                        f"best epoch so far = {best_epoch}\n"
                        f"best step so far = {best_step}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )

                #####################################################
                ### save in case of completed steps in one episode
                if step >= self.env_params.estimated_steps:
                    done = True
                    log.logger.info(
                        f"\nEPISODE COMPLETED\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epsilon = {epsilon}\n"
                        f"fps = {fps}\n"
                    )
                    basename = f"{self.global_params.models_dir}/{self.global_params.states}/{self.global_params.actions}/{self.algoritmhs_params.model_name}_LAPCOMPLETED_Max{int(cumulated_reward)}_inTime{time.strftime('%Y%m%d-%H%M%S')}_Epoch{episode}"
                    dqn_agent.model.save(
                        f"{basename}.model"
                    )

            #####################################################
            #### save best lap in episode
            if current_max_reward <= cumulated_reward and episode > 1:
                (
                    current_max_reward,
                    best_epoch,
                    best_step,
                    best_epoch_training_time,
                ) = save_best_episode_dqn(
                    self.global_params,
                    cumulated_reward,
                    episode,
                    step,
                    start_time_epoch,
                    reward,
                )

                self.global_params.best_current_epoch["best_epoch"].append(best_epoch)
                self.global_params.best_current_epoch["highest_reward"].append(
                    cumulated_reward
                )
                self.global_params.best_current_epoch["best_step"].append(best_step)
                self.global_params.best_current_epoch[
                    "best_epoch_training_time"
                ].append(best_epoch_training_time)
                self.global_params.best_current_epoch[
                    "current_total_training_time"
                ].append(datetime.now() - start_time)
                # save_dataframe_episodes(
                #     self.environment.environment,
                #     self.global_params.metrics_data_dir,
                #     self.global_params.aggr_ep_rewards,
                # )
                dqn_agent.model.save(
                    f"{self.global_params.models_dir}/{self.global_params.states}/{self.global_params.actions}/{self.algoritmhs_params.model_name}_IMPROVED_Max{int(cumulated_reward)}_inTime{time.strftime('%Y%m%d-%H%M%S')}_Epoch{episode}.model"
                )

                log.logger.info(
                    f"\nsaving best lap\n"
                    f"in episode = {episode}\n"
                    f"current_max_reward = {current_max_reward}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"steps = {step}\n"
                    f"epsilon = {epsilon}\n"
                )

                # best episode


            #####################################################
            ### end episode in time settings: 2 hours, 15 hours...
            if (
                datetime.now() - timedelta(hours=self.global_params.training_time)
                > start_time
            ):
                log.logger.info(
                    f"\nTraining Time over\n"
                    f"current_max_reward = {current_max_reward}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"epoch = {episode}\n"
                    f"step = {step}\n"
                    f"epsilon = {epsilon}\n"
                )
                if cumulated_reward > current_max_reward:
                    dqn_agent.model.save(
                        f"{self.global_params.models_dir}/{self.algoritmhs_params.model_name}_END_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )

                break

            if epsilon == 0:
                epsilon = old_epsilon
                zero_epsilon_reward = cumulated_reward


            # reducing exploration
            if epsilon > epsilon_min:
                epsilon *= epsilon_discount
            #####################################################
            ### save every save_episode times
            self.global_params.ep_rewards.append(cumulated_reward)
            average_reward = sum(
                self.global_params.ep_rewards[-self.env_params.save_episodes:]
            ) / len(self.global_params.ep_rewards[-self.env_params.save_episodes:])
            min_reward = min(
                self.global_params.ep_rewards[-self.env_params.save_episodes:]
            )
            max_reward = max(
                self.global_params.ep_rewards[-self.env_params.save_episodes:]
            )

            tensorboard.update_stats(
                reward_avg=int(average_reward),
                reward_max=int(max_reward),
                zero_rewards=zero_epsilon_reward,
                epsilon=epsilon,
            )

            self.global_params.aggr_ep_rewards["episode"].append(episode)
            self.global_params.aggr_ep_rewards["step"].append(step)
            self.global_params.aggr_ep_rewards["avg"].append(average_reward)
            self.global_params.aggr_ep_rewards["max"].append(max_reward)
            self.global_params.aggr_ep_rewards["min"].append(min_reward)
            self.global_params.aggr_ep_rewards["epoch_training_time"].append(
                (datetime.now() - start_time_epoch).total_seconds()
            )
            self.global_params.aggr_ep_rewards["total_training_time"].append(
                (datetime.now() - start_time).total_seconds()
            )

            if not episode % self.env_params.save_episodes:
                basename = f"{self.global_params.models_dir}/{self.global_params.states}/{self.global_params.actions}/{self.algoritmhs_params.model_name}_PERIODIC_Max{int(cumulated_reward)}_inTime{time.strftime('%Y%m%d-%H%M%S')}_Epoch{episode}"
                # dqn_agent.model.save(
                #    f"{basename}.model"
                # )
                save_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )
                if self.global_params.debug_stats:
                    with open(f"{basename}-rewards-states.pickle", 'wb') as f:
                        pickle.dump(self.averaged_rewards, f)
                    with open(f"{basename}-actions-state.pickle", 'wb') as f:
                        pickle.dump(self.times_action, f)
                    with open(f"{basename}-actions.pickle", 'wb') as f:
                        pickle.dump(self.total_times_action, f)
                    plot_histograms(self.averaged_rewards, self.times_action, self.total_times_action)

                log.logger.info(
                    f"\nsaving BATCH\n"
                    f"current_max_reward = {current_max_reward}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"best_epoch = {best_epoch}\n"
                    f"best_step = {best_step}\n"
                    f"best_epoch_training_time = {best_epoch_training_time}\n"
                    f"epsilon = {epsilon}"
                )
                # ZERO EPSILON FOR EVALUATING INTERMEDIATE AGENTS
                # old_epsilon=epsilon
                # epsilon=0


        #####################################################
        ### save last episode, not neccesarily the best one
        save_dataframe_episodes(
            self.environment.environment,
            self.global_params.metrics_data_dir,
            self.global_params.aggr_ep_rewards,
        )
        env.close()

def plot_histograms(averaged_rewards, actions, total_actions):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': '3d'})
    plot_3d(axs[0], averaged_rewards)
    axs[0].set_title("Histogram for averaged_rewards")
    axs[0].set_xlabel("states")
    axs[0].set_ylabel("actions")
    axs[0].set_zlabel("average reward")
    plot_3d(axs[1], actions)
    axs[1].set_title("Histogram for actions")
    axs[1].set_xlabel("states")
    axs[1].set_ylabel("actions")
    axs[1].set_zlabel("Frequency")
    fig2, axs2 = plt.subplots(1, 1, figsize=(10, 5))
    axs2.bar(total_actions.keys(), total_actions.values())
    axs2.set_title("Histogram for total_actions")
    axs2.set_xlabel("Value")
    axs2.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_3d(ax, arr2):
    # Extract data from dictionary
    x, y, freq = [], [], []
    for k1, v1 in arr2.items():
        for k2, v2 in v1.items():
                x.append(k1)
                y.append(k2)
                freq.append(v2)

    # Compute histogram
    hist, xedges, yedges = np.histogram2d(x, y, weights=freq, bins=(len(set(x)), len(set(y))))
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct the bars with the 3D coordinates and frequencies
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

