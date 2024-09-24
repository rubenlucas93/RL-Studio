import logging
from datetime import datetime, timedelta
import glob
import time
import pynvml
import psutil

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.policies import BasePolicy

from gym import spaces
from typing import Callable, Tuple

import torch as th
import torch.nn as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym
import tensorflow as tf
from tqdm import tqdm
from rl_studio.agents.utilities.plot_npy_dataset import plot_rewards
from rl_studio.agents.utilities.push_git_repo import git_add_commit_push
from rl_studio.algorithms.utils import (
    save_actorcritic_baselines_model,
)
from stable_baselines3.common.callbacks import CallbackList

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback

from wandb.integration.sb3 import WandbCallback
import wandb


from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)


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
    OUActionNoise
)

from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.carla_env import CarlaEnv

from stable_baselines3 import DDPG
# from stable_baselines3 import DDPG
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure

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


# Function to update scatter plot with new data
def update_scatter_plot(ax, x, y, z, xlabel, ylabel, zlabel):
    ax.clear()
    ax.scatter(x, y, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.draw()
    plt.pause(0.001)


def collect_usage():
    cpu_usage = psutil.cpu_percent(interval=None)  # Get CPU usage percentage
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_usage = gpu_info.gpu
    return cpu_usage, gpu_usage


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


class PeriodicSaveCallback(BaseCallback):
    def __init__(self, save_path, save_freq=10000, verbose=1):
        super(PeriodicSaveCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.step_count = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            model_save_path = os.path.join(self.save_path, f"model_{self.step_count}_steps")
            self.model.save(model_save_path)
            if self.verbose > 0:
                print(f"Saved model at step {self.step_count} to {model_save_path}")
        return True

class ExplorationRateCallback(BaseCallback):
    def __init__(self, tensorboard, initial_exploration_rate=0.2, decay_rate=0.01, decay_steps=10000, exploration_min=0.005, verbose=1):
        super(ExplorationRateCallback, self).__init__(verbose)
        self.initial_exploration_rate = initial_exploration_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.current_step = 0
        self.tensorboard = tensorboard
        self.exploration_min = exploration_min
        self.exploration_rate = initial_exploration_rate

    def _on_step(self) -> bool:
        self.current_step += 1
        if self.current_step % self.decay_steps == 1:
            self.exploration_rate = max(self.exploration_min, self.exploration_rate - self.decay_rate)
            # Assuming self.model is a DDPG model
            self.model.action_noise = NormalActionNoise(
                mean=np.zeros(2),
                sigma=self.exploration_rate * np.ones(2)
            )
            if self.verbose > 0:
                print(f"Step {self.current_step}: Setting exploration rate to {self.exploration_rate}")
            self.tensorboard.update_stats(std_dev=self.exploration_rate)
        return True


class CustomActor(nn.Module):
    """
    Custom actor network for policy function.
    :param feature_dim: dimension of the features extracted with the features_extractor
    """

    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 64):
        super(CustomActor, self).__init__()

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi),
            nn.ReLU(),
            nn.Linear(last_layer_dim_pi, 32),
            nn.ReLU(),
        )
        # Separate heads for velocity and steering
        self.velocity_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )
        self.steering_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()  # Output range [-1, 1], will be scaled to [-0.5, 0.5]
        )
        self.optimizer = th.optim.Adam(self.parameters())

    def forward(self, features: th.Tensor) -> th.Tensor:
        features = th.tensor(features, dtype=th.float32).to("cuda")
        policy_latent = self.policy_net(features)
        velocity = self.velocity_head(policy_latent)
        steering = self.steering_head(policy_latent) * 0.5  # Scale to [-0.5, 0.5]
        return th.cat((velocity, steering), dim=-1)


class CustomCritic(nn.Module):
    """
    Custom critic network for value function.
    :param feature_dim: dimension of the features extracted with the features_extractor
    """

    def __init__(self, feature_dim: int, last_layer_dim_vf: int = 64):
        super(CustomCritic, self).__init__()

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf),
            nn.ReLU(),
            nn.Linear(last_layer_dim_vf, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.optimizer = th.optim.Adam(self.parameters())

    def forward(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomDDPGPolicy(BasePolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        kwargs.pop('n_critics', None)
        super(CustomDDPGPolicy, self).__init__(
            observation_space,
            action_space,
            *args,
            **kwargs,
        )

        # Feature extraction dimensions
        self.features_dim = observation_space.shape[0]

        # Actor network
        self.actor = CustomActor(self.features_dim)
        self.actor_target = CustomActor(self.features_dim)
        # Critic network
        self.critic = CustomCritic(self.features_dim)
        self.critic_target = CustomCritic(self.features_dim)

        # Action noise for exploration
        self.action_noise = NormalActionNoise(
            mean=th.zeros(action_space.shape), sigma=0.1 * th.ones(action_space.shape)
        )

        # Initialize optimizers
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=lr_schedule(1))
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=lr_schedule(1))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.actor(obs)

    def critic_value(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        return self.critic(th.cat([obs, actions], dim=1))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        Predict actions for the given observation.
        :param observation: (th.Tensor) Observations from the environment.
        :param deterministic: (bool) Whether to use deterministic or stochastic actions.
        :return: (th.Tensor) Predicted actions.
        """
        actions = self.forward(observation).detach()
        if deterministic:
            return actions, None
        else:
            # Apply noise for exploration in non-deterministic settings
            return actions.cpu().numpy() + self.action_noise(), None

    def predict(self, observation: th.Tensor, state, start, deterministic: bool = True) -> th.Tensor:
        return self._predict(observation, deterministic)

class TrainerFollowLaneDDPGCarla:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DDPG
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):

        pynvml.nvmlInit()

        self.actor_loss = 0
        self.critic_loss = 0
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.global_params = LoadGlobalParams(config)
        self.environment = LoadEnvVariablesDDPGCarla(config)
        self.environment.environment["debug_waypoints"] = False
        self.environment.environment["estimated_steps"] = 5000
        logs_dir = f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        self.tensorboard = ModifiedTensorBoard(
            log_dir=logs_dir
        )
        self.environment.environment["tensorboard"] = self.tensorboard

        self.loss = 0

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)

        agent_log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_baselines_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        # self.log = LoggingHandler(agent_log_file)

        ## Load Carla server
        # CarlaEnv.__init__(self)

        self.environment.environment["entropy_factor"] = config["settings"]["entropy_factor"]
        self.env = gym.make(self.env_params.env_name, **self.environment.environment)
        self.all_steps = 0
        self.current_max_reward = 0
        self.best_epoch = 0
        self.episodes_speed = []
        self.episodes_d_reward = []
        self.episodes_steer = []
        self.episodes_reward = []
        self.step_fps = []
        self.bad_perceptions = 0
        self.crash = 0

        self.all_steps_reward = []
        self.all_steps_velocity = []
        self.all_steps_steer = []
        self.all_steps_state0 = []
        self.all_steps_state11 = []

        self.cpu_usages = 0
        self.gpu_usages = 0

        self.exploration = self.algoritmhs_params.std_dev if self.global_params.mode != "inference" else 0

        self.params = {
            "policy": "CustomPolicy",
            "learning_rate": 0.00035,
            "buffer_size": 100000,
            "batch_size": 256,
            "gamma": 0.90,
            "tau": 0.005,
            "total_timesteps": 5000000
        }

        n_actions = self.env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

        # Init Agents
        if self.environment.environment["mode"] in ["inference", "retraining"]:
            actor_retrained_model = self.environment.environment['retrain_ddpg_tf_model_name']
            self.ddpg_agent = DDPG.load(actor_retrained_model)
            # Set the environment on the loaded model
            self.ddpg_agent.set_env(self.env)

        else:
            # Assuming `self.params` and `self.global_params` are defined properly
            self.ddpg_agent = DDPG(
                # CustomDDPGPolicy,
                "MlpPolicy",
                self.env,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64,64, 64], qf=[64, 64, 64, 64])),
                learning_rate=self.params["learning_rate"],
                buffer_size=self.params["buffer_size"],
                batch_size=self.params["batch_size"],
                tau=self.params["tau"],
                gamma=self.params["gamma"],
                verbose=1,
                # tensorboard_log=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
            )

        print(self.ddpg_agent.actor)
        # Set the action noise on the loaded model
        self.ddpg_agent.action_noise = action_noise

        agent_logger = configure(agent_log_file, ["stdout", "csv", "tensorboard"])

        self.ddpg_agent.set_logger(agent_logger)
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

    def main(self):
        run = wandb.init(
            project="rl-follow-lane",
            config=self.params,
            sync_tensorboard=True,
        )
        exploration_rate_callback = ExplorationRateCallback(self.tensorboard,
                                                            initial_exploration_rate=self.exploration,
                                                            decay_rate= self.global_params.decrease_substraction,
                                                            decay_steps=self.global_params.steps_to_decrease,
                                                            exploration_min=self.global_params.decrease_min,
                                                            verbose=1)
        # wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        periodic_save_callback = PeriodicSaveCallback(
            save_path=f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}",
            verbose=1
        )

        callback_list = CallbackList([exploration_rate_callback, eval_callback, periodic_save_callback])

        self.ddpg_agent.learn(total_timesteps=self.params["total_timesteps"],
                              callback=callback_list)

        # self.env.close()

    def set_stats(self, info, prev_state):
        self.episodes_speed.append(info["velocity"])
        self.episodes_steer.append(info["steering_angle"])
        self.episodes_d_reward.append(info["d_reward"])
        self.episodes_reward.append(info["reward"])

        self.all_steps_reward.append(info["reward"])
        self.all_steps_velocity.append(info["velocity"])
        self.all_steps_steer.append(info["steering_angle"])
        self.all_steps_state0.append(prev_state[0])
        self.all_steps_state11.append(prev_state[6])

        pass