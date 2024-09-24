import optuna
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import gym
from rl_studio.agents.f1.loaders import (
    LoadEnvVariablesDDPGCarla,
    LoadEnvParams,
)
import argparse
import yaml

def objective(trial):

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument(
        "-f",
        "--file",
        type=argparse.FileType("r"),
        required=True
    )
    config_file = yaml.load(args.file, Loader=yaml.FullLoader)
    environment = LoadEnvVariablesDDPGCarla(config_file)
    env_params = LoadEnvParams(config_file)

    # Define the search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    buffer_size = trial.suggest_int('buffer_size', 50000, 1000000)
    batch_size = trial.suggest_int('batch_size', 64, 512)
    tau = trial.suggest_uniform('tau', 0.001, 0.01)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.99)
    environment.environment["beta_1"] = trial.suggest_uniform('beta', 0, 0.99)

    # Policy network architecture
    net_arch_pi = trial.suggest_categorical('net_arch_pi', [[32, 32], [64, 64, 64], [128, 128]])
    net_arch_qf = trial.suggest_categorical('net_arch_qf', [[32, 32, 32], [64, 64, 64], [128, 128, 128]])

    env = gym.make(env_params.env_name, **environment.environment)

    model = DDPG(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=net_arch_pi, qf=net_arch_qf)),
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        verbose=1
    )

    # Callback for evaluation
    eval_env = gym.make(env_params.env_name, **environment.environment)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/hyp_tuning/model', log_path='./logs/hyp_tuning', eval_freq=5000)

    # Train the model
    model.learn(total_timesteps=10000, callback=eval_callback)  # Set an appropriate number of timesteps

    # Evaluate the model
    mean_reward, _ = eval_callback.best_mean_reward, eval_callback.last_eval_mean_reward

    return mean_reward


study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=10)
optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
