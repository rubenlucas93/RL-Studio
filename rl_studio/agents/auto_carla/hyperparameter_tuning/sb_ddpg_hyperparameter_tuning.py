import optuna
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
import gym
from rl_studio.agents.f1.loaders import LoadEnvVariablesDDPGCarla, LoadEnvParams
import yaml


def objective(trial):
    steps_completed = 0  # Counter for completed steps

    try:
        with open("/home/ruben/Desktop/RL-Studio/rl_studio/config/config_training_followlane_bs_ddpg_f1_carla.yaml",
                  'r') as file:
            config_file = yaml.safe_load(file)

        environment = LoadEnvVariablesDDPGCarla(config_file)
        env_params = LoadEnvParams(config_file)
        environment.environment["debug_waypoints"] = False
        environment.environment["estimated_steps"] = 5000
        environment.environment["entropy_factor"] = trial.suggest_uniform('entropy_factor', 0, 0.1)
        environment.environment["punish_zig_zag_value"] = trial.suggest_uniform('punish_zig_zag_value', 0, 3)

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
        eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/hyp_tuning/model',
                                     log_path='./logs/hyp_tuning', eval_freq=2000)

        # Total number of timesteps for training
        total_timesteps = 15000
        timesteps_per_iteration = 1000  # How many timesteps to train per iteration

        # Loop through to train the model in increments
        while steps_completed < total_timesteps:
            try:
                model.learn(total_timesteps=timesteps_per_iteration, callback=eval_callback)
                steps_completed += timesteps_per_iteration

                # Report intermediate result to Optuna
                trial.report(eval_callback.best_mean_reward, steps_completed)

                # Check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            except Exception as e:
                print(f"Exception occurred at step {steps_completed}: {e}. Continuing the trial.")

        # Return the best mean reward after training
        return eval_callback.best_mean_reward

    except Exception as e:
        print(f"Trial setup failed with exception: {e}")
        raise optuna.exceptions.TrialPruned()  # Prune the trial if setup fails


# Create the study
study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())

# Optimize with a try-except block to catch failures
study.optimize(objective, n_trials=15)

# Set logging level
optuna.logging.set_verbosity(optuna.logging.INFO)

# Visualize the optimization history and parameter importance
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)

# Retrieve the best trial
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Retrieve and print all trials' results
print("\nCompleted trials so far:")
for trial in study.trials:
    print(f"Trial {trial.number}:")
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    print(f"  State: {trial.state}")  # Shows whether it succeeded, failed, or was pruned
