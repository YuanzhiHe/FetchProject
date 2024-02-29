import gymnasium as gym
from stable_baselines3 import HerReplayBuffer, DDPG
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "HandReach-v1"
env = gym.make(env_name)
env = Monitor(env)
model = DDPG(
    env=env,
    policy='MultiInputPolicy',
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
    tau=0.05,
    buffer_size=1000000,
    batch_size=256,
    # gamma=0.95,
    learning_rate=0.001,
    verbose=1,  # output log
    policy_kwargs=dict(net_arch=[256, 256, 256]),
    tensorboard_log="./tensorboard/HandReach/"
)

model.learn(total_timesteps=2e6)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)
model.save("./model/HandReach.pkl")
