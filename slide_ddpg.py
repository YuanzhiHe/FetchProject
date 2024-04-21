import gymnasium as gym
from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import datetime

env_name = "FetchSlide-v2"
env = gym.make(env_name)
env = Monitor(env)
model = DDPG(
    env=env,
    policy='MultiInputPolicy',
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
    tau=0.05,
    buffer_size=1000000,
    batch_size=256,
    gamma=0.95,
    learning_rate=0.001,
    verbose=1,
    policy_kwargs=dict(net_arch=[512, 512, 512], n_critics=2),
    tensorboard_log="./tensorboard/FetchSlide_DDPG_HER-v2/"
)
model.learn(total_timesteps=3e6)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)
model.save("./model/FetchSlide_ddpg.pkl")
print(datetime.datetime.now())
