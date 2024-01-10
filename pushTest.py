import datetime
import gymnasium as gym
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "FetchPush-v2"
env = gym.make(env_name, render_mode='human')
env2 = gym.make("FetchReach-v2")
# env = DummyVecEnv([lambda: env])  # Vectorization, reducing time for future use
env = Monitor(env)
model = TQC.load("./model/FetchReach_tqc_her2.pkl", env=env2)
model2 = TQC.load("./model/FetchPush_tqc_her.pkl", env=env)
model.tensorboard_log = "./tensorboard/FetchPushTest_TQC_HER-v2/"
model.policy.observation_space = env.observation_space
model.actor.observation_space = env.observation_space
model.critic.observation_space = env.observation_space
model.policy.critic_kwargs['observation_space'] = env.observation_space
model.critic_target.observation_space = env.observation_space
model.env.observation_space = env.observation_space
model.env.unwrapped.buf_obs = env.observation_space
model.observation_space = env.observation_space
model.learn(total_timesteps=1e6)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
env.close()
print(mean_reward, std_reward)
model.save("./model/FetchPushTest_tqc_her.pkl")
print(datetime.datetime.now())
