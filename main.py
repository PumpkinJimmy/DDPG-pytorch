import numpy as np
import torch
import gym
import argparse
import os

import utils
import DDPG
import tqdm

def eval_policy(policy, env_name, seed, eval_episodes=10):
	'对eval_episodes个轮次计算综合收益的平均值'
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym 环境名
	parser.add_argument("--seed", default=0, type=int)              # 随机种子
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Replay buffer 冷启动预热时间轮次
	parser.add_argument("--eval_freq", default=5e3, type=int)       # 估值频率
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # 最大训练步数
	parser.add_argument("--expl_noise", default=0.1)                # 探索高斯噪声的标准差
	parser.add_argument("--batch_size", default=256, type=int)      # batch大小
	parser.add_argument("--discount", default=0.99)                 # Discount系数
	parser.add_argument("--tau", default=0.005)                     # Soft update系数
	parser.add_argument("--save_model", action="store_true")        # 保存模型参数和优化器参数
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	policy = DDPG.DDPG(**kwargs)
		
	
	print(f"Device: {DDPG.device}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# 计算初始随机策略的收益水平
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in tqdm.tqdm(range(int(args.max_timesteps))):
		
		episode_timesteps += 1

		# 利用噪声随机选择探索的动作
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(state)
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# step
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# 存储经验
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# 热身以及结束了，可以训练
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# 记录数据，打checkpoint
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
