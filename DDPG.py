import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	'''
	基于MLP的Actor
	输入state，输出对应的action
	'''
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	'''
	基于MLP的Critic
	作为Q function approximator
	输入状态-动作，输出长期收益估值
	'''
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)


class DDPG(object):
	'''
	DDPG网络主体
	'''
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
		'''
		分别初始一个Actor和一个Critic，并对应构造两个参数不共享的target网络
		注意optimizer是policy网络而不是target网络的优化器
		'''
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor) # target网络参数不共享
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic) # target网络参数不共享
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

		self.discount = discount
		self.tau = tau


	def select_action(self, state):
		'''
		利用Actor选择动作
		'''
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state.reshape(1,-1)).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=64):
		'''
		训练一个step
		其中完成一次sample，以及actor和critic的梯度下降各一次
		'''
		# 从ReplayBuffer中采样一个minibatch
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# 利用target网络计算Q值估计
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# 计算当前Critic网络的估值
		current_Q = self.critic(state, action)

		# 计算Critic损失
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Critic梯度下降
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# 计算Actor损失
		actor_loss = -self.critic(state, self.actor(state)).mean()
		
		# Actor梯度下降 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# 应用Soft target update策略，更新两个target网络
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		'保存模型'
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		'载入模型'
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		