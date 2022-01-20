import numpy as np
import torch


class ReplayBuffer(object):
	'''
	GPU experience replay buffer
	将整个缓冲区都装入GPU的ReplayBuffer
	'''
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		'初始化状态空间维数、动作空间维数，以及ReplayBuffer总容量'
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.state = torch.zeros(max_size, state_dim, device=self.device,dtype=torch.float32)
		self.action = torch.zeros((max_size, action_dim), device=self.device,dtype=torch.float32)
		self.next_state = torch.zeros((max_size, state_dim), device=self.device,dtype=torch.float32)
		self.reward = torch.zeros((max_size, 1), device=self.device,dtype=torch.float32)
		self.not_done = torch.zeros((max_size, 1), device=self.device,dtype=torch.float32)

		
		print(f"ReplayBuffer device: {self.device}")


	def add(self, state, action, next_state, reward, done):
		'''
		向ReplayBuffer插入一个状态
		state: 当前状态
		action: 采取动作
		next_state: 转移状态
		reward: 当前状态-动作收益
		done: 是否结束
		'''
		self.state[self.ptr] = torch.tensor(state, device=self.device,dtype=torch.float32)
		self.action[self.ptr] = torch.tensor(action, device=self.device,dtype=torch.float32)
		self.next_state[self.ptr] = torch.tensor(next_state, device=self.device,dtype=torch.float32)
		self.reward[self.ptr] = torch.tensor(reward, device=self.device,dtype=torch.float32)
		self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device,dtype=torch.float32)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		'''
		采样大小为batch_size的一个minibatch
		'''
		ind = torch.randint(0, self.size, size=(batch_size,))
		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)