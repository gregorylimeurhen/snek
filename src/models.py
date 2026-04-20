import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self, image_size, hidden_size):
		super().__init__()
		self.c1 = nn.Conv2d(3, 32, 3, padding=1)
		self.c2 = nn.Conv2d(32, 32, 3, padding=1)
		self.f = nn.Linear(32 * image_size * image_size, hidden_size)

	def forward(self, x):
		x = F.relu(self.c1(x))
		x = F.relu(self.c2(x))
		x = x.flatten(1)
		return self.f(x)


class WorldModel(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.a = nn.Embedding(5, 16)
		self.f1 = nn.Linear(hidden_size + 16, hidden_size)
		self.f2 = nn.Linear(hidden_size, hidden_size)

	def forward(self, h, a):
		x = torch.cat([h, self.a(a)], 1)
		x = F.relu(self.f1(x))
		return h + self.f2(x)


class Evaluator(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
		self.s = nn.Linear(hidden_size, 1)
		self.c = nn.Linear(hidden_size, 1)

	def forward(self, hs):
		ys, _ = self.rnn(hs)
		survival = torch.sigmoid(self.s(ys[:, -1]).squeeze(1))
		consumption = torch.sigmoid(self.c(ys).squeeze(-1))
		return survival, consumption


class System(nn.Module):
	def __init__(self, image_size, hidden_size):
		super().__init__()
		self.enc = Encoder(image_size, hidden_size)
		self.wm = WorldModel(hidden_size)
		self.ev = Evaluator(hidden_size)

	def rollout(self, x0, acts):
		h = self.enc(x0)
		hs = []
		for i in range(acts.shape[1]):
			h = self.wm(h, acts[:, i])
			hs.append(h)
		hs = torch.stack(hs, 1)
		survival, consumption = self.ev(hs)
		return hs, survival, consumption
