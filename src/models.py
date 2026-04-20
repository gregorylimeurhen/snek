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


class TransformerEvaluator(nn.Module):
	def __init__(self, hidden_size, depth):
		super().__init__()
		self.p = nn.Embedding(depth, hidden_size)
		layer = nn.TransformerEncoderLayer(hidden_size, 1, 2 * hidden_size, 0.0, batch_first=True)
		self.tf = nn.TransformerEncoder(layer, 1)
		self.s = nn.Linear(hidden_size, 1)
		self.c = nn.Linear(hidden_size, 1)

	def forward(self, hs):
		t = hs.shape[1]
		i = torch.arange(t, device=hs.device)
		p = self.p(i).unsqueeze(0)
		m = torch.full((t, t), float("-inf"), device=hs.device).triu(1)
		ys = self.tf(hs + p, mask=m)
		survival = torch.sigmoid(self.s(ys[:, -1]).squeeze(1))
		consumption = torch.sigmoid(self.c(ys).squeeze(-1))
		return survival, consumption


def build_evaluator(hidden_size, depth, evaluator_type):
	if evaluator_type == "gru":
		return Evaluator(hidden_size)
	if evaluator_type == "tf":
		return TransformerEvaluator(hidden_size, depth)
	raise ValueError("evaluator_type must be gru or tf")


class System(nn.Module):
	def __init__(self, image_size, hidden_size, depth=1, evaluator_type="gru"):
		super().__init__()
		self.enc = Encoder(image_size, hidden_size)
		self.wm = WorldModel(hidden_size)
		self.ev = build_evaluator(hidden_size, depth, evaluator_type)

	def rollout_h(self, h, acts):
		hs = []
		for i in range(acts.shape[1]):
			h = self.wm(h, acts[:, i])
			hs.append(h)
		hs = torch.stack(hs, 1)
		survival, consumption = self.ev(hs)
		return hs, survival, consumption

	def rollout(self, x0, acts):
		h = self.enc(x0)
		return self.rollout_h(h, acts)


class SpatialEncoder(nn.Module):
	def __init__(self, extent, hidden_size):
		super().__init__()
		self.c1 = nn.Conv2d(3, 32, 3, padding=1)
		self.c2 = nn.Conv2d(32, hidden_size, 3, padding=1)
		self.p = nn.AdaptiveAvgPool2d((extent, extent))

	def forward(self, x):
		x = F.relu(self.c1(x))
		x = F.relu(self.c2(x))
		return self.p(x)


class SpatialWorldModel(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.a = nn.Embedding(5, hidden_size)
		self.c1 = nn.Conv2d(2 * hidden_size, hidden_size, 3, padding=1)
		self.c2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)

	def forward(self, h, a):
		y = self.a(a).unsqueeze(-1).unsqueeze(-1)
		y = y.expand(-1, -1, h.shape[2], h.shape[3])
		x = torch.cat([h, y], 1)
		x = F.relu(self.c1(x))
		return h + self.c2(x)


class SpatialEvaluator(nn.Module):
	def __init__(self, extent, hidden_size, depth, evaluator_type):
		super().__init__()
		self.a = nn.Linear(hidden_size * extent * extent, hidden_size)
		self.e = build_evaluator(hidden_size, depth, evaluator_type)

	def forward(self, hs):
		zs = hs.flatten(2)
		zs = self.a(zs)
		return self.e(zs)


class SpatialSystem(nn.Module):
	def __init__(self, extent, hidden_size, depth=1, evaluator_type="gru"):
		super().__init__()
		self.enc = SpatialEncoder(extent, hidden_size)
		self.wm = SpatialWorldModel(hidden_size)
		self.ev = SpatialEvaluator(extent, hidden_size, depth, evaluator_type)

	def rollout_h(self, h, acts):
		hs = []
		for i in range(acts.shape[1]):
			h = self.wm(h, acts[:, i])
			hs.append(h)
		hs = torch.stack(hs, 1)
		survival, consumption = self.ev(hs)
		return hs, survival, consumption

	def rollout(self, x0, acts):
		h = self.enc(x0)
		return self.rollout_h(h, acts)


def build_system(image_size, extent, hidden_size, depth, wm_type, evaluator_type):
	if wm_type == "spatial":
		return SpatialSystem(extent, hidden_size, depth, evaluator_type)
	if wm_type == "flat":
		return System(image_size, hidden_size, depth, evaluator_type)
	raise ValueError("wm_type must be flat or spatial")
