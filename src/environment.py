import random
import torch


ACTIONS = ((0, 0), (0, 1), (0, -1), (-1, 0), (1, 0))
AI = {a: i for i, a in enumerate(ACTIONS)}


class State:
	__slots__ = ("snake", "food", "time", "alive", "won")

	def __init__(self, snake, food, time=0, alive=True, won=False):
		self.snake = tuple(tuple(x) for x in snake)
		self.food = None if food is None else tuple(food)
		self.time = int(time)
		self.alive = bool(alive)
		self.won = bool(won)


class Simulator:
	def __init__(self, extent, upsampling=1, rng=None):
		self.extent = int(extent)
		self.upsampling = int(upsampling)
		self.rng = rng or random.Random()
		self.starts = self._starts()
		self.last_consumed = False
		self.state = None

	def _starts(self):
		out = []
		for row in range(1, self.extent + 1):
			for col in range(1, self.extent + 1):
				head = row, col
				for dr, dc in ACTIONS[1:]:
					neck = row + dr, col + dc
					if not self._in(neck):
						continue
					out.append((head, neck))
		return tuple(out)

	def _in(self, point):
		return 1 <= point[0] <= self.extent and 1 <= point[1] <= self.extent

	def _empty(self, snake):
		used = set(snake)
		for row in range(1, self.extent + 1):
			for col in range(1, self.extent + 1):
				cell = row, col
				if cell in used:
					continue
				return cell
		return None

	def reset(self, snake=None, food=None):
		snake = self.rng.choice(self.starts) if snake is None else snake
		snake = tuple(tuple(x) for x in snake)
		food = self._empty(snake) if food is None else tuple(food)
		won = food is None
		self.last_consumed = False
		self.state = State(snake, food, 0, not won, won)
		return self.state

	def snapshot(self):
		state = self.state
		rng_state = self.rng.getstate()
		return state.snake, state.food, state.time, state.alive, state.won, rng_state

	def restore(self, snap):
		snake, food, time, alive, won, rng_state = snap
		self.rng.setstate(rng_state)
		self.last_consumed = False
		self.state = State(snake, food, time, alive, won)
		return self.state

	def head(self, snake, action):
		if action == (0, 0):
			row = snake[0][0] + snake[0][0] - snake[1][0]
			col = snake[0][1] + snake[0][1] - snake[1][1]
			return row, col
		return snake[0][0] + action[0], snake[0][1] + action[1]

	def legal_actions(self, state=None):
		state = self.state if state is None else state
		if state.alive:
			return ACTIONS
		return ()

	def step(self, action):
		state = self.state
		self.last_consumed = False
		if not state.alive:
			return state
		head = self.head(state.snake, action)
		grew = head == state.food
		if grew:
			snake = (head,) + state.snake
		else:
			snake = (head,) + state.snake[:-1]
		body = set(snake[1:])
		if not self._in(head) or head in body:
			self.state = State(snake, None, state.time + 1, False, False)
			return self.state
		food = state.food
		won = False
		if grew:
			self.last_consumed = True
			food = self._empty(snake)
			won = food is None
		alive = not won
		self.state = State(snake, food, state.time + 1, alive, won)
		return self.state

	def display(self, state=None):
		state = self.state if state is None else state
		size = self.extent
		if not state.alive and not state.won:
			base = torch.zeros((size, size, 3), dtype=torch.uint8)
		elif state.won:
			base = torch.full((size, size, 3), 255, dtype=torch.uint8)
		else:
			base = torch.zeros((size, size, 3), dtype=torch.uint8)
			if state.food is not None:
				row, col = state.food
				base[row - 1, col - 1, 1] = 255
			scale = size * size
			for i, segment in enumerate(state.snake):
				row, col = segment
				value = round(255 * (1 - i / scale))
				base[row - 1, col - 1, 2] = int(value)
		if self.upsampling == 1:
			return base
		return base.repeat_interleave(self.upsampling, 0).repeat_interleave(self.upsampling, 1)
