import random


def _in(extent, point):
	return 1 <= point[0] <= extent and 1 <= point[1] <= extent


class HamiltonianPolicy:
	_random = {}

	def __init__(self, extent, cycle=None):
		self.extent = int(extent)
		if self.extent < 2 or self.extent % 2:
			raise ValueError("Hamiltonian cycle requires an even extent")
		self.cycle = self._build(self.extent) if cycle is None else tuple(cycle)
		self.pos = {point: i for i, point in enumerate(self.cycle)}

	def _build(self, extent):
		cycle = [(1, col) for col in range(1, extent + 1)]
		for row in range(2, extent + 1):
			cols = range(extent, 1, -1) if row % 2 == 0 else range(2, extent + 1)
			cycle.extend((row, col) for col in cols)
		cycle.extend((row, 1) for row in range(extent, 1, -1))
		return tuple(cycle)

	def index(self, point):
		return self.pos[point]

	def successor(self, point):
		i = (self.index(point) + 1) % len(self.cycle)
		return self.cycle[i]

	def predecessor(self, point):
		i = (self.index(point) - 1) % len(self.cycle)
		return self.cycle[i]

	def advance(self, point, direction):
		if direction == 1:
			return self.successor(point)
		if direction == -1:
			return self.predecessor(point)
		raise ValueError("direction must be 1 or -1")

	def direction(self, snake):
		if snake[1] == self.successor(snake[0]):
			return -1
		return 1

	def distance(self, start, end, direction=1):
		if direction == 1:
			return (self.index(end) - self.index(start)) % len(self.cycle)
		if direction == -1:
			return (self.index(start) - self.index(end)) % len(self.cycle)
		raise ValueError("direction must be 1 or -1")

	def next_point(self, snake, direction):
		return self.advance(snake[0], direction)

	def move_to(self, snake, point):
		head = snake[0]
		delta = point[0] - head[0], point[1] - head[1]
		straight = head[0] + head[0] - snake[1][0], head[1] + head[1] - snake[1][1]
		if point == straight:
			return 0, 0
		return delta

	def neighbors(self, point):
		row, col = point
		return (row, col + 1), (row, col - 1), (row - 1, col), (row + 1, col)

	def targets(self, snake):
		out = []
		for point in self.neighbors(snake[0]):
			if not _in(self.extent, point):
				continue
			if point == snake[1]:
				continue
			if point in snake[:-1]:
				continue
			out.append(point)
		return tuple(out)

	def ordered(self, snake, direction):
		rs = []
		for point in reversed(snake):
			rs.append(self.distance(snake[-1], point, direction))
		return all(left < right for left, right in zip(rs, rs[1:]))

	def rank(self, snake, point, direction):
		return self.distance(snake[-1], point, direction)

	def shortcut(self, snake, food, point, direction):
		head = self.rank(snake, snake[0], direction)
		rank = self.rank(snake, point, direction)
		if rank <= head:
			return False
		food_rank = self.rank(snake, food, direction)
		if food_rank > head and rank > food_rank:
			return False
		return True

	@classmethod
	def _pick(cls, rng, items, weights):
		total = sum(weights)
		choice = rng.randrange(total)
		for item, weight in zip(items, weights):
			if choice < weight:
				return item
			choice -= weight
		raise ValueError("weighted choice failed")

	@classmethod
	def _cycle_from_rows(cls, rows, width):
		height = len(rows)
		edges = set()
		for col in range(width):
			for row in range(height):
				if not (rows[row] >> col) & 1:
					continue
				parts = (
					((col + 1, row + 1), (col + 2, row + 1)),
					((col + 1, row + 1), (col + 1, row + 2)),
					((col + 2, row + 2), (col + 1, row + 2)),
					((col + 2, row + 2), (col + 2, row + 1)),
				)
				for left, right in parts:
					edge = (left, right) if left <= right else (right, left)
					if edge in edges:
						edges.remove(edge)
					else:
						edges.add(edge)
		out = set()
		for left, right in edges:
			a = left[0], height + 2 - left[1]
			b = right[0], height + 2 - right[1]
			edge = (a, b) if a <= b else (b, a)
			out.add(edge)
		nbrs = {}
		for row in range(1, width + 2):
			for col in range(1, height + 2):
				nbrs[row, col] = []
		for left, right in out:
			nbrs[left].append(right)
			nbrs[right].append(left)
		order = [(1, 1), (2, 1)]
		nbrs[1, 1] = [(1, 2)]
		nbrs[1, 2] = [(1, 1)]
		nbrs[2, 1] = [point for point in nbrs[2, 1] if point != (1, 1)]
		cur = (2, 1)
		while cur != (1, 2):
			nxt = nbrs[cur][0]
			order.append(nxt)
			nbrs[nxt] = [point for point in nbrs[nxt] if point != cur]
			cur = nxt
		return tuple(order)

	@classmethod
	def _sampler(cls, extent):
		data = cls._random.get(extent)
		if data is not None:
			return data
		n = extent - 1
		masks = tuple(range(1, 1 << n))
		bad = {(0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 0, 1), (1, 1, 1, 1)}

		def parts(mask):
			out = []
			i = 0
			while i < n:
				if not (mask >> i) & 1:
					i += 1
					continue
				block = 0
				while i < n and (mask >> i) & 1:
					block |= 1 << i
					i += 1
				out.append(block)
			return tuple(out)

		def comp(left, right):
			for i in range(n + 1):
				a0 = 0 if i == 0 else (left >> (i - 1)) & 1
				b0 = 0 if i == 0 else (right >> (i - 1)) & 1
				a1 = 0 if i == n else (left >> i) & 1
				b1 = 0 if i == n else (right >> i) & 1
				if (a0, b0, a1, b1) in bad:
					return False
			return True

		def merge(old, new):
			graph = [[] for _ in new]
			for i in range(len(new)):
				for j in range(i + 1, len(new)):
					for block in old:
						if not block & new[i]:
							continue
						if not block & new[j]:
							continue
						graph[i].append(j)
						graph[j].append(i)
						break
			out = []
			seen = set()
			for i in range(len(new)):
				if i in seen:
					continue
				stack = [i]
				mask = 0
				seen.add(i)
				while stack:
					j = stack.pop()
					mask |= new[j]
					for k in graph[j]:
						if k in seen:
							continue
						seen.add(k)
						stack.append(k)
				out.append(mask)
			out.sort()
			return tuple(out)

		def cycle1(old, new):
			for left in old:
				if left.bit_count() < 2:
					continue
				for right in new:
					if right.bit_count() < 2:
						continue
					if (left & right).bit_count() >= 2:
						return True
			return False

		def cycle2(old, new):
			hits = []
			for right in new:
				row = set()
				for i, left in enumerate(old):
					if left & right:
						row.add(i)
				hits.append(row)
			for i in range(len(hits)):
				for j in range(i + 1, len(hits)):
					if len(hits[i] & hits[j]) >= 2:
						return True
			return False

		def alive(old, new):
			mask = 0
			for block in new:
				mask |= block
			for block in old:
				if block & mask:
					continue
				return False
			return True

		starters = []
		enders = set()
		for mask in masks:
			if comp(0, mask):
				starters.append((mask, parts(mask)))
			if comp(mask, 0):
				enders.add((mask, (mask,)))
		starters = tuple(sorted(starters))
		followers = {}
		seen = set(starters)
		queue = list(starters)
		while queue:
			state = queue.pop()
			out = set()
			for mask in masks:
				if not comp(state[0], mask):
					continue
				raw = parts(mask)
				cur = merge(state[1], raw)
				if cycle1(state[1], raw) or cycle2(state[1], raw):
					continue
				if not alive(state[1], cur):
					continue
				nxt = mask, cur
				out.add(nxt)
				if nxt in seen:
					continue
				seen.add(nxt)
				queue.append(nxt)
			followers[state] = tuple(sorted(out))
		data = {"counts": {}, "enders": enders, "followers": followers, "rows": n, "starters": starters}
		cls._random[extent] = data
		return data

	@classmethod
	def random_backbone(cls, extent, rng):
		data = cls._sampler(extent)
		rows = data["rows"]
		enders = data["enders"]
		followers = data["followers"]
		counts = data["counts"]

		def count(state, left):
			key = state, left
			total = counts.get(key)
			if total is not None:
				return total
			if left == 1:
				total = int(state in enders)
				counts[key] = total
				return total
			total = 0
			for nxt in followers[state]:
				total += count(nxt, left - 1)
			counts[key] = total
			return total

		starts = []
		weights = []
		for state in data["starters"]:
			ways = count(state, rows)
			if not ways:
				continue
			starts.append(state)
			weights.append(ways)
		state = cls._pick(rng, starts, weights)
		mat = [state[0]]
		left = rows
		while left > 1:
			nexts = []
			weights = []
			for nxt in followers[state]:
				ways = count(nxt, left - 1)
				if not ways:
					continue
				nexts.append(nxt)
				weights.append(ways)
			state = cls._pick(rng, nexts, weights)
			mat.append(state[0])
			left -= 1
		cycle = cls._cycle_from_rows(mat, rows)
		return cls(extent, cycle)


class PerturbedHamiltonianPolicy:
	def __init__(self, extent, rng=None, base=None):
		self.extent = int(extent)
		self.rng = rng or random.Random()
		self.base = HamiltonianPolicy(self.extent) if base is None else base

	def _direction(self, snake):
		direction = self.base.direction(snake)
		if self.base.ordered(snake, direction):
			return direction
		if self.base.ordered(snake, -direction):
			return -direction
		return direction

	def action(self, state):
		if not state.alive:
			return 0, 0
		snake = state.snake
		direction = self._direction(snake)
		if state.food is None:
			point = self.base.next_point(snake, direction)
			return self.base.move_to(snake, point)
		if not self.base.ordered(snake, direction):
			point = self.base.next_point(snake, direction)
			return self.base.move_to(snake, point)
		best = [self.base.next_point(snake, direction)]
		dist = self.base.distance(best[0], state.food, direction)
		for point in self.base.targets(snake):
			if not self.base.shortcut(snake, state.food, point, direction):
				continue
			cur = self.base.distance(point, state.food, direction)
			if cur < dist:
				best = [point]
				dist = cur
				continue
			if cur == dist:
				best.append(point)
		point = self.rng.choice(best)
		return self.base.move_to(snake, point)


class RandomBackbonePerturbedHamiltonianPolicy(PerturbedHamiltonianPolicy):
	def __init__(self, extent, rng=None):
		rng = random.Random() if rng is None else rng
		base = HamiltonianPolicy.random_backbone(extent, rng)
		super().__init__(extent, rng, base)
