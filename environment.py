"""
ARIA Environment — Phase 2
Grid world with currency nodes, coordination nodes, stigmergy, and finite resources.

State per agent (14 elements):
  (x, y,
   currency_dir, coord_dir, partner_dir,
   currency_dist, coord_dist, partner_dist,
   msg[0], msg[1], msg[2], msg[3],
   trail_dir, trail_dist)

Stigmergy:
  A 16×16 marker grid persists across episodes. Agents leave pheromone-like
  markers when they collect currency (strength 0.7) or achieve coordination
  (strength 1.0). Markers decay by MARKER_DECAY each step. The state encodes
  direction and distance to the nearest significant marker — agents can follow
  trails left by previous agents without any direct communication.

Finite resources:
  Currency nodes have limited stock (CURRENCY_NODE_CAPACITY collections each).
  On depletion, the node disappears from the active set and a regen timer starts.
  After CURRENCY_REGEN_STEPS the node returns at full stock. This creates genuine
  ecological pressure: agents clustering on one node deplete it and must disperse.
"""

import random
import numpy as np
from config import (
    GRID_SIZE, N_CURRENCY_NODES, N_COORD_NODES,
    REWARD_CURRENCY, REWARD_COORD, REWARD_STEP, N_SIGNALS,
    MAX_MSG_LEN, MIN_COORD_AGENTS, ENV_DRIFT_N_NODES,
    MARKER_DECAY, MARKER_THRESHOLD, MARKER_STRENGTH_COORD, MARKER_STRENGTH_CURR,
    CURRENCY_NODE_CAPACITY, CURRENCY_REGEN_STEPS,
)

_NO_SIGNAL = N_SIGNALS   # sentinel: empty message slot


class Environment:
    def __init__(self, agent_ids):
        self.grid_size       = GRID_SIZE
        self.agent_ids       = list(agent_ids)
        self.currency_nodes  = set()   # active (stocked) nodes
        self.coord_nodes     = set()
        self.agent_positions = {}
        self._msg_buffers    = {}
        self._last_msgs      = {}
        self.steps           = 0

        # Finite resources
        self._node_stock   = {}   # pos -> remaining stock (active nodes only)
        self._depleted     = {}   # pos -> steps until regen

        # Stigmergy: persists across episodes (environmental memory)
        self._markers = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        self.reset()

    def reset(self, agent_ids=None):
        if agent_ids is not None:
            self.agent_ids = list(agent_ids)
        self.currency_nodes  = set()
        self.coord_nodes     = set()
        self.agent_positions = {}
        self.steps           = 0
        self._msg_buffers    = {a: [] for a in self.agent_ids}
        self._last_msgs      = {a: [_NO_SIGNAL] * MAX_MSG_LEN for a in self.agent_ids}
        self._node_stock     = {}
        self._depleted       = {}
        # Markers are NOT reset — they persist as environmental memory
        self._place_entities()
        return self._get_states()

    def _place_entities(self):
        occupied = set()
        for agent_id in self.agent_ids:
            while True:
                pos = (random.randint(0, self.grid_size - 1),
                       random.randint(0, self.grid_size - 1))
                if pos not in occupied:
                    self.agent_positions[agent_id] = pos
                    occupied.add(pos)
                    break
        for _ in range(N_CURRENCY_NODES):
            while True:
                pos = (random.randint(0, self.grid_size - 1),
                       random.randint(0, self.grid_size - 1))
                if pos not in occupied:
                    self.currency_nodes.add(pos)
                    self._node_stock[pos] = CURRENCY_NODE_CAPACITY
                    occupied.add(pos)
                    break
        for _ in range(N_COORD_NODES):
            while True:
                pos = (random.randint(0, self.grid_size - 1),
                       random.randint(0, self.grid_size - 1))
                if pos not in occupied:
                    self.coord_nodes.add(pos)
                    occupied.add(pos)
                    break

    # ── Spatial helpers ────────────────────────────────────────────────────────

    def _nearest_direction(self, from_pos, node_set):
        if not node_set:
            return 8
        nearest = min(node_set,
                      key=lambda n: abs(n[0] - from_pos[0]) + abs(n[1] - from_pos[1]))
        dx = nearest[0] - from_pos[0]
        dy = nearest[1] - from_pos[1]
        if dx == 0 and dy == 0:
            return 8
        angle = np.arctan2(dy, dx)
        return int((angle + np.pi) / (2 * np.pi) * 8) % 8

    def _nearest_dist_bin(self, from_pos, node_set):
        """Chebyshev distance to nearest node, bucketed: 0=close(≤3), 1=mid(4-8), 2=far(>8)."""
        if not node_set:
            return 0
        dist = min(max(abs(n[0] - from_pos[0]), abs(n[1] - from_pos[1])) for n in node_set)
        if dist <= 3:
            return 0
        if dist <= 8:
            return 1
        return 2

    def _nearest_agent_id(self, agent_id):
        pos    = self.agent_positions[agent_id]
        others = [(a, self.agent_positions[a]) for a in self.agent_ids if a != agent_id]
        if not others:
            return None
        return min(others,
                   key=lambda x: max(abs(x[1][0] - pos[0]), abs(x[1][1] - pos[1])))[0]

    def _nearest_marker_dir_dist(self, pos):
        """
        Direction and distance bin to the nearest pheromone trail above threshold.
        Returns (8, 0) when no significant trail exists nearby.
        Excludes the agent's own cell so it always gets navigation guidance.
        """
        x, y = pos
        ys, xs = np.where(self._markers >= MARKER_THRESHOLD)
        if len(xs) == 0:
            return 8, 0

        # Exclude own cell — find direction TO a trail, not the one under foot
        mask  = ~((xs == x) & (ys == y))
        xs, ys = xs[mask], ys[mask]
        if len(xs) == 0:
            return 8, 0

        dists   = np.maximum(np.abs(xs - x), np.abs(ys - y))
        i       = int(dists.argmin())
        dx      = int(xs[i]) - x
        dy      = int(ys[i]) - y
        d       = int(dists[i])

        angle   = np.arctan2(dy, dx)
        dir_idx = int((angle + np.pi) / (2 * np.pi) * 8) % 8
        dist_bin = 0 if d <= 3 else (1 if d <= 8 else 2)
        return dir_idx, dist_bin

    # ── State ──────────────────────────────────────────────────────────────────

    def _get_state(self, agent_id):
        pos          = self.agent_positions[agent_id]
        currency_dir = self._nearest_direction(pos, self.currency_nodes)
        coord_dir    = self._nearest_direction(pos, self.coord_nodes)

        nearest      = self._nearest_agent_id(agent_id)
        partner_dir  = (self._nearest_direction(pos, {self.agent_positions[nearest]})
                        if nearest else 8)
        currency_dist = self._nearest_dist_bin(pos, self.currency_nodes)
        coord_dist    = self._nearest_dist_bin(pos, self.coord_nodes)
        partner_dist  = (self._nearest_dist_bin(pos, {self.agent_positions[nearest]})
                         if nearest else 0)

        # Messages received from agents within 1 cell — nearest heard first.
        # Contact bonus still requires same-cell collision (tracked in main.py).
        nearby = sorted(
            [a for a in self.agent_ids if a != agent_id
             and max(abs(self.agent_positions[a][0] - pos[0]),
                     abs(self.agent_positions[a][1] - pos[1])) <= 1],
            key=lambda a: max(abs(self.agent_positions[a][0] - pos[0]),
                              abs(self.agent_positions[a][1] - pos[1]))
        )
        if nearby and nearby[0] in self._last_msgs:
            received = tuple(self._last_msgs[nearby[0]])
        else:
            received = (_NO_SIGNAL,) * MAX_MSG_LEN

        trail_dir, trail_dist = self._nearest_marker_dir_dist(pos)

        return (pos[0], pos[1], currency_dir, coord_dir, partner_dir,
                currency_dist, coord_dist, partner_dist) + received + (trail_dir, trail_dist)

    def _get_states(self):
        return {a: self._get_state(a) for a in self.agent_ids}

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, actions, signals_sent):
        """
        actions:      dict agent_id -> int (0-3 move, 4+ signal)
        signals_sent: dict agent_id -> int signal index or None
        Returns: next_states, rewards, done, info, signals_sent

        Variable-length messages: consecutive signal actions accumulate in a
        buffer. The first move action flushes the buffer as a complete message.

        Finite resources: currency nodes deplete after CURRENCY_NODE_CAPACITY
        collections and regenerate after CURRENCY_REGEN_STEPS steps.

        Stigmergy: markers are placed on currency collection and coordination
        success. All markers decay by MARKER_DECAY each step.
        """
        self.steps += 1
        rewards = {a: REWARD_STEP for a in self.agent_ids}
        info    = {'coord_achieved': False, 'currency_collected': [],
                   'messages_sent': {}, 'coord_agents': []}

        deltas = [(0, -1), (0, 1), (1, 0), (-1, 0),    # N S E W
                  (1, -1), (-1, -1), (1, 1), (-1, 1)]  # NE NW SE SW

        # ── Message buffers ────────────────────────────────────────────────────
        for agent_id, action in actions.items():
            if action < 8:
                buf = self._msg_buffers[agent_id]
                if buf:
                    padded = buf + [_NO_SIGNAL] * (MAX_MSG_LEN - len(buf))
                    self._last_msgs[agent_id]      = padded
                    info['messages_sent'][agent_id] = list(buf)
                    self._msg_buffers[agent_id]    = []
            else:
                token = action - 8
                if len(self._msg_buffers[agent_id]) < MAX_MSG_LEN:
                    self._msg_buffers[agent_id].append(token)

        # ── Move agents ────────────────────────────────────────────────────────
        new_positions = {}
        for agent_id, action in actions.items():
            pos = self.agent_positions[agent_id]
            if action < 8:
                dx, dy = deltas[action]
                nx = max(0, min(self.grid_size - 1, pos[0] + dx))
                ny = max(0, min(self.grid_size - 1, pos[1] + dy))
                new_positions[agent_id] = (nx, ny)
            else:
                new_positions[agent_id] = pos
        self.agent_positions = new_positions

        # ── Finite resource: regen depleted nodes ──────────────────────────────
        regen_ready = [p for p, t in self._depleted.items() if t <= 1]
        for pos in regen_ready:
            del self._depleted[pos]
            self.currency_nodes.add(pos)
            self._node_stock[pos] = CURRENCY_NODE_CAPACITY
        for pos in self._depleted:
            self._depleted[pos] -= 1

        # ── Currency collection (finite stock) ─────────────────────────────────
        for agent_id in self.agent_ids:
            pos = self.agent_positions[agent_id]
            if pos in self.currency_nodes:
                stock = self._node_stock.get(pos, 0)
                if stock > 0:
                    self._node_stock[pos] = stock - 1
                    rewards[agent_id] += REWARD_CURRENCY
                    info['currency_collected'].append(agent_id)
                    # Stigmergy: mark currency location
                    self._place_marker(pos, MARKER_STRENGTH_CURR)
                    if self._node_stock[pos] == 0:
                        # Node depleted — remove from active set, start regen
                        self.currency_nodes.discard(pos)
                        del self._node_stock[pos]
                        self._depleted[pos] = CURRENCY_REGEN_STEPS

        # ── Coordination ───────────────────────────────────────────────────────
        for coord_pos in list(self.coord_nodes):
            near = [a for a in self.agent_ids
                    if max(abs(self.agent_positions[a][0] - coord_pos[0]),
                           abs(self.agent_positions[a][1] - coord_pos[1])) <= 1]
            if len(near) >= MIN_COORD_AGENTS:
                self.coord_nodes.discard(coord_pos)
                for a in near:
                    rewards[a] += REWARD_COORD
                info['coord_achieved'] = True
                info['coord_agents'].extend(near)
                # Stigmergy: strong marker at coordination point and agent positions
                self._place_marker(coord_pos, MARKER_STRENGTH_COORD)
                for a in near:
                    self._place_marker(self.agent_positions[a], MARKER_STRENGTH_COORD * 0.8)

        # ── Marker decay ───────────────────────────────────────────────────────
        self._markers *= MARKER_DECAY

        next_states = self._get_states()
        done        = not self.currency_nodes and not self.coord_nodes

        return next_states, rewards, done, info, signals_sent

    def _place_marker(self, pos, strength):
        """Place or reinforce a marker at pos, taking the max of existing and new strength."""
        x, y = pos
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self._markers[y, x] = max(self._markers[y, x], strength)

    # ── Node drift ─────────────────────────────────────────────────────────────

    def drift_nodes(self, n=None):
        """
        Relocate n active nodes to new random positions (environmental co-evolution).
        Only active (non-depleted) currency nodes are candidates.
        """
        if n is None:
            n = ENV_DRIFT_N_NODES
        all_nodes = list(self.currency_nodes) + list(self.coord_nodes)
        if not all_nodes:
            return
        to_move  = random.sample(all_nodes, min(n, len(all_nodes)))
        occupied = set(self.agent_positions.values())

        for old_pos in to_move:
            was_coord = old_pos in self.coord_nodes
            self.currency_nodes.discard(old_pos)
            self.coord_nodes.discard(old_pos)
            # Clean up stock for moved currency node
            self._node_stock.pop(old_pos, None)

            for _ in range(200):
                new_pos = (random.randint(0, self.grid_size - 1),
                           random.randint(0, self.grid_size - 1))
                if (new_pos not in occupied
                        and new_pos not in self.currency_nodes
                        and new_pos not in self.coord_nodes):
                    if was_coord:
                        self.coord_nodes.add(new_pos)
                    else:
                        self.currency_nodes.add(new_pos)
                        self._node_stock[new_pos] = CURRENCY_NODE_CAPACITY
                    occupied.add(new_pos)
                    break

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_positions(self):
        return dict(self.agent_positions)

    def get_nodes(self):
        return set(self.currency_nodes), set(self.coord_nodes)

    def get_marker_grid(self):
        """Return a copy of the current marker grid (for visualisation)."""
        return self._markers.copy()
