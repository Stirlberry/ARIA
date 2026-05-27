"""
ARIA-2 Environment — Lewis Signaling Game
Grid world with currency nodes, typed target nodes, ghost nodes, and fog of war.

Lewis game mechanics:
  - Agents have a type (0 or 1), assigned at birth.
  - Target nodes come in two types: target_nodes[0] and target_nodes[1].
  - A type-0 agent can ONLY see type-0 target nodes and vice versa.
  - Reward fires only when BOTH types occupy the SAME target tile simultaneously.
  - Agents only receive signals from cross-type agents.
  - Result: the sighted agent (who sees the target) MUST signal the blind agent.

State per agent (same element count as ARIA):
  (x, y,
   currency_dir, target_dir, partner_dir,
   currency_dist, target_dist, partner_dist,
   msg[0], msg[1], msg[2], msg[3],
   sender_id_int)

  target_dir/dist : direction/distance to nearest VISIBLE target of own type.
  partner_dir/dist: direction/distance to nearest CROSS-TYPE agent in fog radius.
  msg             : last message received from nearest cross-type agent in shout range.
"""

import random
import numpy as np
from config import (
    GRID_W, GRID_H, N_CURRENCY_NODES, N_TARGET_NODES_PER_TYPE,
    REWARD_CURRENCY, REWARD_COORD, REWARD_STEP, N_SIGNALS,
    MAX_MSG_LEN, ENV_DRIFT_N_NODES,
    CURRENCY_NODE_CAPACITY, REGEN_MIN_STEPS, REGEN_MAX_STEPS, FOG_RADIUS,
    ENERGY_FROM_CURRENCY, ENERGY_FROM_CO,
    GHOST_NODE_ACCESSES, CHATTER_RANGE, SHOUT_RANGE, ENERGY_MAX,
)

_NO_SIGNAL = N_SIGNALS


class Environment:
    def __init__(self, agent_ids, agent_types=None):
        self.grid_w          = GRID_W
        self.grid_h          = GRID_H
        self.agent_ids       = list(agent_ids)
        self.agent_types     = dict(agent_types) if agent_types else {}
        self.currency_nodes  = set()
        self.target_nodes    = [set(), set()]   # [type-0 targets, type-1 targets]
        self.agent_positions = {}
        self._msg_buffers    = {}
        self._last_msgs      = {}
        self.steps           = 0

        self._node_stock     = {}
        self._pending_spawns = []
        self.ghost_nodes     = {}

        self.reset()

    def set_agent_types(self, agent_types):
        self.agent_types = dict(agent_types)

    def reset(self, agent_ids=None, agent_types=None, soft=False):
        if agent_ids is not None:
            self.agent_ids = list(agent_ids)
        if agent_types is not None:
            self.agent_types = dict(agent_types)

        if not soft:
            self.currency_nodes  = set()
            self.target_nodes    = [set(), set()]
            self.agent_positions = {}
            self.steps           = 0
            self._node_stock     = {}
            self._pending_spawns = []
            self.ghost_nodes     = {}
            self._msg_buffers    = {a: [] for a in self.agent_ids}
            self._last_msgs      = {a: [_NO_SIGNAL] * MAX_MSG_LEN for a in self.agent_ids}
            self._place_entities()
        else:
            self.steps = 0
            self.agent_positions = {a: p for a, p in self.agent_positions.items()
                                    if a in self.agent_ids}
            occupied = (set(self.agent_positions.values()) |
                        self.currency_nodes |
                        self.target_nodes[0] |
                        self.target_nodes[1])
            for a in self.agent_ids:
                if a not in self.agent_positions:
                    for _ in range(200):
                        pos = (random.randint(0, self.grid_w - 1),
                               random.randint(0, self.grid_h - 1))
                        if pos not in occupied:
                            self.agent_positions[a] = pos
                            occupied.add(pos)
                            break
            old_bufs = self._msg_buffers
            old_msgs = self._last_msgs
            self._msg_buffers = {a: old_bufs.get(a, []) for a in self.agent_ids}
            self._last_msgs   = {a: old_msgs.get(a, [_NO_SIGNAL] * MAX_MSG_LEN)
                                 for a in self.agent_ids}

        return self._get_states()

    def _place_entities(self):
        occupied = set()
        for agent_id in self.agent_ids:
            while True:
                pos = (random.randint(0, self.grid_w - 1),
                       random.randint(0, self.grid_h - 1))
                if pos not in occupied:
                    self.agent_positions[agent_id] = pos
                    occupied.add(pos)
                    break

        for _ in range(N_CURRENCY_NODES):
            while True:
                pos = (random.randint(0, self.grid_w - 1),
                       random.randint(0, self.grid_h - 1))
                if pos not in occupied:
                    self.currency_nodes.add(pos)
                    self._node_stock[pos] = CURRENCY_NODE_CAPACITY
                    occupied.add(pos)
                    break

        for t_type in range(2):
            for _ in range(N_TARGET_NODES_PER_TYPE):
                while True:
                    pos = (random.randint(0, self.grid_w - 1),
                           random.randint(0, self.grid_h - 1))
                    if pos not in occupied:
                        self.target_nodes[t_type].add(pos)
                        occupied.add(pos)
                        break

    # ── Spatial helpers ────────────────────────────────────────────────────────

    def _chebyshev(self, pos_a, pos_b):
        return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1]))

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
        if not node_set:
            return 2
        dist = min(self._chebyshev(from_pos, n) for n in node_set)
        if dist <= 3:
            return 0
        if dist <= 8:
            return 1
        return 2

    def _in_fog(self, pos, target):
        return self._chebyshev(pos, target) > FOG_RADIUS

    def _visible_nodes(self, pos, node_set):
        return {n for n in node_set if not self._in_fog(pos, n)}

    # ── Spawning ───────────────────────────────────────────────────────────────

    def _queue_spawn(self, node_type):
        delay = random.randint(REGEN_MIN_STEPS, REGEN_MAX_STEPS)
        self._pending_spawns.append([delay, node_type])

    def _spawn_node(self, node_type):
        occupied = (set(self.agent_positions.values()) |
                    self.currency_nodes |
                    self.target_nodes[0] |
                    self.target_nodes[1])
        for _ in range(200):
            pos = (random.randint(0, self.grid_w - 1),
                   random.randint(0, self.grid_h - 1))
            if pos not in occupied:
                if node_type == 'currency':
                    self.currency_nodes.add(pos)
                    self._node_stock[pos] = CURRENCY_NODE_CAPACITY
                elif node_type == 'target_0':
                    self.target_nodes[0].add(pos)
                elif node_type == 'target_1':
                    self.target_nodes[1].add(pos)
                return

    # ── State ──────────────────────────────────────────────────────────────────

    def _get_state(self, agent_id):
        pos        = self.agent_positions[agent_id]
        agent_type = self.agent_types.get(agent_id, 0)
        cross_type = 1 - agent_type

        # Currency: visible to all
        visible_currency = self._visible_nodes(pos, self.currency_nodes)
        currency_dir     = self._nearest_direction(pos, visible_currency)
        currency_dist    = self._nearest_dist_bin(pos, visible_currency)

        # Own-type targets only (cross-type targets are invisible — Lewis game)
        my_targets       = self.target_nodes[agent_type]
        visible_targets  = self._visible_nodes(pos, my_targets)
        target_dir       = self._nearest_direction(pos, visible_targets)
        target_dist      = self._nearest_dist_bin(pos, visible_targets)

        # Partner = nearest CROSS-TYPE agent within fog radius
        cross_agent_positions = {
            self.agent_positions[a]
            for a in self.agent_ids
            if a != agent_id
            and self.agent_types.get(a, 0) == cross_type
            and a in self.agent_positions
            and not self._in_fog(pos, self.agent_positions[a])
        }
        partner_dir  = self._nearest_direction(pos, cross_agent_positions)
        partner_dist = self._nearest_dist_bin(pos, cross_agent_positions)

        # Hear signals from nearest cross-type agent within shout range
        cross_in_range = sorted(
            [a for a in self.agent_ids
             if a != agent_id
             and self.agent_types.get(a, 0) == cross_type
             and a in self.agent_positions
             and self._chebyshev(pos, self.agent_positions[a]) <= SHOUT_RANGE],
            key=lambda a: self._chebyshev(pos, self.agent_positions[a])
        )
        chatters = [a for a in cross_in_range
                    if self._chebyshev(pos, self.agent_positions[a]) <= CHATTER_RANGE]
        source   = chatters[0] if chatters else (cross_in_range[0] if cross_in_range else None)
        received = tuple(self._last_msgs[source]) if source and source in self._last_msgs \
                   else (_NO_SIGNAL,) * MAX_MSG_LEN

        if source is not None:
            try:
                sender_id_int = int(source.split('-')[1], 16)
            except (IndexError, ValueError):
                sender_id_int = 0
        else:
            sender_id_int = 0

        return (pos[0], pos[1], currency_dir, target_dir, partner_dir,
                currency_dist, target_dist, partner_dist) + received + (sender_id_int,)

    def _get_states(self):
        return {a: self._get_state(a) for a in self.agent_ids}

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, actions, signals_sent, agent_energies=None):
        self.steps += 1
        rewards = {a: REWARD_STEP for a in self.agent_ids}
        info    = {
            'coord_achieved':    False,
            'currency_collected': [],
            'messages_sent':     {},
            'coord_agents':      [],
            'energy_gains':      {},
            'ghost_collected':   [],
        }

        deltas = [(0, -1), (0, 1), (1, 0), (-1, 0),
                  (1, -1), (-1, -1), (1, 1), (-1, 1)]

        # ── Message buffers ────────────────────────────────────────────────────
        for agent_id, action in actions.items():
            if action < 8:
                buf = self._msg_buffers[agent_id]
                if buf:
                    padded = buf + [_NO_SIGNAL] * (MAX_MSG_LEN - len(buf))
                    self._last_msgs[agent_id]       = padded
                    info['messages_sent'][agent_id] = list(buf)
                    self._msg_buffers[agent_id]     = []
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
                nx = max(0, min(self.grid_w - 1, pos[0] + dx))
                ny = max(0, min(self.grid_h - 1, pos[1] + dy))
                new_positions[agent_id] = (nx, ny)
            else:
                new_positions[agent_id] = pos
        self.agent_positions = new_positions

        # ── Pending spawns ─────────────────────────────────────────────────────
        still_pending = []
        for entry in self._pending_spawns:
            entry[0] -= 1
            if entry[0] <= 0:
                self._spawn_node(entry[1])
            else:
                still_pending.append(entry)
        self._pending_spawns = still_pending

        # ── Currency collection ────────────────────────────────────────────────
        for agent_id in self.agent_ids:
            pos = self.agent_positions[agent_id]
            if pos in self.currency_nodes:
                stock  = self._node_stock.get(pos, 0)
                energy = agent_energies.get(agent_id, 0) if agent_energies else 0
                if stock > 0 and energy < ENERGY_MAX:
                    self._node_stock[pos] = stock - 1
                    rewards[agent_id] += REWARD_CURRENCY
                    info['currency_collected'].append(agent_id)
                    info['energy_gains'][agent_id] = (
                        info['energy_gains'].get(agent_id, 0) + ENERGY_FROM_CURRENCY)
                    if self._node_stock[pos] == 0:
                        self.currency_nodes.discard(pos)
                        del self._node_stock[pos]
                        self._queue_spawn('currency')

        # ── Lewis game coordination: both types must be on the same target tile ─
        for t_type in range(2):
            for target_pos in list(self.target_nodes[t_type]):
                type0_here = [a for a in self.agent_ids
                              if self.agent_positions.get(a) == target_pos
                              and self.agent_types.get(a, 0) == 0]
                type1_here = [a for a in self.agent_ids
                              if self.agent_positions.get(a) == target_pos
                              and self.agent_types.get(a, 0) == 1]
                if type0_here and type1_here:
                    participants = type0_here + type1_here
                    self.target_nodes[t_type].discard(target_pos)
                    for a in participants:
                        rewards[a] += REWARD_COORD
                        info['energy_gains'][a] = (
                            info['energy_gains'].get(a, 0) + ENERGY_FROM_CO)
                    info['coord_achieved'] = True
                    info['coord_agents'].extend(participants)
                    self._queue_spawn(f'target_{t_type}')

        # ── Ghost node collection ──────────────────────────────────────────────
        for agent_id in self.agent_ids:
            pos = self.agent_positions[agent_id]
            if pos in self.ghost_nodes:
                ghost = self.ghost_nodes[pos]
                info['ghost_collected'].append((agent_id, ghost['data']))
                ghost['accesses'] -= 1
                if ghost['accesses'] <= 0:
                    del self.ghost_nodes[pos]

        next_states = self._get_states()
        done        = (not self.currency_nodes
                       and not self.target_nodes[0]
                       and not self.target_nodes[1]
                       and not self._pending_spawns)

        return next_states, rewards, done, info, signals_sent

    # ── Node drift ─────────────────────────────────────────────────────────────

    def drift_nodes(self, n=None):
        if n is None:
            n = ENV_DRIFT_N_NODES
        all_nodes = ([(pos, 'currency') for pos in self.currency_nodes] +
                     [(pos, 'target_0') for pos in self.target_nodes[0]] +
                     [(pos, 'target_1') for pos in self.target_nodes[1]])
        if not all_nodes:
            return
        to_move  = random.sample(all_nodes, min(n, len(all_nodes)))
        occupied = set(self.agent_positions.values())

        for old_pos, node_type in to_move:
            if node_type == 'currency':
                self.currency_nodes.discard(old_pos)
                self._node_stock.pop(old_pos, None)
            elif node_type == 'target_0':
                self.target_nodes[0].discard(old_pos)
            elif node_type == 'target_1':
                self.target_nodes[1].discard(old_pos)

            for _ in range(200):
                new_pos = (random.randint(0, self.grid_w - 1),
                           random.randint(0, self.grid_h - 1))
                if (new_pos not in occupied
                        and new_pos not in self.currency_nodes
                        and new_pos not in self.target_nodes[0]
                        and new_pos not in self.target_nodes[1]):
                    if node_type == 'currency':
                        self.currency_nodes.add(new_pos)
                        self._node_stock[new_pos] = CURRENCY_NODE_CAPACITY
                    elif node_type == 'target_0':
                        self.target_nodes[0].add(new_pos)
                    elif node_type == 'target_1':
                        self.target_nodes[1].add(new_pos)
                    occupied.add(new_pos)
                    break

    # ── Ghost nodes ────────────────────────────────────────────────────────────

    def add_ghost_node(self, pos, data):
        if pos not in self.ghost_nodes:
            self.ghost_nodes[pos] = {'data': data, 'accesses': GHOST_NODE_ACCESSES}

    def add_newborn(self, agent_id, position, agent_type=0):
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
        self.agent_positions[agent_id] = position
        self.agent_types[agent_id]     = agent_type
        self._msg_buffers[agent_id]    = []
        self._last_msgs[agent_id]      = [_NO_SIGNAL] * MAX_MSG_LEN

    def clear_msg_buffer(self, agent_id):
        if agent_id in self._msg_buffers:
            self._msg_buffers[agent_id] = []

    def remove_agent(self, agent_id):
        self.agent_ids         = [a for a in self.agent_ids if a != agent_id]
        self.agent_positions.pop(agent_id, None)
        self.agent_types.pop(agent_id, None)
        self._msg_buffers.pop(agent_id, None)
        self._last_msgs.pop(agent_id, None)

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_positions(self):
        return dict(self.agent_positions)

    def get_nodes(self):
        return (set(self.currency_nodes),
                set(self.target_nodes[0]),
                set(self.target_nodes[1]))

    def get_ghost_nodes(self):
        return dict(self.ghost_nodes)
