"""
ARIA Environment — Phase 3
Grid world with currency nodes, CO nodes, ghost nodes, fog of war, and finite resources.

State per agent (12 elements):
  (x, y,
   currency_dir, coord_dir, partner_dir,
   currency_dist, coord_dist, partner_dist,
   msg[0], msg[1], msg[2], msg[3])

Communication:
  Chatter (CHATTER_RANGE=1): messages from adjacent agents take priority.
  Shout  (SHOUT_RANGE=3):    if no adjacent message, the nearest agent within
                              shout range is heard instead.

Energy:
  Currency nodes yield ENERGY_FROM_CURRENCY per collection (returned in info).
  CO nodes yield ENERGY_FROM_CO per participating agent (returned in info).
  Ghost nodes (dead agent knowledge) yield one absorption event then disappear.

CO hold:
  An agent waiting alone on a CO node accumulates hold steps. When it exceeds
  CO_HOLD_MAX_STEPS the environment reports a timeout in info so the caller
  can apply a nudge; the node itself is unaffected.
"""

import random
import numpy as np
from config import (
    GRID_SIZE, N_CURRENCY_NODES, N_COORD_NODES,
    REWARD_CURRENCY, REWARD_COORD, REWARD_STEP, N_SIGNALS,
    MAX_MSG_LEN, MIN_COORD_AGENTS, ENV_DRIFT_N_NODES,
    CURRENCY_NODE_CAPACITY, REGEN_MIN_STEPS, REGEN_MAX_STEPS, FOG_RADIUS,
    ENERGY_FROM_CURRENCY, ENERGY_FROM_CO,
    GHOST_NODE_ACCESSES, CHATTER_RANGE, SHOUT_RANGE, CO_HOLD_MAX_STEPS,
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

        self._node_stock      = {}   # pos -> remaining stock (active nodes only)
        self._pending_spawns  = []   # list of [steps_remaining, node_type]
        self.ghost_nodes      = {}   # pos -> {'data': any, 'accesses': int}
        self._co_hold_steps   = {}   # agent_id -> steps spent waiting alone on a CO node

        self.reset()

    def reset(self, agent_ids=None, soft=False):
        """
        soft=False (default): full reset — randomise all positions (startup / save-restore).
        soft=True:            keep positions; only clear step counter and message buffers.
                              New agents (births) are placed at a random empty cell.
                              Dead agents are removed from agent_positions.
        """
        if agent_ids is not None:
            self.agent_ids = list(agent_ids)

        if not soft:
            self.currency_nodes  = set()
            self.coord_nodes     = set()
            self.agent_positions = {}
            self.steps           = 0
            self._node_stock     = {}
            self._pending_spawns = []
            self.ghost_nodes     = {}
            self._co_hold_steps  = {}
            self._msg_buffers    = {a: [] for a in self.agent_ids}
            self._last_msgs      = {a: [_NO_SIGNAL] * MAX_MSG_LEN for a in self.agent_ids}
            self._place_entities()
        else:
            self.steps = 0
            # Remove agents that have left the population
            self.agent_positions = {a: p for a, p in self.agent_positions.items()
                                    if a in self.agent_ids}
            self._co_hold_steps  = {a: v for a, v in self._co_hold_steps.items()
                                    if a in self.agent_ids}
            # Place any new agents (newborns) at a random empty cell
            occupied = (set(self.agent_positions.values()) |
                        self.currency_nodes | self.coord_nodes)
            for a in self.agent_ids:
                if a not in self.agent_positions:
                    for _ in range(200):
                        pos = (random.randint(0, self.grid_size - 1),
                               random.randint(0, self.grid_size - 1))
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
            return 2
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

    # ── Spawning ───────────────────────────────────────────────────────────────

    def _queue_spawn(self, node_type):
        """Queue a node for random-position, random-time respawn."""
        delay = random.randint(REGEN_MIN_STEPS, REGEN_MAX_STEPS)
        self._pending_spawns.append([delay, node_type])

    def _spawn_node(self, node_type):
        """Place a node at a random cell not currently occupied by any agent."""
        occupied = (set(self.agent_positions.values()) |
                    self.currency_nodes | self.coord_nodes)
        for _ in range(200):
            pos = (random.randint(0, self.grid_size - 1),
                   random.randint(0, self.grid_size - 1))
            if pos not in occupied:
                if node_type == 'currency':
                    self.currency_nodes.add(pos)
                    self._node_stock[pos] = CURRENCY_NODE_CAPACITY
                else:
                    self.coord_nodes.add(pos)
                return

    # ── State ──────────────────────────────────────────────────────────────────

    def _in_fog(self, pos, target):
        """Return True if target is beyond the agent's fog-of-war radius."""
        return max(abs(target[0] - pos[0]), abs(target[1] - pos[1])) > FOG_RADIUS

    def _visible_nodes(self, pos, node_set):
        """Filter node_set to only those within FOG_RADIUS (Chebyshev)."""
        return {n for n in node_set if not self._in_fog(pos, n)}

    def _get_state(self, agent_id):
        pos = self.agent_positions[agent_id]

        visible_currency = self._visible_nodes(pos, self.currency_nodes)
        visible_coord    = self._visible_nodes(pos, self.coord_nodes)

        currency_dir  = self._nearest_direction(pos, visible_currency)
        coord_dir     = self._nearest_direction(pos, visible_coord)
        currency_dist = self._nearest_dist_bin(pos, visible_currency)
        coord_dist    = self._nearest_dist_bin(pos, visible_coord)

        nearest     = self._nearest_agent_id(agent_id)
        partner_pos = self.agent_positions[nearest] if nearest else None
        if partner_pos and not self._in_fog(pos, partner_pos):
            partner_dir  = self._nearest_direction(pos, {partner_pos})
            partner_dist = self._nearest_dist_bin(pos, {partner_pos})
        else:
            partner_dir  = 8
            partner_dist = 2

        def _chebyshev(a):
            return max(abs(self.agent_positions[a][0] - pos[0]),
                       abs(self.agent_positions[a][1] - pos[1]))

        others_in_range = sorted(
            [a for a in self.agent_ids if a != agent_id
             and _chebyshev(a) <= SHOUT_RANGE],
            key=_chebyshev
        )
        # Chatter (adjacent) takes priority over shout
        chatters = [a for a in others_in_range if _chebyshev(a) <= CHATTER_RANGE]
        source   = chatters[0] if chatters else (others_in_range[0] if others_in_range else None)
        received = tuple(self._last_msgs[source]) if source and source in self._last_msgs \
                   else (_NO_SIGNAL,) * MAX_MSG_LEN

        return (pos[0], pos[1], currency_dir, coord_dir, partner_dir,
                currency_dist, coord_dist, partner_dist) + received

    def _get_states(self):
        return {a: self._get_state(a) for a in self.agent_ids}

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, actions, signals_sent):
        """
        actions:      dict agent_id -> int (0-7 move, 8+ signal)
        signals_sent: dict agent_id -> int signal index or None
        Returns: next_states, rewards, done, info, signals_sent
        """
        self.steps += 1
        rewards = {a: REWARD_STEP for a in self.agent_ids}
        info    = {'coord_achieved': False, 'currency_collected': [],
                   'messages_sent': {}, 'coord_agents': [],
                   'energy_gains': {},        # agent_id -> energy gained this step
                   'ghost_collected': [],     # list of (agent_id, ghost_data)
                   'co_hold_timeout': []}

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

        # ── Pending spawns: tick down and spawn ready nodes ────────────────────
        still_pending = []
        for entry in self._pending_spawns:
            entry[0] -= 1
            if entry[0] <= 0:
                self._spawn_node(entry[1])
            else:
                still_pending.append(entry)
        self._pending_spawns = still_pending

        # ── Currency collection (finite stock) ─────────────────────────────────
        for agent_id in self.agent_ids:
            pos = self.agent_positions[agent_id]
            if pos in self.currency_nodes:
                stock = self._node_stock.get(pos, 0)
                if stock > 0:
                    self._node_stock[pos] = stock - 1
                    rewards[agent_id] += REWARD_CURRENCY
                    info['currency_collected'].append(agent_id)
                    info['energy_gains'][agent_id] = (
                        info['energy_gains'].get(agent_id, 0) + ENERGY_FROM_CURRENCY)
                    if self._node_stock[pos] == 0:
                        self.currency_nodes.discard(pos)
                        del self._node_stock[pos]
                        self._queue_spawn('currency')

        # ── CO node collection (requires MIN_COORD_AGENTS) ─────────────────────
        for coord_pos in list(self.coord_nodes):
            near = [a for a in self.agent_ids
                    if self.agent_positions[a] == coord_pos]
            if len(near) >= MIN_COORD_AGENTS:
                self.coord_nodes.discard(coord_pos)
                for a in near:
                    rewards[a] += REWARD_COORD
                    info['energy_gains'][a] = (
                        info['energy_gains'].get(a, 0) + ENERGY_FROM_CO)
                info['coord_achieved'] = True
                info['coord_agents'].extend(near)
                self._queue_spawn('coord')
            elif len(near) == 1:
                # One agent waiting alone — track hold steps
                a = near[0]
                self._co_hold_steps[a] = self._co_hold_steps.get(a, 0) + 1
                if self._co_hold_steps[a] >= CO_HOLD_MAX_STEPS:
                    info['co_hold_timeout'].append(a)
                    self._co_hold_steps[a] = 0   # reset so timeout fires once per window
            else:
                # No agent on this CO node — clear any stale hold counters for it
                pass

        # Reset hold counter for agents that left a CO node
        on_co = {a for coord_pos in self.coord_nodes
                 for a in self.agent_ids if self.agent_positions[a] == coord_pos}
        for a in list(self._co_hold_steps):
            if a not in on_co:
                self._co_hold_steps.pop(a, None)

        # ── Ghost node collection (one access, then gone) ──────────────────────
        for agent_id in self.agent_ids:
            pos = self.agent_positions[agent_id]
            if pos in self.ghost_nodes:
                ghost = self.ghost_nodes[pos]
                info['ghost_collected'].append((agent_id, ghost['data']))
                ghost['accesses'] -= 1
                if ghost['accesses'] <= 0:
                    del self.ghost_nodes[pos]

        next_states = self._get_states()
        done        = (not self.currency_nodes and not self.coord_nodes
                       and not self._pending_spawns)

        return next_states, rewards, done, info, signals_sent

    # ── Node drift ─────────────────────────────────────────────────────────────

    def drift_nodes(self, n=None):
        """
        Immediately relocate n active nodes to new random positions.
        This is an explicit environmental shift event, not a regen — nodes move
        at once rather than queuing through _pending_spawns.
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

    # ── Ghost nodes ────────────────────────────────────────────────────────────

    def add_ghost_node(self, pos, data):
        """Place a ghost node at pos carrying opaque data (dead agent's network weights)."""
        if pos not in self.ghost_nodes:
            self.ghost_nodes[pos] = {'data': data, 'accesses': GHOST_NODE_ACCESSES}

    def add_newborn(self, agent_id, position):
        """Register a newborn agent in the environment at a specific position."""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
        self.agent_positions[agent_id] = position
        self._msg_buffers[agent_id]    = []
        self._last_msgs[agent_id]      = [_NO_SIGNAL] * MAX_MSG_LEN
        self._co_hold_steps.setdefault(agent_id, 0)

    def clear_msg_buffer(self, agent_id):
        """Clear an agent's outgoing message buffer to suppress spawn-pause phantom signals."""
        if agent_id in self._msg_buffers:
            self._msg_buffers[agent_id] = []

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_positions(self):
        return dict(self.agent_positions)

    def get_nodes(self):
        return set(self.currency_nodes), set(self.coord_nodes)

    def get_ghost_nodes(self):
        return dict(self.ghost_nodes)
