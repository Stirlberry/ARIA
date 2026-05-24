"""
ARIA Agent — Phase 3
Full evolutionary DQN with brain evolution, world model, cultural inheritance,
curiosity, sub-goals, specialisation tracking, shared PER replay, dense rewards,
Theory of Mind, and energy-based survival.

Bottleneck fixes applied:
  - visit_counts: evicts states with count > _VISIT_COUNT_EVICT_THRESH when
    dict exceeds _VISIT_COUNT_MAX (prevents unbounded memory growth).
  - Training: one replay.sample() per cycle feeds Q-learning, world model,
    and Dyna-Q — down from 5 separate samples.
  - Dyna-Q uses random imagined actions on real states for diversity without
    extra buffer sampling.
  - ToM: partner features encoded once per step; single forward pass serves
    both the coordination-intent bonus and the training loss.
  - ToM warmup: bonus is suppressed until _tom_steps >= WORLD_MODEL_WARMUP
    to avoid misleading Q-values from random early predictions.
  - PER update_priorities: vectorised — numpy computes all (|err|+ε)^α at
    once; SumTree propagates via level-by-level numpy slicing instead of a
    Python loop.
  - torch.from_numpy() for zero-copy tensor construction in training.
"""

import json
import os
import random
import numpy as np
import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

from config import (
    GRID_SIZE, N_SIGNALS, MAX_MSG_LEN,
    LEARNING_RATE, DISCOUNT_FACTOR,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    HIDDEN_SIZE, N_LAYERS_DEFAULT, ACTIVATION_OPTIONS,
    BATCH_SIZE, REPLAY_BUFFER_SIZE,
    MIN_REPLAY_SIZE, TARGET_UPDATE_FREQ, TRAIN_FREQ,
    MUTATION_STD, INTRINSIC_BETA, EPISODIC_BETA, HYPERPARAM_MUTATION_STD,
    DYNA_STEPS, WORLD_MODEL_LR, WORLD_MODEL_HIDDEN, WORLD_MODEL_WARMUP,
    MEMORY_SIZE, MEMORY_REWARD_THRESH,
    SUB_GOAL_DURATION, SUB_GOAL_MAX_BONUS,
    REWARD_COORD, REWARD_CURRENCY,
    GOAL_DISCOVERY_MIN_SAMPLES, GOAL_DISCOVERY_CRYSTALLISE_EVERY,
    GOAL_DISCOVERY_MIN_LIFT, GOAL_DISCOVERY_BONUS, GOAL_DISCOVERY_LOG_PATH,
    SHARED_REPLAY_SIZE, PER_ALPHA, PER_BETA_START, PER_BETA_STEPS,
    POTENTIAL_COORD_WEIGHT, POTENTIAL_CURRENCY_WEIGHT,
    TOM_LR, TOM_POS_WEIGHT, TOM_INTENT_THRESHOLD, TOM_BONUS_SCALE,
    PLATEAU_DELTA_THRESH,
    SELF_REPL_WINDOW, SELF_REPL_COOLDOWN, SELF_REPL_FITNESS_MIN,
    ENERGY_START, ENERGY_MAX, ENERGY_NEWBORN,
    ENERGY_DRAIN_LOW, ENERGY_DRAIN_MED, ENERGY_DRAIN_HIGH,
)

# ── Input encoding ─────────────────────────────────────────────────────────────

_N_ENERGY_BINS = 5   # critical / low / medium / high / full (bin 4 = above reproduction threshold)

_INPUT_SIZE = (GRID_SIZE + GRID_SIZE              # x, y
               + 9 + 9 + 9                        # currency_dir, coord_dir, partner_dir
               + 3 + 3 + 3                        # currency_dist, coord_dist, partner_dist
               + (N_SIGNALS + 1) * MAX_MSG_LEN    # received message (4 token slots)
               + _N_ENERGY_BINS)                  # own energy level

# Observable partner features for Theory of Mind
_PARTNER_FEATURE_SIZE = 9 + 3 + (N_SIGNALS + 1) * MAX_MSG_LEN  # dir + dist + msg = 80

# visit_counts eviction: evict when dict exceeds this size; keep only entries
# with count <= threshold (entries above contribute < beta/10 curiosity bonus)
_VISIT_COUNT_MAX          = 50_000
_VISIT_COUNT_EVICT_THRESH = 100

# Reputation system — per-partner trust scores
_REP_INIT        = 0.5    # starting reputation for an unknown partner
_REP_BOOST       = 0.05   # increase per episode when coordination achieved
_REP_PENALTY     = 0.01   # decrease per episode when no coordination
_REP_BONUS_SCALE = 0.5    # max extra reward per step when partner rep = 1.0

# ── Hyperparameter evolution bounds ───────────────────────────────────────────

_HP_BOUNDS = {
    'learning_rate':  (1e-4,  1e-2),
    'epsilon_decay':  (0.990, 0.9999),
    'intrinsic_beta': (0.0,   5.0),
    'episodic_beta':  (0.0,   5.0),
    'hidden_size':    (64,    512),
    'n_layers':       (1.0,   4.0),
    'use_skip':       (0.0,   1.0),
    'drain_rate':     (ENERGY_DRAIN_LOW, ENERGY_DRAIN_HIGH),  # evolvable energy cost per step
}

# ── Activation registry ────────────────────────────────────────────────────────

_ACTS = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}


# ── Networks ───────────────────────────────────────────────────────────────────

class _QNet(nn.Module):
    """DQN with evolvable depth, activation, and optional skip connections."""

    def __init__(self, hidden_size, n_actions, n_layers=2,
                 activation='relu', use_skip=False):
        super().__init__()
        act_cls       = _ACTS.get(activation, nn.ReLU)
        self.use_skip = use_skip and n_layers >= 2
        self.act      = act_cls()

        self.input_layer = nn.Linear(_INPUT_SIZE, hidden_size)
        self.hidden      = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(max(0, n_layers - 1))]
        )
        self.output      = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        h = self.act(self.input_layer(x))
        for layer in self.hidden:
            h_new = self.act(layer(h))
            h = h + h_new if self.use_skip else h_new
        return self.output(h)


class _WorldModel(nn.Module):
    """Predicts (next_state_vec, reward) from (state_vec, action_onehot)."""

    def __init__(self, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(_INPUT_SIZE + n_actions, WORLD_MODEL_HIDDEN),
            nn.ReLU(),
            nn.Linear(WORLD_MODEL_HIDDEN, WORLD_MODEL_HIDDEN),
            nn.ReLU(),
        )
        self.state_head  = nn.Linear(WORLD_MODEL_HIDDEN, _INPUT_SIZE)
        self.reward_head = nn.Linear(WORLD_MODEL_HIDDEN, 1)

    def forward(self, sv, a_oh):
        h = self.trunk(torch.cat([sv, a_oh], dim=-1))
        return self.state_head(h), self.reward_head(h).squeeze(-1)


class _PartnerModel(nn.Module):
    """Predicts coordination intent from observable partner features (logit output)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_PARTNER_FEATURE_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Prioritised Experience Replay ──────────────────────────────────────────────

class _SumTree:
    """Binary sum-tree for O(log n) priority sampling."""

    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = np.empty(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Scalar propagation used only during single-entry add()."""
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx, s):
        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        change         = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change         = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def batch_update(self, indices, priorities):
        """
        Vectorised batch update: set leaf priorities then propagate
        level-by-level using numpy slicing instead of per-entry Python loops.
        Reduces priority-update cost from O(n * log(cap)) Python ops to
        O(log(cap)) numpy ops.
        """
        idx_arr = np.asarray(indices, dtype=np.int64)
        pri_arr = np.asarray(priorities, dtype=np.float64)
        self.tree[idx_arr] = pri_arr

        current = idx_arr
        while len(current) > 0:
            # Move up one level, only from nodes that have a parent
            with_parent  = current[current > 0]
            if len(with_parent) == 0:
                break
            parent_idxs  = np.unique((with_parent - 1) // 2)
            left         = 2 * parent_idxs + 1
            right        = np.minimum(left + 1, len(self.tree) - 1)
            self.tree[parent_idxs] = self.tree[left] + self.tree[right]
            current = parent_idxs

    def get(self, s):
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class _PrioritizedReplayBuffer:
    """Shared, prioritised experience replay for all agents."""

    def __init__(self, capacity, alpha=PER_ALPHA):
        self.alpha         = alpha
        self.tree          = _SumTree(capacity)
        self._max_priority = 1.0

    def push(self, s, a, r, ns):
        if len(s) != _INPUT_SIZE or len(ns) != _INPUT_SIZE:
            return  # reject state vectors from a different input-size era
        self.tree.add(self._max_priority, (s, a, r, ns))

    def sample(self, n, beta):
        n = min(n, len(self))
        if n == 0:
            return None

        segment  = self.tree.total() / n
        indices, priorities, batch = [], [], []

        for i in range(n):
            lo = segment * i
            hi = segment * (i + 1)
            s  = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                continue
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        if not batch:
            return None

        probs      = np.array(priorities, dtype=np.float64) / (self.tree.total() + 1e-12)
        is_weights = (len(self) * probs) ** (-beta)
        is_weights = (is_weights / is_weights.max()).astype(np.float32)

        s_b, a_b, r_b, ns_b = zip(*batch)
        return (np.array(s_b,  dtype=np.float32),
                np.array(a_b,  dtype=np.int64),
                np.array(r_b,  dtype=np.float32),
                np.array(ns_b, dtype=np.float32),
                is_weights,
                indices)

    def update_priorities(self, indices, errors):
        """Vectorised: compute all priorities at once then batch-update the tree."""
        priorities = (np.abs(np.asarray(errors, dtype=np.float64)) + 1e-6) ** self.alpha
        self._max_priority = max(self._max_priority, float(priorities.max()))
        self.tree.batch_update(indices, priorities)

    def __len__(self):
        return self.tree.n_entries


def make_replay_buffer():
    """Factory: creates the shared PER buffer used by all agents."""
    return _PrioritizedReplayBuffer(SHARED_REPLAY_SIZE)


# ── State encoding ─────────────────────────────────────────────────────────────

def _energy_bin(energy):
    """Map energy to one of 5 bins. Bin 4 = at or above reproduction threshold."""
    return min(int(energy * _N_ENERGY_BINS / ENERGY_MAX), _N_ENERGY_BINS - 1)


def _encode(state, energy=ENERGY_START):
    x, y, cd, kd, pd       = state[0], state[1], state[2], state[3], state[4]
    c_dist, k_dist, p_dist = state[5], state[6], state[7]
    msg = state[8:8 + MAX_MSG_LEN]
    vec = np.zeros(_INPUT_SIZE, dtype=np.float32)
    off = 0
    vec[off + x]      = 1.0; off += GRID_SIZE
    vec[off + y]      = 1.0; off += GRID_SIZE
    vec[off + cd]     = 1.0; off += 9
    vec[off + kd]     = 1.0; off += 9
    vec[off + pd]     = 1.0; off += 9
    vec[off + c_dist] = 1.0; off += 3
    vec[off + k_dist] = 1.0; off += 3
    vec[off + p_dist] = 1.0; off += 3
    for tok in msg:
        vec[off + tok] = 1.0; off += N_SIGNALS + 1
    vec[off + _energy_bin(energy)] = 1.0; off += _N_ENERGY_BINS
    return vec


def _potential(state):
    """Φ(s): proximity bonus for coord and currency nodes. Higher = closer."""
    coord_dist    = min(state[6], 2)
    currency_dist = min(state[5], 2)
    return (POTENTIAL_COORD_WEIGHT    * (2 - coord_dist) +
            POTENTIAL_CURRENCY_WEIGHT * (2 - currency_dist))


# ── Cultural inheritance ───────────────────────────────────────────────────────

class EpisodicMemory:
    """Stores high-value transitions for cultural transmission between generations."""

    def __init__(self, max_size=MEMORY_SIZE):
        self.max_size    = max_size
        self.experiences = []

    def store(self, sv, action, reward, nsv):
        if reward >= MEMORY_REWARD_THRESH:
            self.experiences.append((sv.copy(), int(action), float(reward), nsv.copy()))
            if len(self.experiences) > self.max_size * 2:
                self.experiences.sort(key=lambda e: e[2], reverse=True)
                self.experiences = self.experiences[:self.max_size]

    def seed_replay(self, replay, expected_size=None):
        for exp in self.experiences:
            if expected_size is not None and len(exp[0]) != expected_size:
                continue
            replay.push(*exp)

    def inherit_from(self, parent, keep_ratio=0.7):
        n_p = int(len(parent.experiences) * keep_ratio)
        n_s = max(0, self.max_size - n_p)
        p_b = sorted(parent.experiences, key=lambda e: e[2], reverse=True)[:n_p]
        s_b = sorted(self.experiences,   key=lambda e: e[2], reverse=True)[:n_s]
        self.experiences = p_b + s_b

    def to_list(self):
        return [(s.tolist(), a, r, ns.tolist()) for s, a, r, ns in self.experiences]

    @classmethod
    def from_list(cls, data):
        mem = cls()
        mem.experiences = [
            (np.array(s, dtype=np.float32), a, r, np.array(ns, dtype=np.float32))
            for s, a, r, ns in data
        ]
        return mem

    def __len__(self):
        return len(self.experiences)


# ── Goal discovery ─────────────────────────────────────────────────────────────

# State layout: s[2]=currency_dir, s[3]=coord_dir, s[4]=partner_dir (8=absent)
#               s[5]=currency_dist_bin, s[6]=coord_dist_bin, s[7]=partner_dist_bin
_SPEC_CONDITIONS = {
    'coord_close':    lambda s: s[6] == 0,
    'coord_near':     lambda s: s[6] <= 1,
    'currency_close': lambda s: s[5] == 0,
    'partner_close':  lambda s: s[7] == 0,
    'avoid_partner':  lambda s: s[7] == 2,
    'face_coord':     lambda s: s[3] != 8,
    'face_partner':   lambda s: s[4] != 8,
    'face_currency':  lambda s: s[2] != 8,
}
_TRACKABLE_CONDITIONS = list(_SPEC_CONDITIONS.items())


class GoalDiscovery:
    """
    Tracks which state features predict high reward and crystallises the
    strongest correlation into a sub-goal spec.
    """

    def __init__(self):
        self._last_crystallise_ep = 0
        self.stats = {
            name: {'met': 0, 'met_r': 0.0, 'unmet': 0, 'unmet_r': 0.0}
            for name, _ in _TRACKABLE_CONDITIONS
        }
        self.history = []

    def record(self, state, reward):
        for name, pred in _TRACKABLE_CONDITIONS:
            st = self.stats[name]
            if pred(state):
                st['met']   += 1
                st['met_r'] += reward
            else:
                st['unmet']   += 1
                st['unmet_r'] += reward

    def maybe_crystallise(self, episode):
        if episode - self._last_crystallise_ep < GOAL_DISCOVERY_CRYSTALLISE_EVERY:
            return None
        self._last_crystallise_ep = episode

        candidates = []
        for name, _ in _TRACKABLE_CONDITIONS:
            st = self.stats[name]
            if st['met'] < GOAL_DISCOVERY_MIN_SAMPLES:
                continue
            if st['unmet'] < GOAL_DISCOVERY_MIN_SAMPLES:
                continue
            avg_met   = st['met_r']   / st['met']
            avg_unmet = st['unmet_r'] / st['unmet']
            lift = avg_met - avg_unmet
            if lift >= GOAL_DISCOVERY_MIN_LIFT:
                candidates.append((lift, name))

        if not candidates:
            return None

        lift, best = max(candidates)
        spec = {'condition': best, 'bonus': GOAL_DISCOVERY_BONUS, 'negate': False}
        self.history.append((episode, spec, round(lift, 3)))
        return spec, round(lift, 3)

    def summary(self):
        rows = []
        for name, _ in _TRACKABLE_CONDITIONS:
            st = self.stats[name]
            if st['met'] >= 10 and st['unmet'] >= 10:
                avg_met   = st['met_r']   / st['met']
                avg_unmet = st['unmet_r'] / st['unmet']
                rows.append((name, round(avg_met - avg_unmet, 3)))
        return sorted(rows, key=lambda x: -x[1])


# ── Sub-goal templates ─────────────────────────────────────────────────────────

def _sg_face_coord(_):
    return lambda s: 0.3 if s[3] != 8 else 0.0

def _sg_partner_contact(_):
    return lambda s: 0.3 if s[4] != 8 else 0.0

def _sg_currency_direction(_):
    return lambda s: 0.2 if s[2] != 8 else 0.0

def _sg_coord_nearby(_):
    return lambda s: 0.4 if s[6] == 0 else 0.0

def _sg_partner_close(_):
    return lambda s: 0.3 if s[7] == 0 else 0.0

def _sg_from_spec(params):
    condition = params.get('condition', 'face_coord')
    bonus     = float(params.get('bonus', 0.3))
    negate    = bool(params.get('negate', False))
    pred      = _SPEC_CONDITIONS.get(condition, _SPEC_CONDITIONS['face_coord'])
    def fn(s):
        met = pred(s)
        if negate:
            met = not met
        return bonus if met else 0.0
    return fn

SUB_GOAL_TEMPLATES = {
    'face_coord':       _sg_face_coord,
    'partner_contact':  _sg_partner_contact,
    'currency_dir':     _sg_currency_direction,
    'coord_nearby':     _sg_coord_nearby,
    'partner_close':    _sg_partner_close,
    'spec':             _sg_from_spec,
}


class SubGoal:
    """A reward bonus active for SUB_GOAL_DURATION steps, generated by Claude."""

    def __init__(self, template='face_coord', params=None,
                 duration=SUB_GOAL_DURATION):
        params            = params or {}
        factory           = SUB_GOAL_TEMPLATES.get(template, _sg_face_coord)
        self.template     = template
        self.params       = params
        self.duration     = duration
        self._fn          = factory(params)
        self.active_steps = 0
        self.total_bonus  = 0.0

    def bonus(self, state):
        if self.active_steps >= self.duration:
            return 0.0
        b = min(float(self._fn(state)), SUB_GOAL_MAX_BONUS)
        self.active_steps += 1
        self.total_bonus  += b
        return b

    @property
    def is_active(self):
        return self.active_steps < self.duration

    @property
    def worth_inheriting(self):
        return self.total_bonus > 1.0

    def to_dict(self):
        return {'template': self.template, 'params': self.params,
                'duration': self.duration, 'active_steps': self.active_steps,
                'total_bonus': round(self.total_bonus, 4)}

    @classmethod
    def from_dict(cls, d):
        sg = cls(d['template'], d.get('params', {}), d['duration'])
        sg.active_steps = d.get('active_steps', 0)
        sg.total_bonus  = d.get('total_bonus', 0.0)
        return sg


# ── Agent ──────────────────────────────────────────────────────────────────────

class ARIAAgent:
    def __init__(self, agent_id,
                 learning_rate=None, epsilon_decay=None,
                 intrinsic_beta=None, episodic_beta=None,
                 hidden_size=None, n_layers=None,
                 activation=None, use_skip=None,
                 drain_rate=None,
                 energy=None,
                 online_state_dict=None,
                 cultural_memory=None,
                 sub_goal=None,
                 replay=None):

        self.agent_id  = agent_id
        self.epsilon   = EPSILON_START
        self.n_actions = 8 + N_SIGNALS

        # Evolvable hyperparameters
        self.learning_rate  = float(learning_rate)  if learning_rate  is not None else LEARNING_RATE
        self.epsilon_decay  = float(epsilon_decay)  if epsilon_decay  is not None else EPSILON_DECAY
        self.intrinsic_beta = float(intrinsic_beta) if intrinsic_beta is not None else INTRINSIC_BETA
        self.episodic_beta  = float(episodic_beta)  if episodic_beta  is not None else EPISODIC_BETA
        self.hidden_size    = int(round(float(hidden_size))) if hidden_size is not None else HIDDEN_SIZE
        self.n_layers       = max(1, min(4, int(round(float(n_layers))))) if n_layers is not None else N_LAYERS_DEFAULT
        self.activation     = activation if activation in _ACTS else 'relu'
        self.use_skip       = bool(round(float(use_skip))) if use_skip is not None else False
        self.drain_rate     = float(np.clip(float(drain_rate), ENERGY_DRAIN_LOW, ENERGY_DRAIN_HIGH)) if drain_rate is not None else random.choice([ENERGY_DRAIN_LOW, ENERGY_DRAIN_MED, ENERGY_DRAIN_HIGH])

        # Energy — persists across episodes (not reset between episodes)
        self.energy = float(energy) if energy is not None else ENERGY_START

        # Q-networks
        self.online_net = _QNet(self.hidden_size, self.n_actions,
                                self.n_layers, self.activation, self.use_skip)
        self.target_net = _QNet(self.hidden_size, self.n_actions,
                                self.n_layers, self.activation, self.use_skip)
        if online_state_dict is not None:
            try:
                self.online_net.load_state_dict(online_state_dict)
            except Exception:
                pass
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

        # World model
        self.world_model  = _WorldModel(self.n_actions)
        self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=WORLD_MODEL_LR)
        self._wm_steps    = 0

        # Shared PER replay (or per-agent fallback)
        self.replay  = replay if replay is not None else _PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE)
        self._steps  = 0
        self.visit_counts = {}
        self._ep_visited  = set()

        # Cultural memory
        self.cultural_memory = cultural_memory if cultural_memory is not None else EpisodicMemory()
        self.cultural_memory.seed_replay(self.replay, expected_size=_INPUT_SIZE)

        # Sub-goal
        self.sub_goal = sub_goal

        # Internal goal discovery
        self.goal_discovery = GoalDiscovery()

        # Theory of Mind
        self.tom_model     = _PartnerModel()
        self.tom_optimizer = optim.Adam(self.tom_model.parameters(), lr=TOM_LR)
        self.tom_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([TOM_POS_WEIGHT])
        )
        self._tom_steps = 0   # training samples seen; bonus withheld until warmed up

        # Reputation — per-partner trust scores {agent_id: float in [0,1]}
        self.reputation = {}

        # Agent-initiated replication
        self._repl_window     = deque(maxlen=SELF_REPL_WINDOW)
        self._repl_request_ep = -SELF_REPL_COOLDOWN

        # Specialisation
        self.coord_reward_total    = 0.0
        self.currency_reward_total = 0.0

        self.episode_reward = 0.0
        self.total_reward   = 0.0
        self.episodes       = 0
        self.generation     = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            sv = torch.tensor(_encode(state, self.energy)).unsqueeze(0)
            return int(self.online_net(sv).argmax().item())

    def _encode_partner_features(self, state):
        """Extract observable partner features for the ToM model."""
        pd, p_dist = state[4], state[7]
        msg = state[8:8 + MAX_MSG_LEN]
        vec        = np.zeros(_PARTNER_FEATURE_SIZE, dtype=np.float32)
        off        = 0
        vec[off + pd]     = 1.0; off += 9
        vec[off + p_dist] = 1.0; off += 3
        for tok in msg:
            vec[off + tok] = 1.0; off += N_SIGNALS + 1
        return vec

    def _per_beta(self):
        frac = min(1.0, self._steps / max(1, PER_BETA_STEPS))
        return PER_BETA_START + frac * (1.0 - PER_BETA_START)

    def update(self, state, action, reward, next_state, coord_achieved=None,
               pre_energy=None):
        # ── Curiosity ──────────────────────────────────────────────────────────
        vc = self.visit_counts.get(state, 0) + 1
        self.visit_counts[state] = vc
        global_cur = self.intrinsic_beta / np.sqrt(vc)
        ep_cur     = self.episodic_beta if state not in self._ep_visited else 0.0
        self._ep_visited.add(state)


        # ── Sub-goal bonus ─────────────────────────────────────────────────────
        sg_bonus = self.sub_goal.bonus(state) if (self.sub_goal and self.sub_goal.is_active) else 0.0

        # ── Theory of Mind: encode once, single forward pass ──────────────────
        # The same logit is detached for the intent bonus and differentiated for training.
        tom_bonus = 0.0
        if state[4] != 8:   # partner visible
            pf    = self._encode_partner_features(state)
            pf_t  = torch.tensor(pf).unsqueeze(0)
            logit = self.tom_model(pf_t)              # one forward pass

            # Bonus from pre-update prediction (consistent with the policy that acted)
            prob = torch.sigmoid(logit.detach()).item()
            if self._tom_steps >= WORLD_MODEL_WARMUP and prob > TOM_INTENT_THRESHOLD:
                tom_bonus = TOM_BONUS_SCALE * prob

            # Train on the same logit computation graph
            if coord_achieved is not None:
                target = torch.tensor([1.0 if coord_achieved else 0.0])
                loss   = self.tom_criterion(logit, target)
                self.tom_optimizer.zero_grad()
                loss.backward()
                self.tom_optimizer.step()
                self._tom_steps += 1

        # ── Potential-based dense reward shaping ───────────────────────────────
        shaping = DISCOUNT_FACTOR * _potential(next_state) - _potential(state)

        # ── Reputation bonus: extra reward when coordinating with trusted partners
        rep_bonus = 0.0
        if coord_achieved and self.reputation:
            rep_bonus = np.mean(list(self.reputation.values())) * _REP_BONUS_SCALE

        augmented = reward + shaping + global_cur + ep_cur + sg_bonus + tom_bonus + rep_bonus

        sv  = _encode(state,      pre_energy if pre_energy is not None else self.energy)
        nsv = _encode(next_state, self.energy)
        self.replay.push(sv, action, augmented, nsv)
        self._steps += 1

        # ── Training ───────────────────────────────────────────────────────────
        ready = (len(self.replay) >= MIN_REPLAY_SIZE and len(self.replay) >= BATCH_SIZE)
        if self._steps % TRAIN_FREQ == 0 and ready:
            batch = self.replay.sample(BATCH_SIZE, self._per_beta())
            if batch is not None:
                self._train_cycle(batch)

        if self._steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # ── Cultural memory ────────────────────────────────────────────────────
        self.cultural_memory.store(sv, action, augmented, nsv)

        # Goal discovery — raw env reward to avoid bonus feedback loops
        self.goal_discovery.record(state, reward)

        # Specialisation tracking
        if reward >= REWARD_COORD * 0.8:
            self.coord_reward_total += reward
        elif reward >= REWARD_CURRENCY * 0.8:
            self.currency_reward_total += reward

        self.episode_reward += reward
        self.total_reward   += reward

    def _train_cycle(self, batch):
        """
        One training cycle on a pre-sampled batch.
        Order: Q-learning → world model → Dyna-Q (all share the same tensors).
        Reduces replay.sample() calls from 5 to 1 per cycle.
        Dyna-Q uses random imagined actions on real states for diversity.
        """
        s, a, r, ns, is_weights, indices = batch

        # Zero-copy: torch.from_numpy shares memory with numpy arrays
        s_t  = torch.from_numpy(s)
        a_t  = torch.from_numpy(a)
        r_t  = torch.from_numpy(r)
        ns_t = torch.from_numpy(ns)
        w_t  = torch.from_numpy(is_weights)

        # ── Double DQN with IS-weighted loss ───────────────────────────────────
        cur_q = self.online_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_a   = self.online_net(ns_t).argmax(1, keepdim=True)
            nxt_q    = self.target_net(ns_t).gather(1, best_a).squeeze(1)
            target_q = r_t + DISCOUNT_FACTOR * nxt_q

        td_errors = (cur_q - target_q).detach().cpu().numpy()
        self.replay.update_priorities(indices, td_errors)

        loss = (w_t * F.smooth_l1_loss(cur_q, target_q, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # ── World model (reuses same batch tensors) ────────────────────────────
        a_oh = torch.zeros(len(a_t), self.n_actions)
        a_oh.scatter_(1, a_t.unsqueeze(1), 1.0)
        ns_pred, r_pred = self.world_model(s_t, a_oh)
        wm_loss = nn.MSELoss()(ns_pred, ns_t) + nn.MSELoss()(r_pred, r_t)
        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        self.wm_optimizer.step()
        self._wm_steps += 1

        # ── Dyna-Q: imagined rollouts from real states, random actions ─────────
        # Random actions give different imagined transitions each iteration
        # without an extra buffer sample.
        if self._wm_steps >= WORLD_MODEL_WARMUP:
            for _ in range(DYNA_STEPS):
                a_img    = torch.randint(0, self.n_actions, (len(s_t),))
                a_oh_img = torch.zeros(len(s_t), self.n_actions)
                a_oh_img.scatter_(1, a_img.unsqueeze(1), 1.0)
                with torch.no_grad():
                    ns_img, r_img = self.world_model(s_t, a_oh_img)
                cur_q_img = self.online_net(s_t).gather(1, a_img.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    nxt_q_img = self.target_net(ns_img).max(1)[0]
                    tgt_img   = r_img + DISCOUNT_FACTOR * nxt_q_img
                dyna_loss = nn.SmoothL1Loss()(cur_q_img, tgt_img)
                self.optimizer.zero_grad()
                dyna_loss.backward()
                nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
                self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        ep = self.episode_reward
        self.episode_reward = 0.0
        self.episodes += 1
        self._ep_visited.clear()
        if len(self.visit_counts) > _VISIT_COUNT_MAX:
            self.visit_counts = {
                s: c for s, c in self.visit_counts.items()
                if c <= _VISIT_COUNT_EVICT_THRESH
            }
        self._repl_window.append(ep)
        self._try_discover_goal()
        return ep

    def update_reputation(self, partner_ids, coord_this_ep):
        """Update trust scores for each partner after an episode."""
        for pid in partner_ids:
            current = self.reputation.get(pid, _REP_INIT)
            if coord_this_ep:
                current = min(1.0, current + _REP_BOOST)
            else:
                current = max(0.0, current - _REP_PENALTY)
            self.reputation[pid] = round(current, 4)

    def _try_discover_goal(self):
        if self.sub_goal and self.sub_goal.is_active:
            return
        result = self.goal_discovery.maybe_crystallise(self.episodes)
        if result is None:
            return
        spec, lift = result
        self.sub_goal = SubGoal('spec', spec)
        self._log_discovered_goal(spec, lift)

    def _log_discovered_goal(self, spec, lift):
        os.makedirs(os.path.dirname(GOAL_DISCOVERY_LOG_PATH), exist_ok=True)
        record = {
            'event':     'goal_discovered',
            'agent':     self.agent_id,
            'episode':   self.episodes,
            'condition': spec['condition'],
            'bonus':     spec['bonus'],
            'lift':      lift,
            'timestamp': datetime.now().isoformat(),
        }
        with open(GOAL_DISCOVERY_LOG_PATH, 'a') as f:
            f.write(json.dumps(record) + '\n')
        print(f'\n  [Goal] {self.agent_id} discovered: {spec["condition"]} '
              f'(lift={lift:.2f})')

    def assess_replication_readiness(self, episode, population_rewards):
        """
        Return True if this agent should self-request replication.
        Criteria: window is full, recent mean exceeds population mean by
        SELF_REPL_FITNESS_MIN, and cooldown has elapsed.
        population_rewards: list of recent-episode mean rewards for all agents.
        """
        if len(self._repl_window) < SELF_REPL_WINDOW:
            return False
        if episode - self._repl_request_ep < SELF_REPL_COOLDOWN:
            return False
        if not population_rewards:
            return False
        own_mean = np.mean(list(self._repl_window))
        pop_mean = np.mean(population_rewards)
        return own_mean >= SELF_REPL_FITNESS_MIN * pop_mean

    def record_replication_request(self, episode):
        self._repl_request_ep = episode

    def rebuild_optimizer(self):
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

    @property
    def goal_label(self):
        if self.sub_goal is None or not self.sub_goal.is_active:
            return 'none'
        if self.sub_goal.template == 'spec':
            return self.sub_goal.params.get('condition', 'spec')
        return self.sub_goal.template

    @property
    def role(self):
        total = self.coord_reward_total + self.currency_reward_total
        if total < 20:
            return 'exploring'
        ratio = self.coord_reward_total / total
        if ratio > 0.65:
            return 'coordinator'
        if ratio < 0.35:
            return 'forager'
        return 'generalist'

    @staticmethod
    def get_signal_from_action(action):
        return action - 8 if action >= 8 else None

    @staticmethod
    def _cross_hyperparam(val_a, val_b, w_a, lo, hi):
        val = val_a if random.random() < w_a else val_b
        val += random.gauss(0, HYPERPARAM_MUTATION_STD * max(abs(val), 1e-6))
        return float(np.clip(val, lo, hi))

    @classmethod
    def merge(cls, agent_a, agent_b, child_id, shared_replay=None):
        """
        Create a child via network crossover + full hyperparameter evolution.
        Cultural memory and sub-goals inherited from stronger parent.
        """
        min_r = min(agent_a.total_reward, agent_b.total_reward)
        a_s   = agent_a.total_reward - min_r
        b_s   = agent_b.total_reward - min_r
        total = a_s + b_s
        w_a   = (a_s / total) if total > 0 else 0.5
        w_b   = 1.0 - w_a

        hp = {name: cls._cross_hyperparam(
                getattr(agent_a, name), getattr(agent_b, name), w_a, lo, hi)
              for name, (lo, hi) in _HP_BOUNDS.items()}
        hp['activation'] = agent_a.activation if random.random() < w_a else agent_b.activation

        child = cls(child_id, replay=shared_replay, energy=ENERGY_NEWBORN, **hp)
        child.epsilon = min(agent_a.epsilon, agent_b.epsilon)

        archs_match = (agent_a.hidden_size == agent_b.hidden_size == child.hidden_size and
                       agent_a.n_layers    == agent_b.n_layers    == child.n_layers and
                       agent_a.activation  == agent_b.activation  == child.activation)
        if archs_match:
            with torch.no_grad():
                for pa, pb, pc in zip(agent_a.online_net.parameters(),
                                      agent_b.online_net.parameters(),
                                      child.online_net.parameters()):
                    mask    = torch.rand_like(pa) < w_a
                    crossed = torch.where(mask, pa, pb)
                    noise   = torch.randn_like(crossed) * MUTATION_STD
                    crossed = crossed + noise * (crossed != 0).float()
                    pc.copy_(crossed)
        else:
            stronger_net = agent_a.online_net if w_a >= 0.5 else agent_b.online_net
            try:
                child.online_net.load_state_dict(stronger_net.state_dict())
            except Exception:
                pass
        child.target_net.load_state_dict(child.online_net.state_dict())

        stronger = agent_a if w_a >= 0.5 else agent_b
        weaker   = agent_b if stronger is agent_a else agent_a
        child.cultural_memory.inherit_from(stronger.cultural_memory)
        child.cultural_memory.seed_replay(child.replay, expected_size=_INPUT_SIZE)

        for parent in (stronger, weaker):
            if parent.sub_goal and parent.sub_goal.worth_inheriting and child.sub_goal is None:
                child.sub_goal = SubGoal(
                    parent.sub_goal.template,
                    parent.sub_goal.params,
                    parent.sub_goal.duration
                )
                break

        # Inherit reputation from stronger parent
        child.reputation = dict(stronger.reputation)

        return child, round(w_a, 4), round(w_b, 4)
