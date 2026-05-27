"""
ARIA-2 Genetics — Lewis Signaling Game
Population replication. Children are assigned a random type (0 or 1) to
maintain type diversity regardless of parent types. The Monitor culls the
weakest agent when population exceeds MIN_POPULATION.
"""

import os
import json
import random
from collections import deque
from datetime import datetime

from agent import ARIAAgent
from communication import CommunicationChannel
from config import (
    LEXICON_LOG_PATH, RETIRED_LOG_PATH,
    PLATEAU_WINDOW, PLATEAU_DELTA_THRESH,
    MIN_REPLICATION_INTERVAL, MAX_REPLICATION_INTERVAL,
    MAX_POPULATION,
)


class PlateauMonitor:
    def __init__(self):
        self.history = {}

    def register(self, agent_id):
        self.history[agent_id] = deque(maxlen=PLATEAU_WINDOW)

    def deregister(self, agent_id):
        self.history.pop(agent_id, None)

    def record(self, agent_id, ep_reward):
        if agent_id not in self.history:
            self.register(agent_id)
        self.history[agent_id].append(ep_reward)

    def is_plateauing(self, agent_id):
        h = self.history.get(agent_id, deque())
        if len(h) < PLATEAU_WINDOW:
            return False
        half     = PLATEAU_WINDOW // 2
        h_list   = list(h)
        older    = sum(h_list[:half]) / half
        recent   = sum(h_list[half:]) / half
        mean_abs = max(abs(sum(h_list) / len(h_list)), 1.0)
        return (recent - older) < PLATEAU_DELTA_THRESH * mean_abs

    def should_replicate(self, agents, episode, last_replication_ep):
        since_last = episode - last_replication_ep
        if since_last < MIN_REPLICATION_INTERVAL:
            return False, ''
        if since_last >= MAX_REPLICATION_INTERVAL:
            return True, 'max interval reached'
        for agent_id in agents:
            if self.is_plateauing(agent_id):
                return True, f'{agent_id} plateau'
        return False, ''


def get_alpha(agents):
    def fitness(a):
        return a.total_reward / a.episodes if a.episodes > 0 else 0.0
    return max(agents.values(), key=fitness).agent_id


def next_agent_id(parent_a_id, parent_b_id, all_ids_ever):
    a_val = int(parent_a_id.split('-')[1], 16)
    b_val = int(parent_b_id.split('-')[1], 16)
    candidate = f'ARIA-{(a_val | b_val):04X}'
    if candidate not in all_ids_ever:
        return candidate
    for n in range(1, 0x10000):
        candidate = f'ARIA-{n:04X}'
        if candidate not in all_ids_ever:
            return candidate
    raise RuntimeError('Exhausted all 4-digit hex agent IDs')


def select_weakest_per_type(agents, agent_types):
    """Return (weakest_T0, weakest_T1). Either may be None if that type has no agents."""
    by_type = {0: [], 1: []}
    for aid, agent in agents.items():
        by_type[agent_types.get(aid, 0)].append(agent)
    weakest_t0 = min(by_type[0], key=lambda a: a.total_reward) if by_type[0] else None
    weakest_t1 = min(by_type[1], key=lambda a: a.total_reward) if by_type[1] else None
    return weakest_t0, weakest_t1


def kill_agent(agents, episode, target):
    """Kill a specific agent by object reference."""
    summary = {
        'retired_id':           target.agent_id,
        'episode':              episode,
        'retired_total_reward': round(target.total_reward, 2),
        'reason':               'natural_death',
    }
    _save_retired(target, summary)
    _log_death(target, episode)
    new_agents = {a.agent_id: a for a in agents.values()
                  if a.agent_id != target.agent_id}
    return new_agents, summary


def cross_type_reproduce(parent_a, parent_b, agents, channel, episode, all_ids_ever, shared_replay=None):
    """Cross-type collision reproduction: always produces one T0 child and one T1 child."""
    t0_id = next_agent_id(parent_a.agent_id, parent_b.agent_id, all_ids_ever)
    t0, w_a0, w_b0 = ARIAAgent.merge(parent_a, parent_b, t0_id,
                                      child_type=0, shared_replay=shared_replay)

    extended_ids = list(all_ids_ever) + [t0_id]
    t1_id = next_agent_id(parent_a.agent_id, t0_id, extended_ids)
    t1, w_a1, w_b1 = ARIAAgent.merge(parent_a, parent_b, t1_id,
                                      child_type=1, shared_replay=shared_replay)

    child_channel = CommunicationChannel(append_log=True)
    child_channel.inherit_from(channel)

    children = [(t0_id, 0, t0), (t1_id, 1, t1)]
    for child_id, child_type, child in children:
        _log_replication({
            'child_id':   child_id,
            'child_type': child_type,
            'parent_a':   parent_a.agent_id,
            'parent_b':   parent_b.agent_id,
            'episode':    episode,
            'child_hyperparams': {
                'lr':         round(child.learning_rate, 6),
                'n_layers':   int(child.n_layers),
                'activation': child.activation,
                'drain_rate': round(child.drain_rate, 3),
            },
        })

    new_agents = dict(agents)
    new_agents[t0_id] = t0
    new_agents[t1_id] = t1
    return new_agents, child_channel, children


def monitor_replace(culled_type, agents, channel, episode, all_ids_ever, shared_replay=None):
    """Immediately replace a Monitor-culled agent with a new agent of the same type.
    Parents are the two highest-reward survivors regardless of their type."""
    sorted_agents = sorted(agents.values(), key=lambda a: a.total_reward, reverse=True)
    parent_a = sorted_agents[0]
    parent_b = sorted_agents[1] if len(sorted_agents) > 1 else sorted_agents[0]

    child_id        = next_agent_id(parent_a.agent_id, parent_b.agent_id, all_ids_ever)
    child, w_a, w_b = ARIAAgent.merge(parent_a, parent_b, child_id,
                                      child_type=culled_type,
                                      shared_replay=shared_replay)

    child_channel = CommunicationChannel(append_log=True)
    child_channel.inherit_from(channel)

    summary = {
        'child_id':   child_id,
        'child_type': culled_type,
        'parent_a':   parent_a.agent_id,
        'parent_b':   parent_b.agent_id,
        'episode':    episode,
        'weights':    {parent_a.agent_id: w_a, parent_b.agent_id: w_b},
        'child_hyperparams': {
            'lr':         round(child.learning_rate,  6),
            'eps_d':      round(child.epsilon_decay,  4),
            'β_g':        round(child.intrinsic_beta, 4),
            'β_e':        round(child.episodic_beta,  4),
            'h':          int(child.hidden_size),
            'n_layers':   int(child.n_layers),
            'activation': child.activation,
            'use_skip':   bool(child.use_skip),
            'drain_rate': round(child.drain_rate, 3),
        }
    }

    _log_replication(summary)
    new_agents = dict(agents)
    new_agents[child_id] = child
    return new_agents, child_channel, summary


def _save_retired(agent, summary):
    os.makedirs(RETIRED_LOG_PATH, exist_ok=True)
    path = os.path.join(RETIRED_LOG_PATH, f'{agent.agent_id}.json')
    record = {
        'agent_id':              agent.agent_id,
        'agent_type':            agent.agent_type,
        'total_reward':          round(agent.total_reward, 2),
        'episodes':              agent.episodes,
        'epsilon_at_retirement': round(agent.epsilon, 4),
        'role':                  agent.role,
        'hyperparams': {
            'learning_rate':  round(agent.learning_rate,  6),
            'epsilon_decay':  round(agent.epsilon_decay,  4),
            'intrinsic_beta': round(agent.intrinsic_beta, 4),
            'episodic_beta':  round(agent.episodic_beta,  4),
            'hidden_size':    int(agent.hidden_size),
            'n_layers':       int(agent.n_layers),
            'activation':     agent.activation,
            'use_skip':       bool(agent.use_skip),
        },
        'retirement_summary': summary,
        'timestamp':          datetime.now().isoformat()
    }
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)


def _log_death(agent, episode):
    record = {
        'event':        'death',
        'timestamp':    datetime.now().isoformat(),
        'agent_id':     agent.agent_id,
        'agent_type':   agent.agent_type,
        'episode':      episode,
        'total_reward': round(agent.total_reward, 2),
    }
    with open(LEXICON_LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')


def _log_replication(summary):
    record = {'event': 'replication', 'timestamp': datetime.now().isoformat()}
    record.update(summary)
    with open(LEXICON_LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')


def log_energy_death(agent, episode, step):
    record = {
        'event':        'energy_death',
        'timestamp':    datetime.now().isoformat(),
        'agent_id':     agent.agent_id,
        'agent_type':   agent.agent_type,
        'episode':      episode,
        'step':         step,
        'drain_rate':   round(float(agent.drain_rate), 3),
        'total_reward': round(float(agent.total_reward), 2),
        'episodes':     int(agent.episodes),
    }
    with open(LEXICON_LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')


def log_ghost_absorption(agent_id, episode, step, total_reward, energy):
    record = {
        'event':        'ghost_absorbed',
        'timestamp':    datetime.now().isoformat(),
        'agent_id':     agent_id,
        'episode':      episode,
        'step':         step,
        'total_reward': round(float(total_reward), 2),
        'energy':       round(float(energy), 1),
    }
    with open(LEXICON_LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')


def log_extinction(episode, generation, all_ids_ever, channel):
    record = {
        'event':         'extinction',
        'timestamp':     datetime.now().isoformat(),
        'episode':       episode,
        'generation':    generation,
        'lineage':       list(all_ids_ever),
        'total_signals': channel.total_signals,
        'lex_assigned':  channel.assigned_count(),
    }
    with open(LEXICON_LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
