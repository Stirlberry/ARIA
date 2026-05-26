"""
ARIA Genetics — Phase 2
N-agent population replication.
  - Sort all agents by total_reward
  - Top 2 are parents; worst is retired
  - Child merges the top 2 via ARIAAgent.merge()
  - Result maintains MAX_POPULATION agents
"""

import os
import json
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
    """
    Fires replication when any agent's reward stops improving over PLATEAU_WINDOW
    episodes, or when MAX_REPLICATION_INTERVAL episodes have elapsed.
    """

    def __init__(self):
        self.history = {}   # agent_id -> deque(maxlen=PLATEAU_WINDOW)

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
        """Returns (trigger: bool, reason: str)."""
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
    """Return the agent_id with the highest lifetime average reward per episode."""
    def fitness(a):
        return a.total_reward / a.episodes if a.episodes > 0 else 0.0
    return max(agents.values(), key=fitness).agent_id


def next_agent_id(parent_a_id, parent_b_id, all_ids_ever):
    """Generate child ID as hex OR of parent values, with full-range collision fallback."""
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


def kill_weakest(agents, episode):
    """
    Remove the weakest agent from the population (death phase).
    Returns (reduced_agents, death_summary). Population drops by 1.
    """
    dying = min(agents.values(), key=lambda a: a.total_reward)
    summary = {
        'retired_id':           dying.agent_id,
        'episode':              episode,
        'retired_total_reward': round(dying.total_reward, 2),
        'reason':               'natural_death',
    }
    _save_retired(dying, summary)
    _log_death(dying, episode)
    new_agents = {a.agent_id: a for a in agents.values()
                  if a.agent_id != dying.agent_id}
    return new_agents, summary


def energy_reproduce(parent_a, parent_b, agents, channel, episode,
                     all_ids_ever, shared_replay=None):
    """
    Two agents above the energy threshold who physically collide produce a child.
    Works at any population level. Energy cost already deducted by caller.
    Cultural memory is inherited via ARIAAgent.merge().
    """
    child_id        = next_agent_id(parent_a.agent_id, parent_b.agent_id, all_ids_ever)
    child, w_a, w_b = ARIAAgent.merge(parent_a, parent_b, child_id,
                                      shared_replay=shared_replay)

    child_channel = CommunicationChannel(append_log=True)
    child_channel.inherit_from(channel)

    summary = {
        'child_id':  child_id,
        'parent_a':  parent_a.agent_id,
        'parent_b':  parent_b.agent_id,
        'episode':   episode,
        'weights':   {parent_a.agent_id: w_a, parent_b.agent_id: w_b},
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
