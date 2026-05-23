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
    MAX_POPULATION, SELF_REPL_FITNESS_MIN,
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
        half   = PLATEAU_WINDOW // 2
        h_list = list(h)
        older  = sum(h_list[:half]) / half
        recent = sum(h_list[half:]) / half
        return (recent - older) < PLATEAU_DELTA_THRESH

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

    def check_agent_initiated(self, agents, episode, last_replication_ep):
        """
        Returns (trigger: bool, reason: str) when a high-fitness agent requests
        self-replication. Checks each agent's own assessment against the
        population mean of their recent-window rewards.
        """
        since_last = episode - last_replication_ep
        if since_last < MIN_REPLICATION_INTERVAL:
            return False, ''

        pop_means = []
        for a in agents.values():
            w = list(a._repl_window)
            if w:
                pop_means.append(float(sum(w) / len(w)))

        for agent_id, agent in agents.items():
            if agent.assess_replication_readiness(episode, pop_means):
                return True, f'{agent_id} self-requested'

        return False, ''


def next_agent_id(parent_a_id, parent_b_id, all_ids_ever):
    """Generate child ID as hex OR of parent values, with collision fallback."""
    a_val = int(parent_a_id.split('-')[1], 16)
    b_val = int(parent_b_id.split('-')[1], 16)

    candidate = f'ARIA-{(a_val | b_val):04X}'
    if candidate not in all_ids_ever:
        return candidate

    n = max(a_val, b_val) + 1
    while n <= 0xFFFF:
        candidate = f'ARIA-{n:04X}'
        if candidate not in all_ids_ever:
            return candidate
        n += 1
    raise RuntimeError('Exhausted all 4-digit hex agent IDs')


def kill_weakest(agents, episode, all_ids_ever):
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


def reproduce_pair(parent_a, parent_b, agents, channel, episode,
                   all_ids_ever, shared_replay=None):
    """
    Create a child from two agents who met on the grid for REPRODUCTION_STEPS
    consecutive steps. Returns (new_agents, new_channel, summary).
    """
    child_id        = next_agent_id(parent_a.agent_id, parent_b.agent_id, all_ids_ever)
    child, w_a, w_b = ARIAAgent.merge(parent_a, parent_b, child_id,
                                      shared_replay=shared_replay)

    child_channel = CommunicationChannel(append_log=True)
    child_channel.inherit_from(channel)

    summary = {
        'retired_id':             '',
        'surviving_id':           parent_a.agent_id,
        'child_id':               child_id,
        'episode':                episode,
        'weights':                {parent_a.agent_id: w_a, parent_b.agent_id: w_b},
        'retired_total_reward':   0,
        'surviving_total_reward': round(parent_a.total_reward, 2),
        'child_hyperparams': {
            'lr':         round(child.learning_rate,  6),
            'eps_d':      round(child.epsilon_decay,  4),
            'β_g':        round(child.intrinsic_beta, 4),
            'β_e':        round(child.episodic_beta,  4),
            'h':          int(child.hidden_size),
            'n_layers':   int(child.n_layers),
            'activation': child.activation,
            'use_skip':   bool(child.use_skip),
        }
    }

    _log_replication(summary)

    new_agents = dict(agents)
    new_agents[child_id] = child
    return new_agents, child_channel, summary


def replicate(agents, channel, episode, all_ids_ever, shared_replay=None):
    """
    Select top 2 agents as parents, retire the worst, create a child.
    Maintains MAX_POPULATION agents. Returns new_agents, new_channel, summary.
    """
    agent_list = sorted(agents.values(), key=lambda a: a.total_reward, reverse=True)

    parent_a = agent_list[0]
    parent_b = agent_list[1]
    retired  = agent_list[-1]

    child_id         = next_agent_id(parent_a.agent_id, parent_b.agent_id, all_ids_ever)
    child, w_a, w_b  = ARIAAgent.merge(parent_a, parent_b, child_id, shared_replay=shared_replay)

    child_channel = CommunicationChannel(append_log=True)
    child_channel.inherit_from(channel)

    summary = {
        'retired_id':             retired.agent_id,
        'surviving_id':           parent_a.agent_id,
        'child_id':               child_id,
        'episode':                episode,
        'weights':                {parent_a.agent_id: w_a, parent_b.agent_id: w_b},
        'retired_total_reward':   round(retired.total_reward, 2),
        'surviving_total_reward': round(parent_a.total_reward, 2),
        'child_hyperparams': {
            'lr':         round(child.learning_rate,  6),
            'eps_d':      round(child.epsilon_decay,  4),
            'β_g':        round(child.intrinsic_beta, 4),
            'β_e':        round(child.episodic_beta,  4),
            'h':          int(child.hidden_size),
            'n_layers':   int(child.n_layers),
            'activation': child.activation,
            'use_skip':   bool(child.use_skip),
        }
    }

    _save_retired(retired, summary)
    _log_replication(summary)

    # Survivors = all non-retired agents + new child
    new_agents = {a.agent_id: a
                  for a in agent_list if a.agent_id != retired.agent_id}
    new_agents[child_id] = child

    # Safety cap: if somehow over MAX_POPULATION, drop the lowest non-parent
    while len(new_agents) > MAX_POPULATION:
        protected = {parent_a.agent_id, parent_b.agent_id, child_id}
        drop = min(
            (a for a in new_agents.values() if a.agent_id not in protected),
            key=lambda a: a.total_reward
        )
        del new_agents[drop.agent_id]

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
