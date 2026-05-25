"""
ARIA Save System — Phase 2
Persists full simulation state including Phase 2 fields:
  n_layers, activation, use_skip, world model weights,
  cultural memory, sub-goal, and compound lexicon.

Save format (per checkpoint):
  logs/saves/checkpoint_ep{N:06d}.pt   — network weights (PyTorch state dicts)
  logs/saves/checkpoint_ep{N:06d}.json — all other state
  logs/saves/latest.json               — pointer to most recent checkpoint
"""

import os
import json
import torch
from datetime import datetime
from config import (
    LEARNING_RATE, EPSILON_DECAY, INTRINSIC_BETA, EPISODIC_BETA,
    HIDDEN_SIZE, N_LAYERS_DEFAULT
)

SAVE_DIR = 'logs/saves'


def save_checkpoint(episode, agents, channel, generation,
                    last_replication_ep, all_ids_ever, plateau_mon):
    os.makedirs(SAVE_DIR, exist_ok=True)

    base      = f'checkpoint_ep{episode:06d}'
    pt_path   = os.path.join(SAVE_DIR, base + '.pt')
    json_path = os.path.join(SAVE_DIR, base + '.json')

    weights    = {}
    agent_meta = {}
    for agent_id, agent in agents.items():
        key = agent_id.replace('-', '_')
        weights[key]            = agent.online_net.state_dict()
        weights[f'{key}_wm']    = agent.world_model.state_dict()
        weights[f'{key}_tom']   = agent.tom_model.state_dict()
        agent_meta[agent_id] = {
            'epsilon':        round(float(agent.epsilon), 8),
            'total_reward':   round(float(agent.total_reward), 4),
            'episodes':       int(agent.episodes),
            'learning_rate':  round(float(agent.learning_rate), 8),
            'epsilon_decay':  round(float(agent.epsilon_decay), 8),
            'intrinsic_beta': round(float(agent.intrinsic_beta), 6),
            'episodic_beta':  round(float(agent.episodic_beta), 6),
            'hidden_size':    int(agent.hidden_size),
            'n_layers':       int(agent.n_layers),
            'activation':     agent.activation,
            'use_skip':       bool(agent.use_skip),
            'energy':          round(float(agent.energy),    4),
            'drain_rate':      round(float(agent.drain_rate), 4),
            'cultural_memory': agent.cultural_memory.to_list(),
            'sub_goal':       agent.sub_goal.to_dict() if agent.sub_goal else None,
            'tom_steps':       int(agent._tom_steps),
            'steps':           int(agent._steps),
            'wm_steps':        int(agent._wm_steps),
            'reputation':           dict(agent.reputation),
            'coord_reward_total':   round(float(agent.coord_reward_total), 4),
            'currency_reward_total': round(float(agent.currency_reward_total), 4),
            'goal_discovery_last_crystallise_ep': int(agent.goal_discovery._last_crystallise_ep),
        }

    torch.save(weights, pt_path)

    lexicon_data    = {str(i): e.to_dict() for i, e in channel.lexicon.items()}
    plateau_history = {aid: list(h) for aid, h in plateau_mon.history.items()}

    meta = {
        'episode':             episode,
        'generation':          generation,
        'last_replication_ep': last_replication_ep,
        'all_ids_ever':        all_ids_ever,
        'active_agents':       list(agents.keys()),
        'agent_meta':          agent_meta,
        'channel': {
            'total_signals':    channel.total_signals,
            'lexicon':          lexicon_data,
            'compound_lexicon': channel.compound_lexicon.to_list(),
            'sequence_lexicon': channel.sequence_lexicon.to_list(),
        },
        'plateau_history': plateau_history,
        'pt_file':         os.path.basename(pt_path),
        'timestamp':       datetime.now().isoformat()
    }

    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(SAVE_DIR, 'latest.json'), 'w') as f:
        json.dump({'episode': episode, 'json': json_path, 'pt': pt_path}, f)

    print(f'  [Save] Episode {episode:,} → {json_path}')
    return json_path


def _load_latest():
    ptr_path = os.path.join(SAVE_DIR, 'latest.json')
    if not os.path.exists(ptr_path):
        return None, None

    with open(ptr_path) as f:
        ptr = json.load(f)

    json_path = ptr['json']
    pt_path   = ptr.get('pt')

    if not pt_path or not pt_path.endswith('.pt'):
        return None, None

    if not os.path.exists(json_path) or not os.path.exists(pt_path):
        return None, None

    with open(json_path) as f:
        meta = json.load(f)

    return meta, pt_path


def prompt_resume():
    meta, pt_path = _load_latest()
    if meta is None:
        return None, None

    print()
    print('  ' + '─' * 58)
    print(f'  SAVE FOUND — episode {meta["episode"]:,}  gen {meta["generation"]}')
    print(f'  Agents  : {", ".join(meta["active_agents"])}')
    print(f'  Saved   : {meta["timestamp"][:19]}')
    print('  ' + '─' * 58)

    while True:
        try:
            r = input('  Resume from save? [Y/n]: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            r = 'n'
        if r in ('', 'y', 'yes'):
            print('  Resuming.\n')
            return meta, pt_path
        if r in ('n', 'no'):
            print('  Starting fresh.\n')
            return None, None
        print('  Please enter Y or N.')


def restore(meta, pt_path, shared_replay=None):
    from agent import ARIAAgent, EpisodicMemory, SubGoal
    from communication import CommunicationChannel, CompoundLexicon, SequenceLexicon

    weights = torch.load(pt_path, weights_only=True)

    agents = {}
    for agent_id in meta['active_agents']:
        key = agent_id.replace('-', '_')
        am  = meta['agent_meta'][agent_id]

        # Restore cultural memory
        cm_data = am.get('cultural_memory', [])
        cultural_memory = EpisodicMemory.from_list(cm_data) if cm_data else None

        # Restore sub-goal
        sg_data  = am.get('sub_goal')
        sub_goal = SubGoal.from_dict(sg_data) if sg_data else None

        agent = ARIAAgent(
            agent_id,
            learning_rate   = float(am.get('learning_rate',   LEARNING_RATE)),
            epsilon_decay   = float(am.get('epsilon_decay',   EPSILON_DECAY)),
            intrinsic_beta  = float(am.get('intrinsic_beta',  INTRINSIC_BETA)),
            episodic_beta   = float(am.get('episodic_beta',   EPISODIC_BETA)),
            hidden_size     = int(am.get('hidden_size',       HIDDEN_SIZE)),
            n_layers        = int(am.get('n_layers',          N_LAYERS_DEFAULT)),
            activation      = am.get('activation',            'relu'),
            use_skip        = bool(am.get('use_skip',         False)),
            energy          = float(am.get('energy',          100.0)),
            drain_rate      = float(am.get('drain_rate',      1.0)),
            online_state_dict = weights.get(key),
            cultural_memory = cultural_memory,
            sub_goal        = sub_goal,
            replay          = shared_replay,
        )
        agent.target_net.load_state_dict(agent.online_net.state_dict())

        wm_key = f'{key}_wm'
        if wm_key in weights:
            try:
                agent.world_model.load_state_dict(weights[wm_key])
            except Exception:
                pass  # input-size mismatch — world model starts fresh

        tom_key = f'{key}_tom'
        if tom_key in weights:
            try:
                agent.tom_model.load_state_dict(weights[tom_key])
            except Exception:
                pass  # architecture mismatch — ToM starts fresh

        agent.epsilon                = float(am['epsilon'])
        agent.total_reward           = float(am['total_reward'])
        agent.episodes               = int(am['episodes'])
        agent._tom_steps             = int(am.get('tom_steps', 0))
        agent._steps                 = int(am.get('steps', 0))
        agent._wm_steps              = int(am.get('wm_steps', 0))
        agent.reputation             = dict(am.get('reputation', {}))
        agent.coord_reward_total     = float(am.get('coord_reward_total', 0.0))
        agent.currency_reward_total  = float(am.get('currency_reward_total', 0.0))
        agent.goal_discovery._last_crystallise_ep = int(
            am.get('goal_discovery_last_crystallise_ep', 0))
        agents[agent_id]        = agent

    channel               = CommunicationChannel(append_log=True)
    channel.total_signals = meta['channel']['total_signals']
    for sig_str, ed in meta['channel']['lexicon'].items():
        entry                 = channel.lexicon[int(sig_str)]
        entry.symbol          = ed['symbol']
        entry.use_count       = ed['use_count']
        entry.coord_successes = ed['coord_successes']
        entry.assigned        = ed['assigned']
        if entry.assigned and entry.symbol in channel._symbol_pool:
            channel._symbol_pool.remove(entry.symbol)

    compound_data = meta['channel'].get('compound_lexicon', [])
    if compound_data:
        channel.compound_lexicon = CompoundLexicon.from_list(compound_data)

    sequence_data = meta['channel'].get('sequence_lexicon', [])
    if sequence_data:
        channel.sequence_lexicon = SequenceLexicon.from_list(sequence_data)

    return (
        agents,
        channel,
        meta['generation'],
        meta['last_replication_ep'],
        meta['all_ids_ever'],
        meta['plateau_history'],
        meta['episode'] + 1
    )
